"""
@file: sifs_utils.py
@brief: Spike-Induced Federated Sparsity (SIFS) utilities.

This module implements the four building blocks of the SIFS pipeline:

  1. SpikeSilenceTracker      -- per-layer, per-channel mean spike rate via
                                 SpikingJelly forward hooks on LIF neurons.
  2. compute_sifs_importance  -- I_Prune = lambda_s * silence + lambda_t * |w*grad|
                                 I_Grow  = |grad|  on currently-zero positions.
  3. update_mask              -- magnitude-vs-silence prune & rebirth grow.
  4. polynomial_sparsity      -- s(t) schedule (Zhu & Gupta 2017 style).
  5. mask_aware_aggregate     -- federated averaging that divides each element
                                 by the number of clients whose mask kept it,
                                 instead of the (uniform) number of clients.
  6. crisis_loss              -- soft hinge that prevents global silence.

Design notes:

  * The SIFS mask lives on the *filter_bank* parameters of every BasicBlock
    in spiking_resnet_flanc.py.  These are the only learnable Conv weights
    that are actually shared across capacity tiers in SFedHIFI/FLANC,
    so they are the natural target for federated sparsity.
  * Silence is measured on the *output* spike train of each LIF neuron,
    averaged over T and the spatial-batch dims, and then *broadcast* to the
    corresponding filter_bank's channel dim (basis_size axis).  This gives
    a per-input-channel silence score, which is what the BPTT proposition
    in the paper (Sec. 3) calls s_i(t).
  * All operations are torch only, no numpy round-trips on the hot path.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn

try:
    from spikingjelly.activation_based import neuron as sj_neuron
    _SJ_LIF_CLASSES = (sj_neuron.LIFNode, sj_neuron.IFNode, sj_neuron.ParametricLIFNode)
except Exception:  # pragma: no cover - SpikingJelly is a runtime dependency
    _SJ_LIF_CLASSES = tuple()


# ---------------------------------------------------------------------------
# 0. Shared score helpers
# ---------------------------------------------------------------------------

def _zscore(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Return a numerically stable z-score over a 1-D component score."""
    x = x.float()
    if x.numel() <= 1:
        return torch.zeros_like(x)
    return (x - x.mean()) / (x.std(unbiased=False) + eps)


def _minmax(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Return min-max normalised scores in [0, 1]."""
    x = x.float()
    return (x - x.min()) / (x.max() - x.min() + eps)


# ---------------------------------------------------------------------------
# 1. Spike silence tracking
# ---------------------------------------------------------------------------
class SpikeSilenceTracker:
    """Accumulate per-LIF-layer mean spike rate over a local training round.

    The tracker attaches a forward hook to every spiking neuron in the model.
    On each forward call the hook computes the mean activation over
    (T, batch, spatial) for every output channel and updates a running sum.

    At the end of a round, :meth:`mean_rates` returns a dict
    ``{layer_name: tensor[C]}`` of per-channel spike rates in [0, 1].
    """

    def __init__(self, lif_classes: Tuple[type, ...] = _SJ_LIF_CLASSES) -> None:
        self.lif_classes = lif_classes
        self._sums: Dict[str, torch.Tensor] = {}
        self._counts: Dict[str, int] = defaultdict(int)
        self._hooks: List[torch.utils.hooks.RemovableHandle] = []

    # --- registration ------------------------------------------------------
    def attach(self, model: nn.Module) -> "SpikeSilenceTracker":
        """Register hooks on every spiking neuron in *model*."""
        self.detach()
        for name, module in model.named_modules():
            if isinstance(module, self.lif_classes):
                hook = module.register_forward_hook(self._make_hook(name))
                self._hooks.append(hook)
        return self

    def detach(self) -> None:
        for h in self._hooks:
            try:
                h.remove()
            except Exception:
                pass
        self._hooks = []

    def reset(self) -> None:
        self._sums.clear()
        self._counts.clear()

    # --- hook --------------------------------------------------------------
    def _make_hook(self, name: str):
        def _hook(_module, _inputs, output):
            if not torch.is_tensor(output):
                return
            # SpikingJelly multi-step output: (T, B, C, H, W) or (T, B, C)
            with torch.no_grad():
                if output.dim() >= 3:
                    # channel dim is the third axis under multi-step layout
                    reduce_dims = [d for d in range(output.dim()) if d != 2]
                    rate = output.float().mean(dim=reduce_dims)  # (C,)
                else:
                    rate = output.float().mean(dim=tuple(range(output.dim() - 1)))
                rate = rate.detach().cpu()
                if name in self._sums:
                    self._sums[name] += rate
                else:
                    self._sums[name] = rate.clone()
                self._counts[name] += 1
        return _hook

    # --- query -------------------------------------------------------------
    def mean_rates(self) -> Dict[str, torch.Tensor]:
        out = {}
        for k, s in self._sums.items():
            c = max(1, self._counts[k])
            out[k] = s / c
        return out


# ---------------------------------------------------------------------------
# 2. Importance scores  (Sec. 3.2 in the paper)
# ---------------------------------------------------------------------------
def compute_sifs_importance(
    weight: torch.Tensor,
    grad: torch.Tensor,
    presyn_rate: Optional[torch.Tensor],
    *,
    silence_weight: float = 10.0,
    taylor_weight: float = 1.0,
    silence_threshold: float = 1e-3,
) -> torch.Tensor:
    """SIFS prune-importance score I_Prune.

    Lower score => safer to prune.

    Score = silence_weight * f(rate)  +  taylor_weight * |w * grad|

    where  f(rate) = max(0, threshold - rate) / threshold  in [0, 1]
    is high when the presynaptic channel is silent (exact 0 BPTT grad),
    low when it fires often.  The Taylor magnitude term breaks ties.
    """
    score = torch.zeros_like(weight)

    if presyn_rate is not None and silence_weight != 0.0:
        rate = presyn_rate.to(weight.device).clamp_(min=0.0)
        f = torch.clamp(silence_threshold - rate, min=0.0) / max(silence_threshold, 1e-12)
        # weight shape: (n_basis, basis_size, k, k); broadcast over basis_size
        view = [1] * weight.dim()
        if f.numel() == weight.shape[1]:
            view[1] = -1
        elif f.numel() == weight.shape[0]:
            view[0] = -1
        else:
            # fall back: per-tensor scalar silence
            f = f.mean().expand(1)
            view[0] = -1
        # Lower score => prune, so silent positions get *low* score.
        # We want silent => low importance, so use (1 - f) here.
        score = score + silence_weight * (1.0 - f.view(view))
    else:
        score = score + silence_weight  # uniform high score => prefer Taylor

    if taylor_weight != 0.0:
        score = score + taylor_weight * (weight.detach() * grad.detach()).abs()
    return score


def compute_grow_score(grad: torch.Tensor) -> torch.Tensor:
    """I_Grow = |grad| on currently-zero positions (RigL-style)."""
    return grad.detach().abs()


# ---------------------------------------------------------------------------
# 3. Mask update (prune + rebirth grow)
# ---------------------------------------------------------------------------
def update_mask(
    mask: torch.Tensor,
    weight: torch.Tensor,
    grad: torch.Tensor,
    presyn_rate: Optional[torch.Tensor],
    *,
    target_sparsity: float,
    rebirth_ratio: float,
    silence_weight: float,
    taylor_weight: float,
    silence_threshold: float,
) -> torch.Tensor:
    """Return a new mask of identical shape that respects *target_sparsity*.

    The procedure is:

      a) compute current number of live elements n_live and the desired
         n_target = round((1 - target_sparsity) * mask.numel());
      b) prune the lowest-importance LIVE elements down to
         n_target  (Prune step);
      c) regrow (rebirth_ratio * (mask.numel() - n_target)) DEAD elements
         with the largest |grad|, then re-prune so that the final count
         stays at n_target.
    """
    numel = mask.numel()
    n_target = max(1, int(round((1.0 - target_sparsity) * numel)))

    score_prune = compute_sifs_importance(
        weight, grad, presyn_rate,
        silence_weight=silence_weight,
        taylor_weight=taylor_weight,
        silence_threshold=silence_threshold,
    )
    score_grow = compute_grow_score(grad)

    flat_mask = mask.view(-1).clone()
    flat_p = score_prune.view(-1)
    flat_g = score_grow.view(-1)

    live_idx = (flat_mask > 0).nonzero(as_tuple=True)[0]
    dead_idx = (flat_mask == 0).nonzero(as_tuple=True)[0]

    # --- Prune ------------------------------------------------------------
    # Mark everything dead first, then keep the top n_target live by I_Prune.
    new_mask = torch.zeros_like(flat_mask)
    if live_idx.numel() > 0:
        keep = min(n_target, live_idx.numel())
        topk_vals, topk_pos = torch.topk(flat_p[live_idx], keep, largest=True)
        new_mask[live_idx[topk_pos]] = 1.0

    # --- Grow (rebirth) --------------------------------------------------
    n_dead_target = numel - n_target
    n_grow = int(round(rebirth_ratio * n_dead_target))
    if n_grow > 0 and dead_idx.numel() > 0:
        n_grow = min(n_grow, dead_idx.numel())
        topk_vals, topk_pos = torch.topk(flat_g[dead_idx], n_grow, largest=True)
        grow_positions = dead_idx[topk_pos]
        new_mask[grow_positions] = 1.0
        # Re-prune to exactly n_target by removing the weakest current live
        live_after = (new_mask > 0).nonzero(as_tuple=True)[0]
        overflow = live_after.numel() - n_target
        if overflow > 0:
            scores_after = flat_p[live_after].clone()
            # grow positions: give them a small boost so they survive the trim
            grow_set = set(grow_positions.tolist())
            for i, idx in enumerate(live_after.tolist()):
                if idx in grow_set:
                    scores_after[i] = scores_after[i] + 1e-6
            _, weakest = torch.topk(scores_after, overflow, largest=False)
            new_mask[live_after[weakest]] = 0.0

    return new_mask.view_as(mask)


# ---------------------------------------------------------------------------
# 4. SATR component scores and component-wise masks
# ---------------------------------------------------------------------------
def compute_satr_component_score(
    weight: torch.Tensor,
    *,
    reference_weight: Optional[torch.Tensor] = None,
    presyn_rate: Optional[torch.Tensor] = None,
    mode: str = "full",
    silence_threshold: float = 1e-3,
    spike_weight: float = 1.0,
    taylor_weight: float = 1.0,
    normalise: bool = True,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Compute one score per FLANC filter-bank component.

    Args:
        weight: filter bank of shape ``(n_basis, basis_size, k, k)``.
        reference_weight: previous server value.  When available, the local
            update ``weight - reference_weight`` is used as a first-order
            gradient proxy for the Taylor component score.
        presyn_rate: optional firing rates.  If its length equals
            ``basis_size`` it is interpreted as presynaptic channel activity;
            otherwise a scalar mean activity fallback is used.
        mode: one of ``full``, ``random``, ``magnitude``, ``taylor``,
            ``spike``, ``no_norm``/``no-normalisation``.
        normalise: z-score normalise spike/Taylor/magnitude terms before
            combining.  ``mode='no_norm'`` deliberately disables this.

    Returns:
        Tensor of shape ``(n_basis,)``.  Larger means more important.
    """
    if weight.dim() < 2:
        raise ValueError("SATR expects a componentised filter bank tensor.")

    mode = (mode or "full").replace("-", "_").lower()
    n_basis = weight.shape[0]
    flat = weight.detach().float().view(n_basis, -1)

    if mode == "random":
        return torch.rand(n_basis, generator=generator, device=weight.device)

    magnitude = flat.norm(p=2, dim=1)

    if reference_weight is not None and reference_weight.shape == weight.shape:
        ref = reference_weight.detach().float().to(weight.device)
        update = weight.detach().float() - ref
        taylor = (ref.view(n_basis, -1) * update.view(n_basis, -1)).sum(dim=1).abs()
    else:
        # Fallback used when no previous global snapshot is available.
        taylor = (flat * flat).sum(dim=1).abs()

    # Spike utilisation: fraction of component energy lying on active channels.
    channel_energy = weight.detach().float().pow(2).sum(dim=tuple(range(2, weight.dim())))
    if presyn_rate is not None and presyn_rate.numel() > 0:
        rate = presyn_rate.detach().float().to(weight.device).clamp(min=0.0)
        if rate.numel() == channel_energy.shape[1]:
            active = 1.0 - torch.clamp(silence_threshold - rate, min=0.0) / max(silence_threshold, 1e-12)
            spike = (channel_energy * active.view(1, -1)).sum(dim=1) / (channel_energy.sum(dim=1) + 1e-12)
        elif rate.numel() == n_basis:
            active = 1.0 - torch.clamp(silence_threshold - rate, min=0.0) / max(silence_threshold, 1e-12)
            spike = active[:n_basis].to(weight.device)
        else:
            active_scalar = 1.0 - torch.clamp(silence_threshold - rate.mean(), min=0.0) / max(silence_threshold, 1e-12)
            spike = torch.full((n_basis,), float(active_scalar.item()), device=weight.device)
    else:
        spike = torch.ones(n_basis, device=weight.device)

    if mode == "magnitude":
        return _zscore(magnitude) if normalise else magnitude
    if mode == "taylor":
        return _zscore(taylor) if normalise else taylor
    if mode == "spike":
        return _zscore(spike) if normalise else spike

    if mode in ("no_norm", "no_normalisation", "no_normalization"):
        return spike_weight * spike + taylor_weight * taylor

    # Default SATR: normalised spike + normalised Taylor.
    if normalise:
        return spike_weight * _zscore(spike) + taylor_weight * _zscore(taylor)
    return spike_weight * spike + taylor_weight * taylor


def component_mask_from_scores(
    weight: torch.Tensor,
    scores: torch.Tensor,
    *,
    retain_ratio: float,
) -> torch.Tensor:
    """Build a broadcast component mask from Top-K component scores.

    The returned mask has the same shape as ``weight`` and is constant over all
    non-component dimensions, so component ``k`` is either fully active or
    fully frozen.
    """
    n_basis = weight.shape[0]
    retain_ratio = float(max(0.0, min(1.0, retain_ratio)))
    k_keep = max(1, min(n_basis, int(round(retain_ratio * n_basis))))
    flat_scores = scores.detach().float().view(-1).to(weight.device)
    if flat_scores.numel() != n_basis:
        raise ValueError(f"Expected {n_basis} component scores, got {flat_scores.numel()}.")
    _, idx = torch.topk(flat_scores, k_keep, largest=True)
    comp = torch.zeros(n_basis, device=weight.device, dtype=weight.dtype)
    comp[idx] = 1.0
    view = [n_basis] + [1] * (weight.dim() - 1)
    return comp.view(view).expand_as(weight).clone()


def component_live_ratio(mask: torch.Tensor) -> float:
    """Return fraction of live components for a broadcast component mask."""
    if mask.numel() == 0:
        return 0.0
    comp = mask.detach().float().view(mask.shape[0], -1).mean(dim=1)
    return float((comp > 0).float().mean().item())


# ---------------------------------------------------------------------------
# 5. Sparsity schedule  (Zhu & Gupta 2017; used by FedDST)
# ---------------------------------------------------------------------------
def polynomial_sparsity(
    current_round: int,
    *,
    warmup: int,
    final_round: int,
    init_sparsity: float,
    final_sparsity: float,
    power: float = 3.0,
) -> float:
    """Polynomial decay sparsity schedule.

    s(t) = s_f + (s_0 - s_f) * (1 - (t - warmup) / (final_round - warmup)) ** power
    Clamped to [s_0, s_f] outside [warmup, final_round].
    """
    if current_round <= warmup:
        return init_sparsity
    if current_round >= final_round:
        return final_sparsity
    progress = (current_round - warmup) / max(1, final_round - warmup)
    frac = (1.0 - progress) ** power
    return final_sparsity + (init_sparsity - final_sparsity) * frac


# ---------------------------------------------------------------------------
# 6. Mask-aware federated aggregation
# ---------------------------------------------------------------------------
def mask_aware_aggregate(
    tensor_list: List[torch.Tensor],
    mask_list: List[torch.Tensor],
    eps: float = 1e-12,
) -> torch.Tensor:
    """Aggregate masked parameter copies across clients.

    Standard FedAvg divides by N (the number of clients).  When clients carry
    different masks, that under-counts entries kept by only a few clients.
    SIFS aggregator divides element-wise by the number of clients whose mask
    actually *kept* that entry.  Entries that no client kept stay zero.

    Args:
        tensor_list: list of K tensors with identical shape (already masked,
                     i.e. tensor[i] == 0 wherever mask[i] == 0).
        mask_list:   list of K binary masks with identical shape.

    Returns:
        Aggregated tensor of the same shape.
    """
    assert len(tensor_list) == len(mask_list) and len(tensor_list) > 0
    stacked_w = torch.stack(tensor_list, dim=0)
    stacked_m = torch.stack(mask_list, dim=0).to(stacked_w.dtype)
    num = (stacked_w * stacked_m).sum(dim=0)
    den = stacked_m.sum(dim=0)
    return num / (den + eps)


# ---------------------------------------------------------------------------
# 7. Spike-rate crisis regularizer (prevents catastrophic global silence)
# ---------------------------------------------------------------------------
def crisis_loss(
    silence_tracker: "SpikeSilenceTracker",
    *,
    floor: float = 0.02,
    weight: float = 1.0,
) -> torch.Tensor:
    """soft hinge loss = weight * sum_layer ReLU(floor - mean_rate_layer)^2.

    Used to discourage the optimizer from driving the whole network silent
    in order to satisfy the prune signal.
    """
    if weight == 0.0:
        return torch.zeros(1, requires_grad=False)
    rates = silence_tracker.mean_rates()
    if not rates:
        return torch.zeros(1, requires_grad=False)
    total = 0.0
    for r in rates.values():
        m = r.mean().item()
        deficit = max(0.0, floor - m)
        total += deficit * deficit
    return torch.tensor(weight * total)


# ---------------------------------------------------------------------------
# 8. Helpers to iterate over SIFS/SATR-managed parameters
# ---------------------------------------------------------------------------
def iter_sifs_parameters(model: nn.Module) -> Iterable[Tuple[str, nn.Parameter, torch.Tensor]]:
    """Yield (full_param_name, param, mask_buffer) triples.

    A parameter is SIFS-managed iff its owner module exposes a buffer named
    ``mask_<param_local_name>`` registered by :func:`register_sifs_masks`.
    """
    for mod_name, module in model.named_modules():
        for p_name, param in module.named_parameters(recurse=False):
            buf_name = f"mask_{p_name}"
            if hasattr(module, buf_name):
                full = f"{mod_name}.{p_name}" if mod_name else p_name
                yield full, param, getattr(module, buf_name)


def register_sifs_masks(
    model: nn.Module,
    param_match: Tuple[str, ...] = ("filter_bank_1", "filter_bank_2"),
) -> int:
    """Attach a ``mask_<name>`` buffer (ones_like) to every matching param.

    Returns the number of masks created.
    """
    n = 0
    for module in model.modules():
        for p_name, param in list(module.named_parameters(recurse=False)):
            if p_name in param_match:
                buf_name = f"mask_{p_name}"
                if not hasattr(module, buf_name):
                    module.register_buffer(buf_name, torch.ones_like(param.data))
                    n += 1
    return n


def apply_sifs_masks(model: nn.Module) -> None:
    """In-place: param.data *= mask_buffer for every SIFS-managed param."""
    with torch.no_grad():
        for _, param, mask in iter_sifs_parameters(model):
            param.data.mul_(mask.to(param.data.dtype).to(param.data.device))


def gather_sifs_state(model: nn.Module) -> Dict[str, Dict[str, torch.Tensor]]:
    """Return ``{param_name: {"weight": ..., "mask": ...}}`` (CPU tensors)."""
    out: Dict[str, Dict[str, torch.Tensor]] = {}
    for full, param, mask in iter_sifs_parameters(model):
        out[full] = {
            "weight": param.data.detach().clone().cpu(),
            "mask": mask.detach().clone().cpu(),
        }
    return out
