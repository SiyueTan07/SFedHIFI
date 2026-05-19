"""
SIFS smoke test (no SpikingJelly required for the core logic checks).

Run:  python3 scripts/smoke_test_sifs.py

Verifies:
  1. polynomial_sparsity schedule monotonicity and clamping
  2. mask_aware_aggregate divides element-wise correctly
  3. update_mask respects the requested target_sparsity within +/-1 element
  4. SATR component scores and component masks are well-formed
  5. crisis_loss returns 0 when rates are above the floor
"""

import os
import sys
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from model.sifs_utils import (  # noqa: E402
    polynomial_sparsity,
    mask_aware_aggregate,
    update_mask,
    compute_satr_component_score,
    component_mask_from_scores,
    component_live_ratio,
    SpikeSilenceTracker,
    crisis_loss,
)


def test_polynomial_schedule():
    vals = [
        polynomial_sparsity(t, warmup=5, final_round=20,
                            init_sparsity=0.0, final_sparsity=0.5)
        for t in range(0, 25)
    ]
    assert vals[0] == 0.0 and vals[5] == 0.0
    assert vals[20] == 0.5 and vals[24] == 0.5
    # monotone non-decreasing
    for a, b in zip(vals, vals[1:]):
        assert b + 1e-9 >= a, vals
    print("[OK] polynomial_sparsity schedule monotonic and clamped.")


def test_mask_aware_aggregate():
    w1 = torch.tensor([1.0, 2.0, 3.0])
    w2 = torch.tensor([10.0, 0.0, 30.0])
    m1 = torch.tensor([1.0, 1.0, 0.0])
    m2 = torch.tensor([1.0, 0.0, 1.0])
    out = mask_aware_aggregate([w1 * m1, w2 * m2], [m1, m2])
    # expected: [ (1+10)/2, 2/1, 30/1 ]
    assert torch.allclose(out, torch.tensor([5.5, 2.0, 30.0]), atol=1e-6), out
    print("[OK] mask_aware_aggregate matches expected element-wise division.")


def test_update_mask_target_sparsity():
    torch.manual_seed(0)
    w = torch.randn(4, 8, 3, 3)
    g = torch.randn_like(w)
    rate = torch.linspace(0.0, 0.5, 8)  # first channels silent
    mask = torch.ones_like(w)
    new_mask = update_mask(
        mask, w, g, presyn_rate=rate,
        target_sparsity=0.5,
        rebirth_ratio=0.0,
        silence_weight=10.0,
        taylor_weight=1.0,
        silence_threshold=0.1,
    )
    live = new_mask.sum().item()
    target = int(round(0.5 * mask.numel()))
    assert abs(live - target) <= 1, (live, target)
    # silent channels (low rate) should be pruned more than active ones
    silent_live = new_mask[:, 0, :, :].sum().item()
    active_live = new_mask[:, -1, :, :].sum().item()
    assert silent_live < active_live, (silent_live, active_live)
    print(f"[OK] update_mask hit target ({live}/{mask.numel()}) and prunes silent channels first "
          f"(silent_live={silent_live}, active_live={active_live}).")


def test_satr_component_scores():
    torch.manual_seed(1)
    w = torch.randn(6, 4, 3, 3)
    ref = w + 0.05 * torch.randn_like(w)
    rate = torch.tensor([0.0, 0.02, 0.1, 0.5])
    for mode in ["full", "random", "magnitude", "taylor", "spike", "no_norm"]:
        score = compute_satr_component_score(
            w,
            reference_weight=ref,
            presyn_rate=rate,
            mode=mode,
            silence_threshold=1e-3,
        )
        assert score.shape == (6,), (mode, score.shape)
        assert torch.isfinite(score).all(), (mode, score)
    score = compute_satr_component_score(w, reference_weight=ref, presyn_rate=rate, mode="magnitude")
    mask = component_mask_from_scores(w, score, retain_ratio=0.5)
    assert mask.shape == w.shape, mask.shape
    comp = mask.view(mask.shape[0], -1).mean(dim=1)
    assert int((comp > 0).sum().item()) == 3, comp
    for k in range(mask.shape[0]):
        assert torch.all(mask[k] == mask[k].reshape(-1)[0]), "component mask must be broadcast over component rows"
    assert abs(component_live_ratio(mask) - 0.5) < 1e-6
    print("[OK] SATR component scores and Top-K component masks are well-formed.")


def test_crisis_loss():
    tracker = SpikeSilenceTracker()
    # inject fake stats
    tracker._sums = {'layer1': torch.tensor([0.3, 0.4, 0.5])}
    tracker._counts['layer1'] = 1
    cl_zero = crisis_loss(tracker, floor=0.1, weight=1.0)
    assert cl_zero.item() == 0.0, cl_zero
    # now silent
    tracker._sums = {'layer1': torch.tensor([0.01, 0.0, 0.0])}
    tracker._counts['layer1'] = 1
    cl_pos = crisis_loss(tracker, floor=0.1, weight=1.0)
    assert cl_pos.item() > 0.0, cl_pos
    print(f"[OK] crisis_loss=0 when rates ok, >0 when silent (got {cl_pos.item():.4f}).")


def main():
    test_polynomial_schedule()
    test_mask_aware_aggregate()
    test_update_mask_target_sparsity()
    test_satr_component_scores()
    test_crisis_loss()
    print("\nALL SIFS/SATR UTILITY SMOKE TESTS PASSED.")


if __name__ == "__main__":
    main()
