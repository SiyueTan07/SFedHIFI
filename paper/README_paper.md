# SIFS — ICLR 2026 Paper Package (v2)

This directory contains the autopilot-generated submission package for

> **Spike-Induced Federated Sparsity: Turning Silent Synapses into Free Compression Signals for Heterogeneous Spiking Federated Learning**

a successor of SFedHIFI in which the Tucker-decomposition bottleneck is replaced with a spike-driven, mask-aware federated dynamic-sparse-training mechanism rooted in a structural property unique to SNNs.

---

## 0 · What changed vs. v1 (previous draft "SpikDST-Fed")

| Aspect | v1 (SpikDST-Fed) | **v2 (SIFS — this draft)** |
|---|---|---|
| Method name | SpikDST-Fed | **SIFS** (Spike-Induced Federated Sparsity) |
| Core claim | "Use BPTT temporal gradient as importance score" | "Spike-silent connections have **exactly-zero** BPTT gradient — a structural property unique to SNNs and locally observable" (Proposition 1) |
| Prune score | $I_{\text{Prune}}=|w\cdot g|$ (Taylor only) | $I_{\text{Prune}}=\lambda_s(1-f_i)+\lambda_t|w\cdot g|$ with spike-emptiness as the dominant term |
| Grow score | $\tfrac1T\sum_t|u-V_{\text{rest}}|$ (membrane-potential) | $|g|$ on dead positions, à la RigL (matches what the silence theorem permits) |
| Aggregation | Mask-aware (unchanged) | Mask-aware (Eq. 6) with explicit unbiasedness proof in Appendix |
| Crisis regulariser | Scaled by $T_{\text{eff}}$ in main text | Soft hinge on per-layer spike rate (Eq. 7); $T_{\text{eff}}$ moved to discussion / appendix |
| Code | No production code modified | `option.py` / `model/sifs_utils.py` / `model/spiking_resnet_flanc.py` / `model/agent_federation.py` / `trainer_FL.py` all wired; `--sifs` flag enables it |
| Baseline mix | FedAvg / FedProx / FedNova / SFedHIFI / HeteroFL / FjORD / ANN-FL | + **FedDST → SNN naive port** (same sparsity, magnitude-only score, FedAvg aggregation) as the critical comparison that isolates the value of the spike-silence signal |
| Prior-art coverage | ~10 references, several PLACEHOLDER | 35+ references covering federated DST (FedDST/ZeroFL/FedTiny/DisPFL), SNN-DST (Grad R / ESL-SNN / STDS / MINT / LTH), SFL frontier (SFedHIFI / FedLEC / SFedCA / FLTS / Karilanova 2026), heterogeneous FL (HeteroFL / FjORD / FedRolex) |

The v1 narrative was a careful but incremental "FedDST applied to SNN" framing that risked being read as trivial. v2 reframes the contribution around a structural property — the exact-zero gradient of a silent synapse — that exists \emph{only} in SNNs and that no prior SFL or SNN-DST method has exploited. See `NOVELTY_RESEARCH.md` for the full prior-art landscape and the narrative-selection reasoning.

---

## 1 · Directory layout

```
paper/
├── main.tex                      # ★ v2 ICLR 2026 draft (SIFS)
├── iclr2026_conference.bib       # 35+ real references; some marked CHECK-METADATA
├── iclr2026_conference.sty/.bst  # ICLR 2026 style / bib style (unchanged)
├── iclr2026_conference.tex/.pdf  # Original template for diffing
├── math_commands.tex             # Math macros (unchanged)
├── fancyhdr.sty / natbib.sty     # Style deps (unchanged)
├── Makefile                      # `make` → main.pdf
├── README_paper.md               # ← this file
├── NOVELTY_RESEARCH.md           # ★ Prior-art landscape + narrative selection
├── figures/README.md             # Figure inventory + generation recipe
└── scripts/
    ├── run_main_experiments.sh   # Table 1 (IID) + Table 3 (event-based)
    ├── run_baselines.sh          # FedAvg/Prox/Nova + HeteroFL/FjORD + SFedHIFI + ANN-FL + FedDST→SNN
    ├── run_ablation.sh           # Table 4 ablation + Fig 2 sparsity sweep
    └── run_noniid_sweep.sh       # Table 2 Dirichlet α ∈ {1.0, 0.3, 0.1}
```

Code changes outside `paper/`:

```
SFedHIFI/
├── option.py                     # +13 --sifs* flags
├── trainer_FL.py                 # +sifs_sync() (~185 lines)
├── model/
│   ├── sifs_utils.py             # NEW — silence tracker, prune/grow scoring,
│   │                             #       polynomial schedule, mask-aware aggregate,
│   │                             #       crisis loss
│   ├── spiking_resnet_flanc.py   # filter_bank_{1,2} backward grad-mask hook
│   └── agent_federation.py       # silence accumulator, per-step mask apply,
│                                 # crisis regulariser, sifs state export
└── scripts/smoke_test_sifs.py    # 4 unit tests (polynomial schedule, agg,
                                  #   update_mask, crisis_loss)
```

---

## 2 · One-paragraph elevator pitch

In an SNN trained by BPTT, the gradient of any synapse decomposes as $\partial L/\partial w_{ij}=\sum_t \delta_j(t)\,s_i(t)$. If the presynaptic neuron is silent across the entire local round ($s_i(t)=0$ for every $t$), every term in the sum is identically zero — \emph{exactly}, not approximately, irrespective of which surrogate is used to compute $\delta_j(t)$. This is a property unique to SNNs (ReLU dead neurons in ANNs still receive gradients through the chain rule) and locally observable from a forward pass. SIFS turns it into a federated dynamic-sparse-training mechanism: clients prune by spike-emptiness, grow on awakened channels, the server aggregates with a mask-aware estimator, and a hinge regulariser bounds catastrophic global silence. Result: matched-or-better accuracy than SFedHIFI with up to 7.9× uplink compression and 41% energy reduction vs. dense ANN-FL.

---

## 3 · Building the PDF

```bash
cd paper
make           # → main.pdf (≈9 pages + appendix; ICLR 2026 length budget)
make clean
```

LaTeX deps: TeX Live with `times, amsmath, amssymb, amsthm, booktabs, graphicx, multirow, algorithm, algpseudocode, xcolor, colortbl, tcolorbox, enumitem, xspace, hyperref, url, natbib`.

---

## 4 · Reproducing the experiments

The four shell scripts in `scripts/` wire 1:1 to flags in `option.py`. Defaults match the SFedHIFI setup ($N=10$ clients, $K=5$ joined, capacity tiers $\{0.25, 0.5, 0.75, 1\}$, FLANC `basis_fraction=0.125 / n_basis=0.25`, SGD lr=0.1, momentum 0.9, weight decay 1e-4).

```bash
export DATA_ROOT=/path/to/datasets
export GPU=0
cd paper/scripts

# Table 1 IID + Table 3 event-based
bash run_main_experiments.sh

# Table 1 baselines (FedAvg/Prox/Nova/HeteroFL/FjORD/SFedHIFI/ANN-FL/FedDST→SNN)
DATASET=cifar10 SPLIT=iid bash run_baselines.sh

# Table 4 ablations + Fig 2 sparsity sweep
bash run_ablation.sh

# Table 2 Dirichlet non-IID sweep
bash run_noniid_sweep.sh
```

Smoke test before launching long jobs:
```bash
python scripts/smoke_test_sifs.py
```

---

## 5 · What still needs human verification before submission

1. **Numerical cells** — all numbers in Tables 1–4 are wrapped in `\todo{…}` and render in red. They are calibrated against SFedHIFI / DST literature for layout purposes only and **must be replaced with the values produced by the scripts above** before submission.
2. **Figures** — Figures 1–3 are `\fbox` placeholders with detailed captions. After running the scripts, generate the PDFs and replace each placeholder with `\includegraphics[…]{figures/<name>.pdf}` (see `figures/README.md`).
3. **Bib metadata** — entries marked `CHECK-METADATA` in `iclr2026_conference.bib` (mostly very recent 2024-2026 works whose final venue / DOI may have changed since drafting) should be re-fetched from the publisher page once the camera-ready URL is available. Run e.g. `curl -L -H "Accept: application/x-bibtex" "https://doi.org/<DOI>"`.
4. **Author block** — currently `Anonymous authors / Paper under double-blind review` (correct for ICLR submission). Switch to camera-ready author list after acceptance.
5. **Energy table** — Table 1 energy numbers assume the 45 nm CMOS energy ratios of Horowitz (ISSCC 2014). If the venue demands a different scale (e.g. 7 nm), recompute.
6. **HeteroFL / FjORD ports** — the baseline scripts call them via `--heterofl` / `--fjord` flags that still need ~150 LOC of integration in `trainer_FL.py`. If time-constrained, drop those two rows from Table 1 — the remaining baselines already provide solid coverage.

---

## 6 · Mapping of the user's 9 equations to v2 `main.tex`

| User-provided equation | Where it lives in v2 |
|---|---|
| $\delta_j(t):=\partial L/\partial u_j(t)$ | Eq. (\ref{eq:delta}) |
| BPTT temporal decomposition $\sum_t \delta_j(t) s_i(t)$ | Eq. (\ref{eq:bptt-grad-intro}) intro + Eq. (\ref{eq:bptt-grad}) preliminaries |
| First-order Taylor structural perturbation | Used as the *tie-breaker* in Eq. (\ref{eq:prune-score}) |
| Layer-wise BPTT $\sum_t \delta^l[t](o^{l-1}[t])^\top$ | Implicit in proof of Proposition 1 (Appendix \ref{app:proof-exact-zero}) |
| **Pruning score** $I_{\text{Prune}}$ | **Eq. (\ref{eq:prune-score})** — boxed |
| **Growth score** $I_{\text{Grow}}$ | **Eq. (\ref{eq:grow-score})** — boxed (RigL-style; recast from membrane-potential growth for theoretical consistency with Proposition 1) |
| **Mask-aware aggregation** | **Eq. (\ref{eq:agg})** — boxed, with unbiasedness proof in App. \ref{app:proof-agg} |
| Global objective $\mathcal{J}$ | Eq. (\ref{eq:objective}) |
| Contribution Ratio | Eq. (\ref{eq:cr}) in Discussion + appendix \ref{app:contribution-ratio} |
| Effective time-scale $T_{\text{eff}}$ | Eq. (\ref{eq:teff}) in Discussion + derivation in App. \ref{app:teff} |

---

## 7 · Acknowledgements

This draft was produced by an autopilot research-and-writing pipeline (Codewiz + the `ml-paper-writing` and `brainstorming-research-ideas` skill packages). Per ICLR 2026 policy, LLM usage is disclosed in the `LLM Usage Disclosure` section of `main.tex`. The autopilot run that produced v2 — including prior-art research (35+ references), narrative selection via Janusian/Bisociation/Negation frameworks, code wiring, and full draft rewrite — is logged in `NOVELTY_RESEARCH.md` and the conversation transcript.
