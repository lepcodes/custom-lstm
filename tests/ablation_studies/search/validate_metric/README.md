# validate_metric — the forget-signal "ruler"

Answers the metric-validation blocker (CLAUDE.md §"loss lever", handoff §4): **how do you
know a "forget" is warranted?** Synthetic series with labeled regimes provide ground truth,
so candidate forget signals can be *scored* instead of eyeballed. Results narrative:
`resources/metric_ruler/FINDINGS.md`.

**Ground-truth definition:** `forget_label = 1` when remembering the past would **not
improve the best possible guess of the next value** (conditional-mean relevance under the
thesis's one-step MSE task) — a strict refinement of "dependence is gone." The two coincide
for linear processes; they split exactly at the nonlinear corners (section 4 below).

| Script | Role |
|---|---|
| `generate_regimes.py` | 9 labeled scenarios in 4 sections (table below). Ground truth = per-step `forget_label` + `change_points` + `regime_id`. |
| `signals.py` | Candidate signals on a common `[0,1]` forget scale (1 = past irrelevant): `ewacf_mean`/`ewacf_max` (linear dependence, `EWACFEngine`, lags `[1,2,4,8]`, λ=0.9), `mi_knn` (nonlinear dependence, rolling kNN MI), `ks_shift` (distribution shift, rolling two-window KS). All causal. |
| `score_signals.py` | Scores every (scenario × signal × seed): ROC **AUC** + Spearman vs `forget_label` (sustained tracking), **detection lag** + **miss rate** at change points (responsiveness), **regime_gap** (signal separation between regimes; the key number where the label is single-class, e.g. `arch_switch`). Figures + CSVs → `resources/metric_ruler/`. |
| `plot_matrix.py` | Full **signal × scenario** comparison heatmaps (AUC + regime_gap) from `ruler_results.csv` — the anti-cherry-picking view: every signal's behavior on every scenario. |
| `plot_real_ewacf.py` | Director's ask (2026-07-08), signal half: EW-ACF dependence per real dataset → `resources/metric_ruler/real_datasets/`. The forget-gate half needs trained models (`GateTelemetry`) and is a separate step. |

```bash
python -m tests.ablation_studies.search.validate_metric.score_signals          # full (5 seeds, T=4000)
python -m tests.ablation_studies.search.validate_metric.score_signals --smoke  # quick check
python -m tests.ablation_studies.search.validate_metric.plot_matrix            # after score_signals
python -m tests.ablation_studies.search.validate_metric.plot_real_ewacf
```

## Scenario suite (4 sections)

Dependence-switch scenarios keep the marginal distribution (near-)identical across regimes,
so only the *temporal structure* changes; shift scenarios do the converse.

| Section — question it answers | Scenario | Structured regime ↔ noise | Label |
|---|---|---|---|
| **1. Linear switches** — does the signal track dependence at all? | `ar_noise_switch` | AR(1) φ=0.95 ↔ white noise | 0 ↔ 1 |
| | `phi_switch` | AR(1) φ=0.95 ↔ φ=0.3 | 0 ↔ 1 |
| | `periodic_noise_switch` | sine+noise ↔ white noise | 0 ↔ 1 |
| **2. Distribution shifts** — does it confuse regime change with irrelevance? | `mean_shift` | AR(1), level jumps | all 0 |
| | `var_shift` | AR(1), scale jumps | all 0 |
| **3. Multi-lag** — is the EW-ACF *configuration* (grid, aggregation) right? | `seasonal_ar_switch` | dependence **only at lag 8** (on-grid) ↔ noise | 0 ↔ 1 |
| | `off_grid_lag_switch` | dependence **only at lag 6** (off-grid) ↔ noise | 0 ↔ 1 |
| **4. Nonlinear corners** — does it track *forecast-relevance*, not dependence per se? | `nonlinear_map_switch` | logistic map (deterministic, ACF=0) ↔ i.i.d. same-marginal noise | 0 ↔ 1 |
| | `arch_switch` | ARCH(1) (dependence in scale only, conditional mean 0) ↔ noise | **all 1** |

## Pipeline & artifacts

`generate_regimes.py` and `signals.py` are **libraries** (no CLI, produce no files); the
runnables import them. One file dependency exists between runnables: `plot_matrix.py`
consumes `ruler_results.csv`, so it runs **after** `score_signals.py`. Everything else is
terminal: input for analysis, the meeting, and the thesis, not for further pipeline stages.

```
generate_regimes.py (lib)              signals.py (lib)
  9 scenarios, each with                 4 forget signals: series -> [0,1] per step
  forget_label + change_points           (1 = past irrelevant); uses EWACFEngine
  + regime_id                             from custom_lstm/utils.py
        │                                     │                │
        ▼                                     ▼                ▼
score_signals.py (runnable)                          plot_real_ewacf.py (runnable)
  scenario x signal x seed scoring                     reads data/preprocessed/<7 datasets>.csv
        │                                                       │
        ▼                                                       ▼
resources/metric_ruler/                              resources/metric_ruler/real_datasets/
  ruler_results.csv ──► plot_matrix.py (runnable) ──► ruler_matrix.png
```

| Artifact | Producer | What it is |
|---|---|---|
| `resources/metric_ruler/ruler_results.csv` | `score_signals.py` | Scores aggregated over seeds (AUC mean/std, Spearman, median detection lag, miss rate) — **the ranking table**. |
| `resources/metric_ruler/ruler_results_raw.csv` | `score_signals.py` | Same, per seed (for variance checks / re-aggregation). |
| `resources/metric_ruler/scenario_<name>.png` | `score_signals.py` | Seed-0 small multiples: series + each signal, ground truth shaded — the visual sanity check behind the numbers. |
| `resources/metric_ruler/ruler_matrix.png` | `plot_matrix.py` (from `ruler_results.csv`) | Signal × scenario heatmaps (AUC + regime_gap) — the full comparison, every signal on every scenario. |
| `resources/metric_ruler/real_datasets/ewacf_<dataset>.png` | `plot_real_ewacf.py` | Series + EW-ACF dependence + MI dependence vs its shuffled-series bias floor, per real dataset (director's ask + linearity check). |
| `resources/metric_ruler/real_datasets/ewacf_summary.csv` | `plot_real_ewacf.py` | Per-dataset dependence summary: EW-ACF median, MI median, MI bias floor, **MI excess** (structure above the floor). |
| `resources/metric_ruler/FINDINGS.md` | written by hand | Results narrative: what the numbers mean for the thesis. |

Run from the repo root with `conda activate lstm-env` (CPU; a full scoring run takes a few
minutes, dominated by the rolling-MI windows).

## Known limitations (read before trusting a ranking)

- `mi_knn` / `ks_shift` are window-based (200/150 steps) and inherit that lag; EW-ACF's
  effective memory at λ=0.9 is ~10 steps. Lag comparisons partly reflect window choices.
- The kNN MI estimator is **positively biased** on small windows (pure noise reads as
  nonzero MI), which can invert rankings; a shuffle-corrected (permutation-baseline) MI is
  the known fix and is not implemented.
- Detection uses an oracle-free but ad-hoc rule (>3 robust σ from the pre-change baseline,
  horizon 150); it ranks responsiveness, it is not a deployable detector.
- Not implemented (possible follow-ups): `ma_switch` (hard-cutoff dependence), an
  `ewacf_abs` signal (EW-ACF on |x|, which would fire on ARCH but *shouldn't* under the MSE
  task), nonlinear signals on the real datasets.
