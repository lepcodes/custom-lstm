# Metric ruler — findings (updated 2026-07-12, full-matrix analysis)

Harness: `tests/ablation_studies/search/validate_metric/` (see its README for the scenario
sections, pipeline, and limitations). 9 scenarios × 4 signals × 5 seeds, T=4000. Raw
numbers: `ruler_results.csv` / `ruler_results_raw.csv`; per-scenario figures
`scenario_*.png`; **the full signal × scenario comparison is `ruler_matrix.png`** — every
signal is evaluated on every scenario, not only on the scenario it was designed for.

Ground-truth label: **forget = 1 when remembering the past would not improve the best
possible guess of the next value** (conditional-mean relevance under the thesis's one-step
MSE task). Metrics: AUC/Spearman = sustained tracking of that label; det_lag/miss =
responsiveness at change points; **regime_gap** = signal separation between regimes
(the decisive number where AUC is undefined, and for threshold-based losses).

## The full matrix (mean over 5 seeds)

**AUC** (dependence scenarios; 0.5 = chance, below = anti-correlated):

| scenario | ewacf_max | ewacf_mean | mi_knn | ks_shift |
|---|---|---|---|---|
| ar_noise_switch | **0.973** | 0.965 | 0.865 | 0.327 |
| phi_switch | **0.969** | 0.957 | 0.865 | 0.342 |
| periodic_noise_switch | 0.988 | **0.989** | 0.871 | 0.571 |
| seasonal_ar_switch | **0.977** | 0.966 | 0.844 | 0.368 |
| off_grid_lag_switch | 0.725 | 0.764 | **0.787** | 0.356 |
| nonlinear_map_switch | 0.488 | 0.493 | **0.873** | 0.543 |

**Change-point detection** (single-class scenarios; median lag · miss rate):

| scenario | ewacf_max | ewacf_mean | mi_knn | ks_shift |
|---|---|---|---|---|
| mean_shift | 0 · 43% | 4.5 · 66% | **20.5 · 2.9%** | 62.5 · 5.7% |
| var_shift | 8.5 · 46% | 34 · 74% | **27.5 · 37%** | 50 · 51% |

**regime_gap** on arch_switch (≈ 0 is *correct* — the past never helps an MSE forecast):
ewacf_max +0.010 · ewacf_mean +0.005 · mi_knn +0.028 · ks_shift −0.002. All effectively flat.

## Per-section reading (all four signals each time)

**1. Linear switches.** `ewacf_max` best everywhere (0.97–0.99, onset lag 3–9, zero
misses); `ewacf_mean` matches on AUC but is 3–8× slower at onset with ~25% smaller
regime_gap; `mi_knn` a consistent step behind (0.87) with 30–40-step window lag; `ks_shift`
*below chance* on the AR switches (0.33–0.34) and merely useless on the periodic one (0.57)
— see the KS mechanism below.

**2. Distribution shifts.** The buried headline: **`mi_knn`, not KS, is the best
change-point detector on `mean_shift`** (miss 2.9% vs KS's 5.7%, and 3× faster). Mechanism:
an MI window straddling a level jump sees a two-cluster joint distribution, which registers
as strong "dependence" — MI reacts to shifts through within-window mixture structure, not
temporal dependence per se. `ewacf_max` blips instantly (lag 0) but misses 43%; `ewacf_mean`
is worse (66%). **`var_shift` defeats every signal** — best miss rate is MI's 37%, and KS
misses half *on its home turf*; none of the tested signals reliably detects a pure
volatility regime change. Relevant caveat for real data, where volatility shifts are the
most plausible drift type (exchange_rate, traffic).

**3. Multi-lag.** `seasonal_ar_switch` (dependence only at lag 8, on-grid): prediction
*confirmed with nuance* — `ewacf_max` vs `ewacf_mean` AUC barely separates (0.977/0.966)
because AUC is threshold-free, but onset lag (3 vs 19.5) and regime_gap (0.53 vs 0.24)
split hard; with three dead lags each contributing "forget ≈ 1", the mean signal can't drop
much below ~0.75 during the structured regime. `mi_knn` holds 0.844; `ks_shift` is again
anti-correlated (0.368). `off_grid_lag_switch` (lag 6, invisible to the grid): prediction
*directionally confirmed, magnitude refuted* — EW-ACF drops to 0.73/0.76, a huge cost but
not blindness (the online estimator's recency-weighted mean/variance leaks envelope
information; the sample ACF at monitored lags is ≈ 0, verified with `std_acf`). And the
second buried finding: **`mi_knn` (0.787) beats both EW-ACF variants off-grid** — its
window-mixture sensitivity partially compensates for the missing lag. **The lag grid is a
load-bearing hyperparameter.**

**4. Nonlinear corners.** `nonlinear_map_switch`: EW-ACF *fully blind* (0.488/0.493 —
flat panels in the figure) on a deterministic, perfectly predictable past; `mi_knn` 0.873
with zero misses — the "what dependency" axis has measured stakes. `ks_shift` ≈ chance
(0.543) is itself a designed control that passed: the two regimes share an identical
marginal, so a distribution test *should* see nothing. `arch_switch`: all four signals
correctly flat (gaps ≤ 0.03). The feared MI false alarm did not materialize — kNN MI on
200-sample windows barely resolves ARCH(α=0.5) scale-dependence — right behavior, partly
for weak-estimator reasons (α→0.9 or larger windows could revive the risk).

## Cross-cutting observations (only visible in the full matrix)

- **Peak vs floor.** `ewacf_max` has the best peaks (0.97–0.99) but the worst floor among
  dependence signals (0.488 on the nonlinear map). `mi_knn` never drops below **0.787** on
  any dependence scenario — a robust generalist that wins nothing linear but fails nothing.
  If real data may hide nonlinear or off-grid structure, that floor is worth respecting.
- **MI is secretly a hybrid.** Its window-mixture sensitivity makes it react to level
  shifts and off-grid structure as well as dependence — best-in-class on `mean_shift` and
  `off_grid_lag_switch`, the two scenarios nobody designed it for. The same trait is a
  liability under the MSE task (a mean shift shouldn't trigger "retain").
- **KS has one consistent mechanism, two consequences.** It fires on wandering local level.
  On persistent-AR regimes that makes it *anti-correlated* with the correct forget signal
  (0.33–0.37 on all four wandering switches); on stable-marginal scenarios (periodic,
  nonlinear map) it's merely at chance. Its only earned win, `mean_shift`, is matched and
  beaten by MI. As a *forget* signal KS is disqualified — its remaining value is as a
  separate drift channel, and even there `var_shift` shows it can miss.
- **No tested signal handles variance shifts.** If "regime change" in the thesis datasets
  is predominantly a volatility change, none of the current candidates detects it reliably.

## Overall ranking (sharpened by the matrix)

**`ewacf_max`** remains the recommended forget signal for the thesis's current framing:
best-in-class on all linear and seasonal dependence, correctly silent on shifts and ARCH.
Its two measured failure modes — off-grid lags (0.98 → 0.73) and nonlinear-mean dependence
(blind, 0.49) — are exactly where `mi_knn` (floor 0.787) covers, suggesting a possible
two-signal formulation later. `ks_shift` is disqualified as a forget signal.

## Dependence on the 7 real datasets: EW-ACF (director's ask) + MI linearity check

`real_datasets/ewacf_summary.csv`, figures `real_datasets/ewacf_*.png` (three panels:
series, EW-ACF, MI vs its shuffled-series bias floor). Median dependence
(1 = past informative), lags [1,2,4,8], λ=0.9. **mi_excess** = median MI dependence minus
the same estimator's median on a shuffled copy (same marginal, temporal structure
destroyed) — only that excess counts as real structure, because kNN MI reads ~0.10–0.13
of "dependence" on pure noise:

| dataset | EW-ACF | EW-ACF < 0.3 share | MI | MI bias floor | **MI excess** |
|---|---|---|---|---|---|
| total_sunspots | 0.951 | 0.000 | 0.983 | 0.133 | 0.851 |
| aqua_alta | 0.942 | 0.000 | 0.976 | 0.115 | 0.860 |
| mackey_glass | 0.783 | 0.000 | 0.900 | 0.098 | 0.802 |
| etth1 | 0.558 | 0.051 | 0.844 | 0.119 | 0.726 |
| exchange_rate | 0.540 | 0.168 | 0.925 | 0.115 | **0.810** |
| lbl_tcp_3 | 0.242 | 0.676 | 0.370 | 0.122 | 0.248 |
| ethernet_traffic | 0.163 | 0.926 | 0.565 | 0.129 | **0.436** |

**Vs. Dra. Arana's predictions (2026-07-08):**

- `mackey_glass` → "EW-ACF high": **confirmed** (0.78; and MI reads 0.90 — correlation
  slightly *understates* chaotic predictability, exactly as `nonlinear_map_switch` warned).
- `aqua_alta` → "EW-ACF ≈ 0 (short memory)": **refuted** — second highest (0.94, MI
  agrees at 0.98); tidal periodicity pins short-lag autocorrelation near 1.

**The MI check (2026-07-12) — does EW-ACF miss real structure? Split verdict:**

- `lbl_tcp_3`: **EW-ACF verdict survives.** MI excess (0.25) ≈ EW-ACF reading (0.24) —
  both metrics agree the past is genuinely close to irrelevant most of the time.
- `ethernet_traffic`: **EW-ACF verdict is partly a linearity artifact.** EW-ACF says 0.16
  ("past irrelevant 93% of the time") but MI excess is 0.44 — bursty traffic carries
  substantial structure that correlation cannot see. The meeting story needs the honest
  version: *"low **linear** dependence ↔ stateful overfitting"* still holds as an observed
  association, but "there is nothing worth remembering" does not — a nonlinear reader finds
  plenty. (Arguably this sharpens the overfitting story: the stateful LSTM, a nonlinear
  model, fails to exploit structure that demonstrably exists and memorizes instead.)
- `exchange_rate`: **the largest divergence of all** (EW-ACF 0.54 vs MI 0.93, excess 0.81).
  A near-random-walk in levels is in fact *maximally* dependent (the best guess of x_t is
  x_{t−1}); MI sees that plainly, while the λ=0.9 EW-ACF understates it — the
  recency-weighted variance normalization deflates correlation on slowly wandering series.
  This is an **estimator artifact of EW-ACF on integrated/long-memory series**, distinct
  from the nonlinearity blind spot, and worth knowing before interpreting EW-ACF levels on
  any near-random-walk dataset. (`etth1`, excess 0.73 vs EW-ACF 0.56, mixes both effects:
  smooth wandering + off-grid daily seasonality.)

## What this unblocks / next steps

1. Relate `ewacf_max` to actual forget-gate activations (`GateTelemetry`) on
   `ethernet_traffic` vs `aqua_alta` — the other half of the director's ask.
2. `EWACFLoss` (`custom_lstm/losses/acf_losses.py`) already implements both strategies but
   **defaults to `aggregation_strategy="average"`** — the ruler says `max_pooling` (retain
   if *any* lag is alive) is the better default: same AUC but ~6× faster onset and twice
   the regime separation, which matters directly for the loss's absolute `threshold` clamp.
   Switch the sweeps to `max_pooling` (new registered variant, keep `average` runs
   comparable per CLAUDE.md).
3. Lag-grid sensitivity: the off-grid result makes the grid a first-class hyperparameter.
4. The real-data MI check elevates the metric-design question from hypothetical to
   demonstrated: EW-ACF materially understates dependence on 3 of 7 datasets
   (`ethernet_traffic` via nonlinearity; `exchange_rate`, `etth1` via the λ-weighted
   estimator on wandering series + off-grid seasonality). Candidate cheap fixes to run
   through the ruler: EW rank (Spearman-style) correlation, EW-ACF on returns/increments
   for integrated series, a denser lag grid. MI itself stays a diagnostic (windowed kNN is
   neither online nor O(1)), now with the shuffle-floor correction built in.
