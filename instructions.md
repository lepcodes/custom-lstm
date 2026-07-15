# Task: Extend the LSTM window-size sweep experiments (pre-advisor-meeting deliverables)

## Context (verify against the codebase before changing anything)
This repo runs one-step-ahead prediction experiments comparing two single-layer LSTM
training regimes on univariate time series:

- **Stateful**: hidden/cell states persist across the whole series. Input tensor is
  (batch=1, sequence = series_length − window_size, features = window_size).
  Backpropagation is truncated (TBPTT), currently ~200 steps.
- **Stateless**: the series is cut into independent windows; states reset to zero
  per window.

The existing sweep varies the sliding-window size (fed as input features) over
powers of two: 1, 2, 4, 8, 16, 32. Data split is 70/15/15 (train/val/test).
The current result plots show **best validation MSE** per window size, one curve
per regime (stateful = blue, stateless = red), across five datasets:
Acqua Alta, Ethernet traffic, TCP network traffic, sunspots, Mackey-Glass.

There is also a custom module under development: an MLP-based forget gate plus an
EW-ACF (exponentially-weighted autocorrelation) regularization penalty added to the
loss. Locate its implementation and its integration flag/config before Task 2.

If anything above doesn't match what you find in the code, stop and report the
discrepancy instead of guessing.

## Task 1 — Add training curves to the sweep plots (required)
Goal: test the hypothesis that the stateful regime *overfits* at large windows
(train error stays low while validation error climbs).

1. In the sweep, record the **training MSE** corresponding to the same checkpoint
   at which best validation MSE is achieved (not the final-epoch train MSE).
2. Produce **one figure per dataset** (independent files, not subplots):
   - x-axis: window size (log scale, base 2), values 1–32
   - one color per architecture: stateful vs stateless (keep the existing
     blue/red convention)
   - line style: **solid = train**, **dashed = validation**
   - y-axis: MSE; title = dataset name; include a legend
3. Save figures to a results/figures directory as individual image files
   (e.g. `sweep_<dataset>.png`), plus a CSV/table of the raw values
   (dataset, regime, window_size, train_mse, val_mse).
4. Reuse the existing sweep configuration exactly — same seeds, hyperparameters,
   splits, and epoch budget as the runs that produced the current validation-only
   plots. If cached results exist with train metrics already logged (e.g. MLflow),
   prefer re-plotting from them over re-running.

## Task 2 — Run the sweep with the custom module enabled (required)
1. Run the same window-size sweep (same datasets, windows, splits, seeds) using the
   **stateful LSTM with the custom module enabled** (MLP forget gate + EW-ACF
   penalty, current implementation as-is — do not tune or modify it).
2. Add it as a third curve to the Task 1 figures (new color, same solid/dashed
   train/val convention), or produce parallel figures if that gets too dense.
3. We expect *any* differentiated effect vs vanilla stateful — better or worse both
   matter. Report per dataset whether the module's stateful validation curve moves
   toward the stateless one at large windows.

## Task 3 — Multi-step prediction scaffold (optional, only if time permits)
Prepare (behind a config flag, default off) a multi-step output variant: the model
predicts a vector [x̂(t+1), …, x̂(t+H)] instead of a scalar. The vector output can
come from the dense layer after the LSTM. Do not run the full sweep with it yet —
just implement and smoke-test it.

## Constraints
- Do not change model hyperparameters, data preprocessing, or splits; comparability
  with the existing validation-only results is the whole point.
- Keep runs reproducible (fixed seeds, logged configs).
- Deliverables: the figures, the raw-values table, and a short markdown summary
  stating, per dataset, whether the train/val gap grows with window size for each
  regime, and what effect the module had.