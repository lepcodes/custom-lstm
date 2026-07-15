# validate_window — window-size studies

Two related studies on how window size affects a **stateful** LSTM vs a **stateless**
(Diego) LSTM. Both log to `tests/ablation_studies/mlflow.db`; figures/report land in
`resources/window_study/`.

## 1. Deployed study (best validation MSE)

Reports the performance of the *deployed* model (early stopping + best-weight restore).

| Script | Role |
|---|---|
| `run_window_study.py` | Stateful **A** + memoryless **B**, window sweep 1–128, early-stopped. `--only`, `--smoke`. |
| `run_diego_baseline.py` | Stateless **C** (Diego): window-as-time, shuffled i.i.d. mini-batches. `--only`, `--smoke`. |
| `plot_window_study.py` | Overlay curves → `resources/window_study/window_study*.png` + `window_study_results.csv`. `--drop`, `--out`. |

Outputs + written analysis: `resources/window_study/window_study.png`,
`window_study_stateful_vs_stateless.png`, `REPORT.md`.

```bash
python -m tests.ablation_studies.search.validate_window.run_window_study        # A + B
python -m tests.ablation_studies.search.validate_window.run_diego_baseline      # C
python -m tests.ablation_studies.search.validate_window.plot_window_study
```

## 2. Overfitting diagnostic (train vs val, fixed budget)

Tests whether the **stateful** regime overfits at large windows (train low, val climbs).
Fixed epoch budget (**no early stopping**) so the val climb is visible; logs **per-epoch
train + val MSE**. Only stateful **A** vs stateless **C** (memoryless B excluded).

| Script | Role |
|---|---|
| `run_overfitting_study.py` | A + C, window 1–128, fixed 200 epochs, per-epoch `train_mse`/`val_mse` → `Overfit_<dataset>` experiments. `--only`, `--smoke`, `--resume`. |
| `plot_overfitting.py` | Per-dataset summary + epoch diagnostics, for **both** smoothing filters. Summary point = filter's "last value", band = ±1 std. Flags: `--k` (SMA window), `--ewma` (EWMA span), `--filters sma,ewma`. |

Outputs, split by plot type and filter:

```
resources/window_study/overfitting/
  summary/{sma,ewma}/overfit_<dataset>.png   (+ overfit_results.csv)
  epochs/{sma,ewma}/overfit_epochs_<dataset>.png
```

- **SMA** = simple moving average, mean of last `--k` epochs (default 20).
- **EWMA** = exponentially weighted, `--ewma` span (default 20); `adjust=True` so there is
  no head-start bias from the first (untrained) epoch.

```bash
python -m tests.ablation_studies.search.validate_window.run_overfitting_study --resume
python -m tests.ablation_studies.search.validate_window.plot_overfitting --k 20 --ewma 20
```

Run all from the repo root with `conda activate lstm-env` (device is pinned to CPU —
batch=1 recurrent stepping is faster on CPU than GPU here).
