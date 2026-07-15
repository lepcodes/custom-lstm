"""
Paradigm C — faithful STATELESS windowed LSTM (Diego's paradigm)
================================================================

This is the honest representation of Diego's design and of the classic
"windowed LSTM" the director invokes. It is deliberately DIFFERENT from the
stateful A/B study in two ways that matter:

  1. Window is consumed as TIME, not as features.
     Data layout is [num_windows, window, 1] — each window is a short sequence
     the LSTM UNROLLS over (recurrence within the window), with state reset
     between windows. So window length == recurrence depth. This is what makes
     "bigger window -> more temporal context -> lower MSE" a real mechanism here.
     (Contrast: the stateful A/B study stacks the window as FEATURES,
      [1, N-w, w], and carries state across the whole series.)

  2. Training SHUFFLES the (window, target) pairs into i.i.d. mini-batches.
     A stateless model treats each window as an independent sample, so the
     correct regime is DataLoader(shuffle=True) — exactly what Diego's Keras
     model.fit(...) does by default. This is the opposite of the stateful
     pipeline in tests/ablation_studies/data_loader.py, which forbids shuffling
     ("No random shuffling allowed for stateful RNNs", line 96) because state
     must carry in order. Each paradigm is trained the way it is meant to be.

Because of (1) and (2) this cannot be expressed as a sweep YAML — the harness
data_loader/TBPTT path produces neither this layout nor this training regime.
Hence this standalone runner.

CAVEAT (stated in the report): the stateful (A/B) and stateless (C) models use
different, paradigm-appropriate training regimes, so absolute MSE levels are not
perfectly comparable across paradigms. The scientific object is the SHAPE of
each curve vs window size (flat vs decreasing), not cross-paradigm height.

Model:   custom_lstm.models.lstm_vanilla.LSTMVanilla (stateless; resets state
         every forward; returns the FINAL-step output -> one prediction/window).

Logs to the SAME MLflow experiments as the A/B study (Window_Study_<dataset>)
with architecture tag ARCH_TAG, so plot_window_study.py renders it as a 3rd curve.

Usage:
    python -m tests.ablation_studies.search.validate_window.run_diego_baseline           # full
    python -m tests.ablation_studies.search.validate_window.run_diego_baseline --smoke   # scratch DB
"""

import argparse
import copy
import sys
import time
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

import mlflow
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from custom_lstm.models.lstm_vanilla import LSTMVanilla
from tests.ablation_studies.train import seed_everything


class StatelessWindowedLSTM(LSTMVanilla):
    """The repo's stateless LSTMVanilla, made instantiable.

    LSTMVanilla inherits AblationModel's abstract reset_state() but never
    implements it (it was never used in the harness). A stateless model has no
    persistent state to reset, so this is a no-op — identical in spirit to the
    no-op reset_state() on the stateless lstm_custom.py.
    """

    def reset_state(self):
        pass

# ── Fixed configuration (mirrors run_window_study.py where it should match) ────
LIVE_DB = "C:/Users/Luis/Documents/ML-AI-Projects/custom-lstm/tests/ablation_studies/mlflow.db"
SCRATCH_DB = "C:/Users/Luis/AppData/Local/Temp/claude/C--Users-Luis-Documents-ML-AI-Projects-custom-lstm/82cd04c4-a082-406c-b45a-87288ba4a565/scratchpad/diego_smoke.db"
DATA_DIR = "C:/Users/Luis/Documents/ML-AI-Projects/custom-lstm/data/preprocessed"

DATASETS = [
    ("aqua_alta", "aqua_alta.csv"),
    ("lbl_tcp_3", "lbl_tcp_3.csv"),
    ("mackey_glass", "mackey_glass.csv"),
    ("ethernet_traffic", "ethernet_traffic.csv"),
    ("total_sunspots", "total_sunspots.csv"),
    ("etth1", "etth1.csv"),
    ("exchange_rate", "exchange_rate.csv"),
]
WINDOW_GRID = [1, 2, 4, 8, 16, 32, 64, 128]

HIDDEN_SIZE = 64
OUTPUT_SIZE = 1
LR = 0.001
EPOCHS = 200
PATIENCE = 20
BATCH_SIZE = 512  # large batch: stateless windows re-unroll per-window, so few big batches keep it fast
SEED = 42
DEVICE = "cpu"

# Distinct tag so the plot separates this curve from A (stateful) and B (memoryless).
ARCH_TAG = "lstm_vanilla_stateless_windowed"
EXPERIMENT_PREFIX = "Window_Study"

# Same chronological split as tests/ablation_studies/data_loader.py.
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15


def make_windows_as_time(series_scaled: np.ndarray, window: int):
    """
    Diego layout: X = [num, window, 1] (window unrolled as TIME, 1 feature),
    Y = [num, 1] (the value immediately AFTER each window).
    Returns None if the split is too short for this window.
    """
    if len(series_scaled) <= window + 1:
        return None
    windows = np.lib.stride_tricks.sliding_window_view(series_scaled, window)  # [num, window]
    X = windows[:-1]                       # drop last window (no future target)
    Y = series_scaled[window:]             # next-step target for each window
    X = X[..., None].astype(np.float32)    # [num, window, 1]
    Y = Y.reshape(-1, 1).astype(np.float32)  # [num, 1]
    return torch.from_numpy(X), torch.from_numpy(Y)


def build_diego_dataset(csv_path: str, window: int):
    """Chronological split + train-only scaling + per-split windowing (no leakage)."""
    df = pd.read_csv(csv_path).dropna().reset_index(drop=True)
    raw = df.iloc[:, 0].values

    n = len(raw)
    train_end = int(n * TRAIN_RATIO)
    val_end = train_end + int(n * VAL_RATIO)
    train_raw = raw[:train_end]
    val_raw = raw[train_end:val_end]

    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_raw.reshape(-1, 1)).flatten()
    val_scaled = scaler.transform(val_raw.reshape(-1, 1)).flatten()

    train = make_windows_as_time(train_scaled, window)
    val = make_windows_as_time(val_scaled, window)
    if train is None or val is None:
        return None
    return train[0], train[1], val[0], val[1]


def train_one(csv_path: str, window: int, epochs: int, patience: int) -> float:
    """Train a stateless windowed LSTM with SHUFFLED mini-batches; return best val MSE."""
    data = build_diego_dataset(csv_path, window)
    if data is None:
        print(f"  [skip] window={window} too large for split")
        return float("nan")
    X_tr, Y_tr, X_val, Y_val = data
    X_val, Y_val = X_val.to(DEVICE), Y_val.to(DEVICE)

    # Stateless i.i.d. training: shuffle (window, target) PAIRS together every epoch.
    gen = torch.Generator().manual_seed(SEED)
    loader = DataLoader(TensorDataset(X_tr, Y_tr), batch_size=BATCH_SIZE, shuffle=True, generator=gen)

    model = StatelessWindowedLSTM(input_size=1, hidden_size=HIDDEN_SIZE, output_size=OUTPUT_SIZE).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()

    best_val = float("inf")
    best_state = None
    no_improve = 0

    for epoch in range(1, epochs + 1):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            pred, _ = model(xb)              # [B, 1] — final-step output per window
            loss = criterion(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_pred, _ = model(X_val)
            val_loss = criterion(val_pred, Y_val).item()

        if val_loss < best_val:
            best_val = val_loss
            best_state = copy.deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1

        if epoch == 1 or epoch % 20 == 0:
            print(f"    epoch {epoch:>3}/{epochs}  val MSE: {val_loss:.5f}")

        if no_improve >= patience:
            print(f"    early stop @ epoch {epoch} (best {best_val:.5f})")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return best_val


def main():
    parser = argparse.ArgumentParser(description="Stateless windowed LSTM (Diego paradigm)")
    parser.add_argument("--smoke", action="store_true", help="Scratch DB, 1 dataset, 2 windows, few epochs.")
    parser.add_argument("--only", type=str, default=None,
                        help="Comma-separated dataset labels to run (default: all). "
                             "Use this to add new datasets without re-running finished ones.")
    args = parser.parse_args()

    if args.smoke:
        db, datasets, grid = SCRATCH_DB, DATASETS[:1], [1, 8]
        epochs, patience = 5, 5
        Path(SCRATCH_DB).parent.mkdir(parents=True, exist_ok=True)
        print(">>> SMOKE MODE (scratch DB — live study untouched)")
    else:
        db, datasets, grid = LIVE_DB, DATASETS, WINDOW_GRID
        epochs, patience = EPOCHS, PATIENCE

    if args.only:
        wanted = {s.strip() for s in args.only.split(",")}
        datasets = [d for d in datasets if d[0] in wanted]
        missing = wanted - {d[0] for d in datasets}
        if missing:
            raise SystemExit(f"--only names unknown datasets: {sorted(missing)}")
        print(f">>> --only: running {[d[0] for d in datasets]}")

    mlflow.set_tracking_uri(f"sqlite:///{db}")

    total = len(datasets) * len(grid)
    print(f"{'=' * 72}\n  DIEGO BASELINE (stateless windowed) — {total} runs  (device={DEVICE})\n{'=' * 72}")

    idx = 0
    t0 = time.time()
    for label, csv_filename in datasets:
        mlflow.set_experiment(f"{EXPERIMENT_PREFIX}_{label}")
        for window in grid:
            idx += 1
            print(f"\n  Run {idx}/{total}: {label} | {ARCH_TAG} | ws={window}")
            seed_everything(SEED)
            with mlflow.start_run(run_name=f"{ARCH_TAG}_ws{window}"):
                mlflow.log_params({
                    "architecture": ARCH_TAG,
                    "window_size": window,
                    "hidden_size": HIDDEN_SIZE,
                    "lr": LR,
                    "epochs": epochs,
                    "batch_size": BATCH_SIZE,
                    "dataset_name": label,
                    "data_layout": "diego_window_as_time",
                    "train_shuffle": True,
                    "stateful": False,
                })
                best_val = train_one(str(Path(DATA_DIR) / csv_filename), window, epochs, patience)
                mlflow.log_metric("best_val_loss", best_val)
                print(f"  -> best_val_loss = {best_val:.6f}")

    print(f"\n{'=' * 72}\n  DONE — {idx} runs in {(time.time() - t0) / 60:.1f} min\n{'=' * 72}")


if __name__ == "__main__":
    main()
