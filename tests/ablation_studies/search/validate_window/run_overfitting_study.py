"""
Overfitting diagnostic — Stateful (A) vs Stateless/Diego (C), fixed budget
==========================================================================

Tests the hypothesis that the STATEFUL regime overfits at large windows: train
error stays low while validation error climbs.

Differences from the deployed best-val study (run_window_study / run_diego_baseline):
  - **No early stopping.** Every run trains a FIXED number of epochs, so the "last
    value" is comparable across windows and validation is allowed to climb (early
    stopping would halt at the val minimum and hide the very overfitting we test).
  - **Per-epoch train AND val MSE are logged** (the deployed study logged only the
    final best_val_loss for C). This lets plot_overfitting.py smooth the history
    (moving average) and take the last value — the co-director's middle ground.

Only two architectures (memoryless B is intentionally excluded):
  A = lstm_vanilla_windowed            (stateful, window as features)   -> blue
  C = lstm_vanilla_stateless_windowed  (Diego, window as time)          -> red

Metric definitions (identical for A and C, so they are comparable):
  train_mse = running training MSE averaged over the epoch's updates (standard
              training curve; a conservative, slightly-high estimate of train error
              -> if the val-train gap still widens, that is strong evidence).
  val_mse   = eval-mode MSE on the validation split (fixed end-of-epoch weights).

Logs to MLflow experiments `Overfit_<dataset>` (separate from the deployed study).

Usage:
    python -m tests.ablation_studies.search.validate_window.run_overfitting_study
    python -m tests.ablation_studies.search.validate_window.run_overfitting_study --smoke
    python -m tests.ablation_studies.search.validate_window.run_overfitting_study --only total_sunspots
"""

import argparse
import sys
import time
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

import mlflow
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from tests.ablation_studies.config import DataMode
from tests.ablation_studies.data_loader import load_data
from tests.ablation_studies.train import seed_everything
from custom_lstm.models.lstm_vanilla_stateful import LSTMVanillaStateful
from custom_lstm.training.tbptt import TBPTTTrainerStrategy

# Reuse the Diego (stateless) data + model from the deployed-study runner.
from tests.ablation_studies.search.validate_window.run_diego_baseline import (
    StatelessWindowedLSTM,
    build_diego_dataset,
)

DB_PATH = "C:/Users/Luis/Documents/ML-AI-Projects/custom-lstm/tests/ablation_studies/mlflow.db"
SCRATCH_DB = "C:/Users/Luis/AppData/Local/Temp/claude/C--Users-Luis-Documents-ML-AI-Projects-custom-lstm/82cd04c4-a082-406c-b45a-87288ba4a565/scratchpad/overfit_smoke.db"
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
BPTT_STEPS = 50
LR = 0.001
EPOCHS = 200          # FIXED budget, no early stopping
BATCH_SIZE_C = 512
SEED = 42
DEVICE = torch.device("cpu")

ARCH_A = "lstm_vanilla_windowed"            # stateful
ARCH_C = "lstm_vanilla_stateless_windowed"  # stateless / Diego
EXPERIMENT_PREFIX = "Overfit"


def run_stateful(csv_path: str, window: int, epochs: int):
    """Yield (epoch, train_mse, val_mse) for the stateful windowed LSTM (TBPTT)."""
    seed_everything(SEED)
    splits, _ = load_data(csv_path, window_size=window)
    tr = splits.train.get_by_mode(DataMode.WINDOWED)
    vl = splits.val.get_by_mode(DataMode.WINDOWED)
    Xtr, Ytr = tr.X.to(DEVICE), tr.Y.to(DEVICE)
    Xval, Yval = vl.X.to(DEVICE), vl.Y.to(DEVICE)

    model = LSTMVanillaStateful(input_size=window, hidden_size=HIDDEN_SIZE, output_size=OUTPUT_SIZE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    trainer = TBPTTTrainerStrategy(model, optimizer, nn.MSELoss(), DEVICE, bptt_steps=BPTT_STEPS)

    for epoch in range(1, epochs + 1):
        train_mse = trainer.train_epoch(Xtr, Ytr)["train_loss"]      # running train MSE
        val_mse = trainer.validate_epoch(Xval, Yval)["val_loss"]     # eval-mode val MSE
        yield epoch, train_mse, val_mse


def run_stateless(csv_path: str, window: int, epochs: int):
    """Yield (epoch, train_mse, val_mse) for the stateless windowed LSTM (Diego)."""
    seed_everything(SEED)
    data = build_diego_dataset(csv_path, window)
    if data is None:
        return
    Xtr, Ytr, Xval_cpu, Yval_cpu = data
    Xval, Yval = Xval_cpu.to(DEVICE), Yval_cpu.to(DEVICE)

    model = StatelessWindowedLSTM(input_size=1, hidden_size=HIDDEN_SIZE, output_size=OUTPUT_SIZE).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()
    gen = torch.Generator().manual_seed(SEED)
    loader = DataLoader(TensorDataset(Xtr, Ytr), batch_size=BATCH_SIZE_C, shuffle=True, generator=gen)

    for epoch in range(1, epochs + 1):
        model.train()
        running, n = 0.0, 0
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            pred, _ = model(xb)
            loss = criterion(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running += loss.item() * xb.size(0)
            n += xb.size(0)
        train_mse = running / n                                      # running train MSE
        model.eval()
        with torch.no_grad():
            val_mse = criterion(model(Xval)[0], Yval).item()         # eval-mode val MSE
        yield epoch, train_mse, val_mse


ARCHS = [(ARCH_A, run_stateful), (ARCH_C, run_stateless)]


def finished_cells(datasets) -> set:
    """(dataset, arch, window) triples that already have a FINISHED run — for --resume."""
    c = mlflow.MlflowClient()
    done = set()
    for label, _ in datasets:
        exp = c.get_experiment_by_name(f"{EXPERIMENT_PREFIX}_{label}")
        if exp is None:
            continue
        for r in c.search_runs([exp.experiment_id], max_results=5000):
            if r.info.status == "FINISHED":
                arch = r.data.params.get("architecture", "").split(".")[-1].lower()
                ws = r.data.params.get("window_size")
                if arch and ws is not None:
                    done.add((label, arch, int(ws)))
    return done


def main():
    parser = argparse.ArgumentParser(description="Overfitting diagnostic: stateful vs stateless, fixed budget")
    parser.add_argument("--smoke", action="store_true", help="Scratch DB, 1 dataset, 2 windows, few epochs.")
    parser.add_argument("--only", type=str, default=None, help="Comma-separated dataset labels to run.")
    parser.add_argument("--resume", action="store_true", help="Skip (dataset, arch, window) cells that already finished.")
    args = parser.parse_args()

    if args.smoke:
        db, datasets, grid, epochs = SCRATCH_DB, DATASETS[:1], [1, 128], 6
        Path(SCRATCH_DB).parent.mkdir(parents=True, exist_ok=True)
        print(">>> SMOKE MODE (scratch DB)")
    else:
        db, datasets, grid, epochs = DB_PATH, DATASETS, WINDOW_GRID, EPOCHS

    if args.only:
        wanted = {s.strip() for s in args.only.split(",")}
        datasets = [d for d in datasets if d[0] in wanted]
        print(f">>> --only: {[d[0] for d in datasets]}")

    mlflow.set_tracking_uri(f"sqlite:///{db}")
    done = finished_cells(datasets) if args.resume else set()
    if done:
        print(f">>> --resume: skipping {len(done)} already-finished cells")
    total = len(datasets) * len(ARCHS) * len(grid)
    print(f"{'=' * 72}\n  OVERFITTING STUDY — {total} runs  "
          f"(fixed {epochs} epochs, NO early stopping, device={DEVICE})\n{'=' * 72}")

    idx = 0
    t0 = time.time()
    for label, csv_filename in datasets:
        mlflow.set_experiment(f"{EXPERIMENT_PREFIX}_{label}")
        csv_path = str(Path(DATA_DIR) / csv_filename)
        for arch, runner in ARCHS:
            for window in grid:
                idx += 1
                if (label, arch, window) in done:
                    print(f"  Run {idx}/{total}: {label} | {arch} | ws={window}  [skip: already finished]")
                    continue
                print(f"\n  Run {idx}/{total}: {label} | {arch} | ws={window}")
                seed_everything(SEED)
                with mlflow.start_run(run_name=f"{arch}_ws{window}"):
                    mlflow.log_params({
                        "architecture": arch,
                        "window_size": window,
                        "epochs": epochs,
                        "hidden_size": HIDDEN_SIZE,
                        "lr": LR,
                        "bptt_steps": BPTT_STEPS,
                        "batch_size": BATCH_SIZE_C if arch == ARCH_C else "n/a (stateful)",
                        "early_stopping": False,
                        "dataset_name": label,
                    })
                    last = None
                    for epoch, tr, va in runner(csv_path, window, epochs):
                        mlflow.log_metrics({"train_mse": tr, "val_mse": va}, step=epoch)
                        last = (tr, va)
                        gap = va - tr
                        print(f"    epoch {epoch:>3}/{epochs}  train {tr:.5f}  val {va:.5f}  gap {gap:+.5f}",
                              flush=True)
                    if last:
                        print(f"    -> final train {last[0]:.5f}  val {last[1]:.5f}", flush=True)

    print(f"\n{'=' * 72}\n  DONE — {idx} runs in {(time.time() - t0) / 60:.1f} min\n{'=' * 72}")


if __name__ == "__main__":
    main()
