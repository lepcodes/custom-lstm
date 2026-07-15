"""
Window-Size Study: Stateful vs Stateless LSTM
=============================================

Runs the SAME window-size sweep on two architectures and logs every run to MLflow,
so the resulting curves can be overlaid (see plot_window_study.py):

  A — Stateful (the thesis design):
      architecture = lstm_vanilla_windowed
      Hidden/cell state is carried across TBPTT chunks. Temporal memory lives in the
      recurrence, NOT in the window. Expectation: roughly flat / non-monotonic vs window.

  B — Stateless control (the "windowed predictor" / Diego paradigm):
      architecture = lstm_vanilla_windowed_no_recurrence  (sever_recurrence=True)
      State is re-zeroed every timestep, so the window IS the entire memory.
      Expectation: monotonic decrease vs window (the director's predicted curve).

Both use a plain vanilla LSTM cell with MSE loss — none of the thesis innovations
(MLP forget gate, EW-ACF penalty) are involved, so this isolates the basic
stateful-vs-stateless question.

Device: forced to CPU. For batch=1 single-sequence TBPTT the cell is stepped one
timestep at a time in a Python loop, so per-step kernel-launch overhead makes the GPU
*slower* than the CPU for this recurrent workload — a deliberate choice, not a fallback.

Usage:
    python -m tests.ablation_studies.search.validate_window.run_window_study           # full study
    python -m tests.ablation_studies.search.validate_window.run_window_study --smoke   # quick check
"""

import argparse
import sys
import time
from pathlib import Path

# The shared harness prints box-drawing characters (config.describe_pipeline()).
# Force UTF-8 stdout so it doesn't crash on a cp1252 Windows console.
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

import mlflow

from tests.ablation_studies.config import ExperimentConfig
from tests.ablation_studies.train import run_experiment, seed_everything

# ── Fixed configuration ───────────────────────────────────────────────────────
DB_PATH = "C:/Users/Luis/Documents/ML-AI-Projects/custom-lstm/tests/ablation_studies/mlflow.db"
DATA_DIR = "C:/Users/Luis/Documents/ML-AI-Projects/custom-lstm/data/preprocessed"

# (label, csv filename). mackey_glass is included as a *fair test*: a strongly
# autocorrelated, deterministic series where the stateless control may genuinely
# benefit from a larger window — making the comparison credible, not cherry-picked.
DATASETS = [
    ("aqua_alta", "aqua_alta.csv"),
    ("lbl_tcp_3", "lbl_tcp_3.csv"),
    ("mackey_glass", "mackey_glass.csv"),
    ("ethernet_traffic", "ethernet_traffic.csv"),
    ("total_sunspots", "total_sunspots.csv"),
    ("etth1", "etth1.csv"),
    ("exchange_rate", "exchange_rate.csv"),
]

# (architecture, sever_recurrence). The "_no_recurrence" arch maps to the same
# stateful class but with sever_recurrence=True -> a memoryless windowed predictor.
ARCHITECTURES = [
    ("lstm_vanilla_windowed", False),            # A: stateful (thesis design)
    ("lstm_vanilla_windowed_no_recurrence", True),  # B: stateless control (Diego paradigm)
]

WINDOW_GRID = [1, 2, 4, 8, 16, 32, 64, 128]

# Fixed hyperparameters (moderate-to-rigorous, patience=20 per the agreed design).
HIDDEN_SIZE = 64
OUTPUT_SIZE = 1
BPTT_STEPS = 50
LR = 0.001
EPOCHS = 200
PATIENCE = 20
SEED = 42
DEVICE = "cpu"

EXPERIMENT_PREFIX = "Window_Study"


def build_config(csv_filename: str, architecture: str, sever_recurrence: bool, window_size: int,
                 epochs: int, bptt_steps: int) -> ExperimentConfig:
    """Assemble one ExperimentConfig for a (dataset, architecture, window_size) cell."""
    model_kwargs = {
        "input_size": window_size,   # windowed: feature dim == window_size
        "hidden_size": HIDDEN_SIZE,
        "output_size": OUTPUT_SIZE,
    }
    if sever_recurrence:
        model_kwargs["sever_recurrence"] = True

    return ExperimentConfig(
        experiment_name=EXPERIMENT_PREFIX,  # validator appends _<dataset>
        run_name=f"{architecture}_ws{window_size}",
        data_path=str(Path(DATA_DIR) / csv_filename),
        architecture=architecture,
        data_mode="win",
        model_kwargs=model_kwargs,
        window_size=window_size,
        bptt_steps=bptt_steps,
        lr=LR,
        epochs=epochs,
        loss_type="mse",
    )


def main():
    parser = argparse.ArgumentParser(description="Stateful vs stateless window-size study")
    parser.add_argument("--smoke", action="store_true",
                        help="Quick pipeline check: 1 dataset, 2 windows, few epochs.")
    parser.add_argument("--only", type=str, default=None,
                        help="Comma-separated dataset labels to run (default: all). "
                             "Use this to add new datasets without re-running finished ones.")
    args = parser.parse_args()

    mlflow.set_tracking_uri(f"sqlite:///{DB_PATH}")

    if args.smoke:
        datasets = DATASETS[:1]
        window_grid = [1, 8]
        epochs, bptt_steps, patience = 5, 50, 5
        print(">>> SMOKE MODE: 1 dataset x 2 architectures x 2 windows, 5 epochs each")
    else:
        datasets = DATASETS
        window_grid = WINDOW_GRID
        epochs, bptt_steps, patience = EPOCHS, BPTT_STEPS, PATIENCE

    if args.only:
        wanted = {s.strip() for s in args.only.split(",")}
        datasets = [d for d in datasets if d[0] in wanted]
        missing = wanted - {d[0] for d in datasets}
        if missing:
            raise SystemExit(f"--only names unknown datasets: {sorted(missing)}")
        print(f">>> --only: running {[d[0] for d in datasets]}")

    total = len(datasets) * len(ARCHITECTURES) * len(window_grid)
    print(f"{'=' * 72}")
    print(f"  WINDOW STUDY — {total} runs  (device={DEVICE}, patience={patience}, epochs={epochs})")
    print(f"  Datasets:      {[d[0] for d in datasets]}")
    print(f"  Architectures: {[a[0] for a in ARCHITECTURES]}")
    print(f"  Window grid:   {window_grid}")
    print(f"{'=' * 72}")

    results = []
    run_idx = 0
    t_start = time.time()

    for label, csv_filename in datasets:
        for architecture, sever in ARCHITECTURES:
            for window_size in window_grid:
                run_idx += 1
                print(f"\n{'-' * 72}")
                print(f"  Run {run_idx}/{total}: {label} | {architecture} | ws={window_size}")
                print(f"{'-' * 72}")

                seed_everything(SEED)
                config = build_config(csv_filename, architecture, sever, window_size, epochs, bptt_steps)

                # Pin all runs for a dataset into one experiment so both architectures
                # are plottable together. config.experiment_name == "Window_Study_<dataset>".
                mlflow.set_experiment(config.experiment_name)

                try:
                    result = run_experiment(config, patience=patience, device=DEVICE)
                    val_loss = result["best_val_loss"]
                    results.append({
                        "dataset": label,
                        "architecture": architecture,
                        "window_size": window_size,
                        "best_val_loss": val_loss,
                        "run_id": result["run_id"],
                    })
                    print(f"  -> best_val_loss = {val_loss:.6f}")
                except Exception as exc:  # keep the study going if one cell fails
                    print(f"  !! FAILED: {type(exc).__name__}: {exc}")
                    results.append({
                        "dataset": label,
                        "architecture": architecture,
                        "window_size": window_size,
                        "best_val_loss": float("nan"),
                        "run_id": None,
                    })

    elapsed = time.time() - t_start
    print(f"\n{'=' * 72}")
    print(f"  STUDY COMPLETE — {run_idx} runs in {elapsed / 60:.1f} min")
    print(f"{'=' * 72}")
    for label, _ in datasets:
        print(f"\n  {label}:")
        for arch, _ in ARCHITECTURES:
            row = [r for r in results if r["dataset"] == label and r["architecture"] == arch]
            row.sort(key=lambda r: r["window_size"])
            curve = "  ".join(f"ws{r['window_size']}={r['best_val_loss']:.4f}" for r in row)
            print(f"    {arch:38s}  {curve}")


if __name__ == "__main__":
    main()
