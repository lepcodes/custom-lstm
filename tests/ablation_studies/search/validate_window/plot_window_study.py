"""
Plot the Window-Size Study: Stateful vs Stateless LSTM
======================================================

Reads the runs logged by run_window_study.py from the MLflow SQLite store and renders
one overlaid curve per dataset:

    x = window size (log2 axis)
    y = best validation MSE (early-stopped, pure MSE)
    two lines: Stateful LSTM  vs  Stateless control (sever_recurrence)

For each (dataset, architecture, window_size) cell we keep the MOST RECENT run, so a
fresh full sweep automatically supersedes earlier/smoke runs.

Outputs:
    resources/window_study/window_study.png
    resources/window_study/window_study_results.csv

Usage:
    python -m tests.ablation_studies.search.validate_window.plot_window_study
"""

import math
import os
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import pandas as pd

DB_PATH = "C:/Users/Luis/Documents/ML-AI-Projects/custom-lstm/tests/ablation_studies/mlflow.db"
OUT_DIR = "C:/Users/Luis/Documents/ML-AI-Projects/custom-lstm/resources/window_study"

DATASETS = ["aqua_alta", "lbl_tcp_3", "mackey_glass", "ethernet_traffic", "total_sunspots", "etth1", "exchange_rate"]
EXPERIMENT_PREFIX = "Window_Study"

# architecture -> (legend label, plotting style). Three paradigms:
#   A  stateful LSTM         — window stacked as features, state carried across series
#   B  memoryless baseline   — window as features, NO recurrence (sever_recurrence)
#   C  stateless windowed LSTM (Diego) — window unrolled as time, state reset between windows
ARCH_STYLE = {
    "lstm_vanilla_windowed": {
        "label": "Stateful LSTM (recurrence carries memory)",
        "color": "#1f77b4", "marker": "o", "ls": "-",
    },
    "lstm_vanilla_windowed_no_recurrence": {
        "label": "Memoryless window baseline (no recurrence)",
        "color": "#7f7f7f", "marker": "^", "ls": ":",
    },
    "lstm_vanilla_stateless_windowed": {
        "label": "Stateless windowed LSTM (Diego: window = recurrence depth)",
        "color": "#d62728", "marker": "s", "ls": "--",
    },
}


def collect_results() -> pd.DataFrame:
    """Pull all window-study runs into a tidy, de-duplicated DataFrame."""
    mlflow.set_tracking_uri(f"sqlite:///{DB_PATH}")
    client = mlflow.MlflowClient()

    rows = []
    for dataset in DATASETS:
        exp = client.get_experiment_by_name(f"{EXPERIMENT_PREFIX}_{dataset}")
        if exp is None:
            print(f"  [warn] no experiment for {dataset}")
            continue
        runs = client.search_runs([exp.experiment_id], max_results=5000)
        for r in runs:
            ws = r.data.params.get("window_size")
            arch = r.data.params.get("architecture")
            val = r.data.metrics.get("best_val_loss")
            if ws is None or arch is None or val is None:
                continue
            # A/B log the enum's str ("ArchitectureType.LSTM_VANILLA_WINDOWED"),
            # C logs a plain string. Normalize both onto the ARCH_STYLE keys.
            arch = arch.split(".")[-1].lower()
            rows.append({
                "dataset": dataset,
                "architecture": arch,
                "window_size": int(ws),
                "best_val_loss": float(val),
                "start_time": r.info.start_time,
            })

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # Keep the most recent run per (dataset, architecture, window_size) cell.
    df = (
        df.sort_values("start_time")
          .drop_duplicates(["dataset", "architecture", "window_size"], keep="last")
          .sort_values(["dataset", "architecture", "window_size"])
          .reset_index(drop=True)
    )
    return df


DEFAULT_SUPTITLE = (
    "Window size vs best validation MSE (lower is better)  —  "
    "stateful · memoryless · stateless windowed (Diego)\n"
    "'Bigger window → lower MSE' is not universal: it depends on the series. "
    "Where the stateful model is already best at window=1, its recurrence — not the "
    "window — supplies the memory."
)


def make_plot(df: pd.DataFrame, arch_style: dict = None, out_name: str = "window_study",
              suptitle: str = None) -> str:
    arch_style = arch_style if arch_style is not None else ARCH_STYLE
    os.makedirs(OUT_DIR, exist_ok=True)
    datasets = [d for d in DATASETS if d in set(df["dataset"])]
    n = len(datasets)

    ncols = min(3, n)
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.2 * ncols, 4.8 * nrows), squeeze=False)
    flat = axes.flatten()

    for ax, dataset in zip(flat, datasets):
        sub = df[df["dataset"] == dataset]
        for arch, style in arch_style.items():
            cur = sub[sub["architecture"] == arch].sort_values("window_size")
            if cur.empty:
                continue
            ax.plot(
                cur["window_size"], cur["best_val_loss"],
                label=style["label"], color=style["color"],
                marker=style["marker"], linestyle=style["ls"], linewidth=2, markersize=7,
            )
        ax.set_xscale("log", base=2)
        ax.set_xlabel("Window size (input features per step)")
        ax.set_ylabel("Best validation MSE (early-stopped)")
        ax.set_title(dataset)
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=8, loc="best")

    for ax in flat[n:]:          # hide any unused grid cells
        ax.axis("off")

    fig.suptitle(suptitle if suptitle is not None else DEFAULT_SUPTITLE, fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.90 if nrows > 1 else 0.92])

    png_path = os.path.join(OUT_DIR, f"{out_name}.png")
    fig.savefig(png_path, dpi=150)
    print(f"  saved {png_path}")
    return png_path


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Plot the window study (all models, or a subset).")
    ap.add_argument("--drop", default="", help="Comma-separated ARCH_STYLE keys to exclude "
                    "(e.g. lstm_vanilla_windowed_no_recurrence to drop the memoryless baseline).")
    ap.add_argument("--out", default="window_study", help="Output PNG stem (no extension).")
    ap.add_argument("--title", default=None, help="Custom suptitle (defaults to the standard one).")
    args = ap.parse_args()

    df = collect_results()
    if df.empty:
        print("No window-study runs found. Run run_window_study.py first.")
        return

    os.makedirs(OUT_DIR, exist_ok=True)
    csv_path = os.path.join(OUT_DIR, "window_study_results.csv")
    df.drop(columns=["start_time"]).to_csv(csv_path, index=False)
    print(f"  saved {csv_path}")

    dropped = {s.strip() for s in args.drop.split(",") if s.strip()}
    arch_style = {k: v for k, v in ARCH_STYLE.items() if k not in dropped}

    # Console summary (only the architectures being plotted)
    for dataset in DATASETS:
        sub = df[df["dataset"] == dataset]
        if sub.empty:
            continue
        print(f"\n  {dataset}:")
        for arch in arch_style:
            cur = sub[sub["architecture"] == arch].sort_values("window_size")
            if cur.empty:
                continue
            curve = "  ".join(f"ws{int(r.window_size)}={r.best_val_loss:.4f}" for r in cur.itertuples())
            print(f"    {arch:38s}  {curve}")

    make_plot(df, arch_style=arch_style, out_name=args.out, suptitle=args.title)


if __name__ == "__main__":
    main()
