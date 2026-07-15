"""
Plot the overfitting diagnostic (per dataset), from the fixed-budget runs.

Reads the `Overfit_<dataset>` MLflow experiments (run_overfitting_study.py) and, for
each smoothing filter, writes into deeper, separated folders under
`resources/window_study/overfitting/`:

    summary/<filter>/overfit_<dataset>.png        (+ overfit_results.csv)
    epochs/<filter>/overfit_epochs_<dataset>.png

where <filter> is `sma` or `ewma`:
  SMA  — simple moving average: mean of the last K epochs   (--k,    default 20)
  EWMA — exponentially weighted moving average, span S       (--ewma, default 20)

  SUMMARY: x = window (log2), blue = stateful (A) / red = stateless (C),
           solid = train / dashed = val; each point = the filter's "last value",
           shaded band = ±1 std of the filter.
  EPOCHS (W=1 vs W=128): raw (faint) + filtered curve (bold) + last-value markers.

Usage:
    python -m tests.ablation_studies.search.validate_window.plot_overfitting                 # both filters
    python -m tests.ablation_studies.search.validate_window.plot_overfitting --k 15 --ewma 30
    python -m tests.ablation_studies.search.validate_window.plot_overfitting --filters sma   # one only
"""

import argparse
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
import numpy as np
import pandas as pd

DB_PATH = "C:/Users/Luis/Documents/ML-AI-Projects/custom-lstm/tests/ablation_studies/mlflow.db"
BASE_OUT = "C:/Users/Luis/Documents/ML-AI-Projects/custom-lstm/resources/window_study/overfitting"
DATASETS = ["aqua_alta", "lbl_tcp_3", "mackey_glass", "ethernet_traffic", "total_sunspots", "etth1", "exchange_rate"]
EXPERIMENT_PREFIX = "Overfit"
DIAG_WINDOWS = [1, 128]  # windows shown in the epoch-curve diagnostic

# Keep the blue/red convention; solid=train, dashed=val.
ARCH = {
    "lstm_vanilla_windowed": {"label": "Stateful", "color": "#1f77b4"},
    "lstm_vanilla_stateless_windowed": {"label": "Stateless", "color": "#d62728"},
}

FILT_LABEL = {"sma": lambda p: f"SMA (last {p})", "ewma": lambda p: f"EWMA (span {p})"}


def collect() -> dict:
    """{(dataset, arch, window): {'train':[...], 'val':[...]}} using the latest run per cell."""
    mlflow.set_tracking_uri(f"sqlite:///{DB_PATH}")
    client = mlflow.MlflowClient()
    out = {}
    for dataset in DATASETS:
        exp = client.get_experiment_by_name(f"{EXPERIMENT_PREFIX}_{dataset}")
        if exp is None:
            continue
        latest = {}
        for r in client.search_runs([exp.experiment_id], max_results=5000):
            arch = r.data.params.get("architecture", "").split(".")[-1].lower()
            ws = r.data.params.get("window_size")
            if arch not in ARCH or ws is None:
                continue
            key = (arch, int(ws))
            if key not in latest or r.info.start_time > latest[key].info.start_time:
                latest[key] = r
        for (arch, ws), r in latest.items():
            th = client.get_metric_history(r.info.run_id, "train_mse")
            vh = client.get_metric_history(r.info.run_id, "val_mse")
            if not th or not vh:
                continue
            train = [m.value for m in sorted(th, key=lambda m: m.step)]
            val = [m.value for m in sorted(vh, key=lambda m: m.step)]
            out[(dataset, arch, ws)] = {"train": train, "val": val}
    return out


def filtered_last(values, filt, param):
    """The filter's 'last value' + a ±std band. SMA: mean/std of last K. EWMA: ewm mean/std."""
    s = pd.Series(values, dtype=float)
    if filt == "sma":
        tail = s.iloc[-param:]
        return float(tail.mean()), float(tail.std(ddof=0))
    # adjust=True -> bias-corrected weighting (no artificial "head start" from the
    # first, untrained epoch). Irrelevant to the LAST value anyway: after ~5*span
    # epochs the init transient has fully decayed.
    m = s.ewm(span=param, adjust=True).mean().iloc[-1]
    sd = s.ewm(span=param, adjust=True).std().iloc[-1]
    return float(m), float(0.0 if pd.isna(sd) else sd)


def filtered_curve(values, filt, param):
    """The filtered curve over epochs (SMA rolling mean, or EWMA with bias-corrected init)."""
    s = pd.Series(values, dtype=float)
    return s.rolling(param, min_periods=1).mean() if filt == "sma" else s.ewm(span=param, adjust=True).mean()


def make_summary(data: dict, filt: str, param: int, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    rows = []
    for dataset in DATASETS:
        present = {(a, w) for (d, a, w) in data if d == dataset}
        if not present:
            continue
        fig, ax = plt.subplots(figsize=(7.5, 5.5))
        for arch, style in ARCH.items():
            ws_list = sorted(w for (a, w) in present if a == arch)
            if not ws_list:
                continue
            tr_m, tr_s, va_m, va_s = [], [], [], []
            for w in ws_list:
                tm, ts = filtered_last(data[(dataset, arch, w)]["train"], filt, param)
                vm, vs = filtered_last(data[(dataset, arch, w)]["val"], filt, param)
                tr_m.append(tm)
                tr_s.append(ts)
                va_m.append(vm)
                va_s.append(vs)
                rows.append({"dataset": dataset, "architecture": arch, "window_size": w, "filter": filt,
                             "train_mse": tm, "train_std": ts, "val_mse": vm, "val_std": vs})
            x = np.array(ws_list)
            tr_m, tr_s, va_m, va_s = map(np.array, (tr_m, tr_s, va_m, va_s))
            ax.plot(x, tr_m, color=style["color"], ls="-", marker="o", lw=2, ms=6, label=f"{style['label']} — train")
            ax.fill_between(x, tr_m - tr_s, tr_m + tr_s, color=style["color"], alpha=0.12)
            ax.plot(x, va_m, color=style["color"], ls="--", marker="s", lw=2, ms=6, label=f"{style['label']} — val")
            ax.fill_between(x, va_m - va_s, va_m + va_s, color=style["color"], alpha=0.12)
        ax.set_xscale("log", base=2)
        ax.set_xlabel("Window size")
        ax.set_ylabel(f"MSE — {FILT_LABEL[filt](param)}")
        ax.set_title(dataset)
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=9)
        fig.tight_layout()
        path = os.path.join(out_dir, f"overfit_{dataset}.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"  saved {path}")
    if rows:
        csv = os.path.join(out_dir, "overfit_results.csv")
        pd.DataFrame(rows).to_csv(csv, index=False)
        print(f"  saved {csv}")


def make_diagnostics(data: dict, filt: str, param: int, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    for dataset in DATASETS:
        windows = [w for w in DIAG_WINDOWS if any((dataset, a, w) in data for a in ARCH)]
        if not windows:
            continue
        fig, axes = plt.subplots(1, len(windows), figsize=(6.5 * len(windows), 5.0), squeeze=False)
        for ax, w in zip(axes[0], windows):
            for arch, style in ARCH.items():
                cell = data.get((dataset, arch, w))
                if not cell:
                    continue
                ep = np.arange(1, len(cell["train"]) + 1)
                c = style["color"]
                ax.plot(ep, cell["train"], color=c, ls="-", lw=0.8, alpha=0.35)
                ax.plot(ep, cell["val"], color=c, ls="--", lw=0.8, alpha=0.35)
                ax.plot(ep, filtered_curve(cell["train"], filt, param), color=c, ls="-", lw=2, label=f"{style['label']} — train")
                ax.plot(ep, filtered_curve(cell["val"], filt, param), color=c, ls="--", lw=2, label=f"{style['label']} — val")
                tm, _ = filtered_last(cell["train"], filt, param)
                vm, _ = filtered_last(cell["val"], filt, param)
                ax.plot(ep[-1], tm, color=c, marker="o", ms=8)
                ax.plot(ep[-1], vm, color=c, marker="s", ms=8)
            ax.set_xlabel("Epoch")
            ax.set_ylabel("MSE")
            ax.set_title(f"{dataset} — window {w}")
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)
        fig.suptitle(f"Training dynamics — raw (faint) vs {FILT_LABEL[filt](param)} (bold); "
                     f"markers = filtered last value", fontsize=11)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        path = os.path.join(out_dir, f"overfit_epochs_{dataset}.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"  saved {path}")


def main():
    ap = argparse.ArgumentParser(description="Plot the overfitting diagnostic per dataset (SMA and/or EWMA).")
    ap.add_argument("--k", type=int, default=20, help="SMA window (epochs) for the moving-average 'last value'.")
    ap.add_argument("--ewma", type=int, default=20, help="EWMA span (epochs) for the exponential 'last value'.")
    ap.add_argument("--filters", default="sma,ewma", help="Which filters to render: 'sma', 'ewma', or both (comma-sep).")
    args = ap.parse_args()

    data = collect()
    if not data:
        print("No Overfit_* runs found. Run run_overfitting_study.py first.")
        return

    params = {"sma": args.k, "ewma": args.ewma}
    for filt in [f.strip() for f in args.filters.split(",") if f.strip()]:
        if filt not in params:
            print(f"  [skip] unknown filter '{filt}'")
            continue
        print(f"=== filter: {FILT_LABEL[filt](params[filt])} ===")
        make_summary(data, filt, params[filt], os.path.join(BASE_OUT, "summary", filt))
        make_diagnostics(data, filt, params[filt], os.path.join(BASE_OUT, "epochs", filt))


if __name__ == "__main__":
    main()
