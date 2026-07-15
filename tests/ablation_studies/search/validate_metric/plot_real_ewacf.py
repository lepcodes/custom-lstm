"""
Dependence signals per real dataset — EW-ACF (director's ask) + MI (linearity check).
=====================================================================================

Computes on each of the 7 study datasets:

  EW-ACF dependence (mean/max |EW-ACF| over lags) — Dra. Arana's ask (2026-07-08):
    mackey_glass -> predicted high; aqua_alta -> predicted ~0 (tidal caveat applies).
  MI dependence (rolling kNN, same lags) — tests whether the low-EW-ACF verdicts
    (ethernet_traffic, lbl_tcp_3) are real or a *linearity artifact*: bursty traffic can
    carry nonlinear structure that correlation cannot see (cf. nonlinear_map_switch in
    the ruler, where EW-ACF is blind at AUC 0.49 and MI scores 0.87).
  MI on a SHUFFLED copy — the honesty baseline. kNN MI is positively biased (pure noise
    reads as dependence), so only MI in EXCESS of the shuffled floor counts as structure.

The other half of the director's ask (relating signals to *forget-gate activations* via
GateTelemetry) needs trained models and is a separate step.

Usage:
    python -m tests.ablation_studies.search.validate_metric.plot_real_ewacf
"""

import argparse
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tests.ablation_studies.search.validate_metric.signals import (
    DEFAULT_LAGS,
    DEFAULT_LAMBDA,
    ewacf_forget,
    mi_forget,
)

DATA_DIR = Path("data/preprocessed")
OUT_DIR = Path("resources/metric_ruler/real_datasets")

DATASETS = ["aqua_alta", "lbl_tcp_3", "mackey_glass", "ethernet_traffic",
            "total_sunspots", "etth1", "exchange_rate"]

INK, MUTED, GRID = "#0b0b0b", "#898781", "#e1e0d9"
BLUE, AQUA, VIOLET = "#2a78d6", "#1baf7a", "#4a3aa7"


def load_series(name: str) -> np.ndarray:
    """First column, numeric-coerced (mackey_glass has a bare '0' header)."""
    df = pd.read_csv(DATA_DIR / f"{name}.csv")
    x = pd.to_numeric(df.iloc[:, 0], errors="coerce").dropna().to_numpy(dtype=float)
    return x


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lambda", dest="lambda_", type=float, default=DEFAULT_LAMBDA)
    parser.add_argument("--lags", type=str, default=",".join(map(str, DEFAULT_LAGS)))
    args = parser.parse_args()
    lags = [int(v) for v in args.lags.split(",")]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary = []

    rng = np.random.default_rng(42)
    for name in DATASETS:
        x = load_series(name)
        # dependence scale (1 = strongly autocorrelated) = 1 - forget scale
        dep_mean = 1.0 - ewacf_forget(x, lags=lags, lambda_=args.lambda_, agg="mean")
        dep_max = 1.0 - ewacf_forget(x, lags=lags, lambda_=args.lambda_, agg="max")
        dep_mi = 1.0 - mi_forget(x, lags=lags)
        dep_mi_shuf = 1.0 - mi_forget(rng.permutation(x), lags=lags)  # kNN-bias floor

        fig, axes = plt.subplots(3, 1, figsize=(12, 6.4), sharex=True)
        fig.suptitle(f"{name} — dependence signals (lags {lags}, λ={args.lambda_})",
                     fontsize=11, color=INK)
        axes[0].plot(x, color=INK, linewidth=0.5)
        axes[0].set_title("series", fontsize=9, loc="left", color=MUTED)
        axes[1].plot(dep_mean, color=BLUE, linewidth=1.2)
        axes[1].plot(dep_max, color=AQUA, linewidth=1.2, alpha=0.8)
        axes[1].set_title("mean |EW-ACF| (blue) · max |EW-ACF| (aqua) — 1 = past informative",
                          fontsize=9, loc="left", color=MUTED)
        axes[2].plot(dep_mi, color=VIOLET, linewidth=1.4)
        axes[2].plot(dep_mi_shuf, color=MUTED, linewidth=1.0, linestyle="--", alpha=0.8)
        axes[2].set_title("MI dependence (violet) vs shuffled-series bias floor (gray dashed) — "
                          "only the excess over the floor is real structure",
                          fontsize=9, loc="left", color=MUTED)
        axes[2].set_xlabel("t", color=MUTED)
        for ax in axes[1:]:
            ax.set_ylim(-0.05, 1.05)
        for ax in axes:
            ax.grid(True, color=GRID, linewidth=0.6)
            ax.tick_params(colors=MUTED, labelsize=8)
            for spine in ax.spines.values():
                spine.set_color(GRID)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        fig.savefig(OUT_DIR / f"ewacf_{name}.png", dpi=150)
        plt.close(fig)

        valid = dep_mean[~np.isnan(dep_mean)]
        mi_valid = dep_mi[~np.isnan(dep_mi)]
        mi_floor = float(np.median(dep_mi_shuf[~np.isnan(dep_mi_shuf)]))
        summary.append({
            "dataset": name,
            "median_dep_ewacf": np.median(valid),
            "frac_below_0.3": float(np.mean(valid < 0.3)),
            "median_dep_mi": float(np.median(mi_valid)),
            "mi_bias_floor": mi_floor,
            "mi_excess": float(np.median(mi_valid)) - mi_floor,
        })
        print(f"plotted {name}")

    df = pd.DataFrame(summary).sort_values("median_dep_ewacf", ascending=False).round(3)
    df.to_csv(OUT_DIR / "ewacf_summary.csv", index=False)
    print("\n=== Dependence per dataset (1 = past informative; mi_excess = MI above shuffle floor) ===")
    print(df.to_string(index=False))
    print(f"\nFigures + CSV -> {OUT_DIR}")


if __name__ == "__main__":
    main()
