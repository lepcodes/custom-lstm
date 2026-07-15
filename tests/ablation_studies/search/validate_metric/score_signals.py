"""
Score candidate forget signals against labeled synthetic regimes — the "ruler".
===============================================================================

For every (scenario x signal x seed) this computes:

  auc        — ROC AUC of the signal as a per-timestep classifier of "past is
               irrelevant" (forget_label). Only defined when the scenario has both
               classes (mean_shift / var_shift are all-zero by design → NaN).
  spearman   — rank correlation between signal and forget_label (same caveat).
  det_lag    — median steps after a change point until the signal deviates from its
               pre-change baseline by > K robust stds (two-sided, horizon-capped).
  miss_rate  — fraction of change points never detected within the horizon.
  regime_gap — mean signal in the odd (noise/second) regime minus mean in the even
               (structured/first) regime. Positive = the signal rises when structure
               disappears. For arch_switch the CORRECT value is ~0 (the past never helps
               an MSE forecast in either regime), so a large gap there is a false alarm.

AUC/spearman measure *sustained* tracking of "the past is irrelevant" (what a gate
modulator needs); det_lag/miss_rate measure *responsiveness* at the switch itself
(what a drift detector needs). A signal can win one and lose the other — that split
is exactly the dependence-vs-regime-change distinction the thesis needs to keep straight.

Usage:
    python -m tests.ablation_studies.search.validate_metric.score_signals
    python -m tests.ablation_studies.search.validate_metric.score_signals --smoke
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
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from tests.ablation_studies.search.validate_metric.generate_regimes import SCENARIOS
from tests.ablation_studies.search.validate_metric.signals import SIGNALS

OUT_DIR = Path("resources/metric_ruler")

# Fixed categorical assignment (validated, slot order — never cycled)
SIGNAL_COLORS = {
    "ewacf_mean": "#2a78d6",  # blue
    "ewacf_max": "#1baf7a",   # aqua
    "mi_knn": "#4a3aa7",      # violet
    "ks_shift": "#e34948",    # red
}
INK, MUTED, GRID = "#0b0b0b", "#898781", "#e1e0d9"

DET_HORIZON = 150     # steps a signal has to react to a change point
DET_BASELINE = 150    # pre-change window that defines the baseline
DET_K = 3.0           # robust-std multiplier for "reacted"


def detection_lags(sig: np.ndarray, change_points: np.ndarray,
                   horizon: int = DET_HORIZON) -> tuple[float, float]:
    """Median detection lag and miss rate over all change points (two-sided robust
    exceedance vs the pre-change baseline). NaN-lag change points count as misses."""
    lags, misses = [], 0
    for c in change_points:
        base = sig[max(0, c - DET_BASELINE):c]
        base = base[~np.isnan(base)]
        if len(base) < 20:
            continue
        m = np.median(base)
        s = max(1.4826 * np.median(np.abs(base - m)), 1e-3)
        window = sig[c:c + horizon]
        hits = np.where(np.abs(window - m) > DET_K * s)[0]
        if len(hits):
            lags.append(hits[0])
        else:
            misses += 1
    n = len(lags) + misses
    if n == 0:
        return np.nan, np.nan
    return (float(np.median(lags)) if lags else np.nan), misses / n


def score_one(sig: np.ndarray, series) -> dict:
    mask = ~np.isnan(sig)
    y, s = series.forget_label[mask], sig[mask]
    row = {}
    if y.min() != y.max():
        row["auc"] = roc_auc_score(y, s)
        row["spearman"] = spearmanr(s, y).statistic
    else:
        row["auc"], row["spearman"] = np.nan, np.nan
    row["det_lag"], row["miss_rate"] = detection_lags(sig, series.change_points)
    r = series.regime_id[mask]
    row["regime_gap"] = float(np.mean(s[r == 1]) - np.mean(s[r == 0]))
    return row


def plot_scenario(series, sigs: dict[str, np.ndarray], out: Path):
    """Small multiples: raw series on top, one panel per signal below, ground truth
    shaded on every panel. Identity is carried by panel titles, not color alone."""
    n = 1 + len(sigs)
    fig, axes = plt.subplots(n, 1, figsize=(12, 1.9 * n), sharex=True)
    fig.suptitle(f"{series.name} — {series.description}", fontsize=11, color=INK)

    def shade(ax):
        lab = series.forget_label
        if lab.any():
            ax.fill_between(np.arange(len(lab)), 0, 1, where=lab == 1,
                            transform=ax.get_xaxis_transform(),
                            color=MUTED, alpha=0.15, linewidth=0)
        for c in series.change_points:
            ax.axvline(c, color=MUTED, linewidth=0.8, linestyle="--", alpha=0.6)

    axes[0].plot(series.x, color=INK, linewidth=0.6)
    axes[0].set_title("series (shaded = ground truth 'past irrelevant', dashes = change points)",
                      fontsize=9, loc="left", color=MUTED)
    shade(axes[0])

    for ax, (name, sig) in zip(axes[1:], sigs.items()):
        ax.plot(sig, color=SIGNAL_COLORS[name], linewidth=1.6)
        ax.set_ylim(-0.05, 1.05)
        ax.set_title(f"{name}  (1 = past irrelevant)", fontsize=9, loc="left",
                     color=SIGNAL_COLORS[name], fontweight="bold")
        shade(ax)

    for ax in axes:
        ax.grid(True, color=GRID, linewidth=0.6)
        ax.tick_params(colors=MUTED, labelsize=8)
        for spine in ax.spines.values():
            spine.set_color(GRID)
    axes[-1].set_xlabel("t", color=MUTED)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--T", type=int, default=4000)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--smoke", action="store_true", help="T=1500, 2 seeds")
    parser.add_argument("--out", type=str, default=str(OUT_DIR))
    args = parser.parse_args()
    if args.smoke:
        args.T, args.seeds = 1500, 2

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    for scen_name, gen in SCENARIOS.items():
        for seed in range(args.seeds):
            series = gen(T=args.T, seed=seed)
            sigs = {name: fn(series.x, seed=seed) for name, fn in SIGNALS.items()}
            for sig_name, sig in sigs.items():
                row = {"scenario": scen_name, "signal": sig_name, "seed": seed}
                row.update(score_one(sig, series))
                rows.append(row)
            if seed == 0:
                plot_scenario(series, sigs, out_dir / f"scenario_{scen_name}.png")
        print(f"scored {scen_name} ({args.seeds} seeds)")

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "ruler_results_raw.csv", index=False)

    agg = (df.groupby(["scenario", "signal"])
             .agg(auc_mean=("auc", "mean"), auc_std=("auc", "std"),
                  spearman_mean=("spearman", "mean"),
                  det_lag_median=("det_lag", "median"),
                  miss_rate_mean=("miss_rate", "mean"),
                  regime_gap_mean=("regime_gap", "mean"))
             .round(3).reset_index())
    agg.to_csv(out_dir / "ruler_results.csv", index=False)

    print("\n=== Ruler results (mean over seeds) ===")
    print(agg.to_string(index=False))
    print(f"\nCSV + figures -> {out_dir}")


if __name__ == "__main__":
    main()
