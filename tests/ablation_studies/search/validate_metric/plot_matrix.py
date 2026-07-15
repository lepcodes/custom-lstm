"""
Full signal x scenario comparison matrix — every metric for every scenario.
===========================================================================

Reads `resources/metric_ruler/ruler_results.csv` (produced by score_signals.py) and renders
two annotated heatmaps side by side:

  AUC        — sustained tracking of the forget label (dependence scenarios only;
               single-class scenarios are hatched out). Diverging around 0.5 = chance:
               blue above (informative), red below (anti-correlated — actively misleading).
  regime_gap — mean signal in noise regime minus structured regime, ALL scenarios.
               Diverging around 0. For arch_switch ~0 is the CORRECT answer.

This is the anti-cherry-picking view: each signal's behavior is visible in every scenario,
not only in the scenario it was designed to win.

Usage:
    python -m tests.ablation_studies.search.validate_metric.plot_matrix
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
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

IN_CSV = Path("resources/metric_ruler/ruler_results.csv")
OUT_PNG = Path("resources/metric_ruler/ruler_matrix.png")

# fixed display order: scenario sections, then signals by family
SCENARIO_ORDER = [
    "ar_noise_switch", "phi_switch", "periodic_noise_switch",       # 1: linear switches
    "mean_shift", "var_shift",                                      # 2: distribution shifts
    "seasonal_ar_switch", "off_grid_lag_switch",                    # 3: multi-lag
    "nonlinear_map_switch", "arch_switch",                          # 4: nonlinear corners
]
SIGNAL_ORDER = ["ewacf_max", "ewacf_mean", "mi_knn", "ks_shift"]

INK, MUTED, GRID_C = "#0b0b0b", "#898781", "#e1e0d9"
# diverging blue <-> red with neutral gray midpoint (reference palette)
DIVERGING = LinearSegmentedColormap.from_list("bwr_ref", ["#d03b3b", "#f0efec", "#2a78d6"])


def _heatmap(ax, mat: pd.DataFrame, norm, title: str, fmt: str):
    ax.imshow(mat.to_numpy(dtype=float), cmap=DIVERGING, norm=norm, aspect="auto")
    ax.set_xticks(range(len(mat.columns)), mat.columns, fontsize=9, color=INK)
    ax.set_yticks(range(len(mat.index)), mat.index, fontsize=9, color=INK)
    ax.set_title(title, fontsize=10, loc="left", color=INK, pad=10)
    for i in range(len(mat.index)):
        for j in range(len(mat.columns)):
            v = mat.iloc[i, j]
            if np.isnan(v):
                ax.text(j, i, "—", ha="center", va="center", fontsize=9, color=MUTED)
            else:
                ax.text(j, i, format(v, fmt), ha="center", va="center", fontsize=9, color=INK)
    ax.set_xticks(np.arange(-0.5, len(mat.columns)), minor=True)
    ax.set_yticks(np.arange(-0.5, len(mat.index)), minor=True)
    ax.grid(which="minor", color="#fcfcfb", linewidth=2)
    ax.tick_params(which="both", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default=str(IN_CSV))
    parser.add_argument("--out", type=str, default=str(OUT_PNG))
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    auc = (df.pivot(index="scenario", columns="signal", values="auc_mean")
             .reindex(index=SCENARIO_ORDER, columns=SIGNAL_ORDER))
    gap = (df.pivot(index="scenario", columns="signal", values="regime_gap_mean")
             .reindex(index=SCENARIO_ORDER, columns=SIGNAL_ORDER))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    fig.suptitle("Ruler matrix — every signal on every scenario (mean over seeds)",
                 fontsize=12, color=INK)
    _heatmap(axes[0], auc, TwoSlopeNorm(vcenter=0.5, vmin=0.25, vmax=1.0),
             "AUC vs forget label (0.5 = chance; red = anti-correlated; — = single-class label)",
             ".2f")
    _heatmap(axes[1], gap, TwoSlopeNorm(vcenter=0.0, vmin=-0.1, vmax=0.55),
             "regime_gap: signal(noise) − signal(structured)\n(arch_switch: ≈0 is the correct answer)",
             "+.2f")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(args.out, dpi=150)
    print(f"matrix -> {args.out}")


if __name__ == "__main__":
    main()
