"""
Candidate forget signals, all normalized to a [0,1] "forget scale".
===================================================================

Contract (CLAUDE.md §4 / handoff): every signal maps a raw series to a per-timestep
value where **1 = the past is irrelevant** (safe to forget) and **0 = the past is
informative** (retain). All signals are causal: the value at t uses only x[0..t].
Undefined positions (burn-in) are NaN and are masked out by the scorer.

Three families, matching the metric-design axes:
  ewacf_mean / ewacf_max — linear dependence (EW-ACF over lags, teacher's aggregations)
  mi_knn                 — nonlinear dependence (rolling kNN mutual information)
  ks_shift               — distribution shift (rolling two-window Kolmogorov-Smirnov)
"""

import numpy as np
import torch
from scipy.stats import ks_2samp
from sklearn.feature_selection import mutual_info_regression

from custom_lstm.utils import EWACFEngine

DEFAULT_LAGS = [1, 2, 4, 8]   # matches notebooks/exp_ew_acf_multilag.ipynb
DEFAULT_LAMBDA = 0.9


def ewacf_forget(x: np.ndarray, lags=None, lambda_: float = DEFAULT_LAMBDA,
                 agg: str = "mean") -> np.ndarray:
    """1 - agg_over_lags(|EW-ACF|). Uses the exact online engine the trainer uses."""
    lags = lags or DEFAULT_LAGS
    engine = EWACFEngine(lambda_=lambda_, lags=lags)
    seq = torch.tensor(x, dtype=torch.float32).view(1, -1, 1)
    with torch.no_grad():
        acf = engine(seq)  # (1, T - max_lag, 1, L)
    a = acf[0, :, 0, :].abs()
    dep = a.mean(dim=1) if agg == "mean" else a.max(dim=1).values
    out = np.full(len(x), np.nan)
    out[max(lags):] = np.clip(1.0 - dep.numpy(), 0.0, 1.0)
    return out


def _forward_fill(out: np.ndarray) -> np.ndarray:
    """Causal fill of strided computations: hold the last computed value."""
    idx = np.where(~np.isnan(out))[0]
    if len(idx) == 0:
        return out
    filled = out.copy()
    last = np.nan
    for t in range(idx[0], len(out)):
        if not np.isnan(out[t]):
            last = out[t]
        filled[t] = last
    return filled


def mi_forget(x: np.ndarray, lags=None, window: int = 200, stride: int = 10,
              n_neighbors: int = 3, seed: int = 0) -> np.ndarray:
    """1 - correlation-equivalent of the mean kNN mutual information between x_t and
    x_{t-lag} inside a trailing window. MI is mapped to [0,1] via the Gaussian
    equivalence r = sqrt(1 - exp(-2*MI)) so it lives on the same scale as |ACF|."""
    lags = lags or DEFAULT_LAGS
    out = np.full(len(x), np.nan)
    for t in range(window, len(x), stride):
        w = x[t - window:t]
        rs = []
        for lag in lags:
            mi = mutual_info_regression(w[:-lag].reshape(-1, 1), w[lag:],
                                        n_neighbors=n_neighbors, random_state=seed)[0]
            rs.append(np.sqrt(1.0 - np.exp(-2.0 * max(mi, 0.0))))
        out[t] = np.clip(1.0 - np.mean(rs), 0.0, 1.0)
    return _forward_fill(out)


def ks_forget(x: np.ndarray, window: int = 150, stride: int = 5) -> np.ndarray:
    """Rolling two-sample KS statistic between the recent window and the window before
    it. Detects *distribution* change (level/scale), not dependence — included exactly
    to test whether "regime change" and "past irrelevant" are the same thing."""
    out = np.full(len(x), np.nan)
    for t in range(2 * window, len(x), stride):
        stat, _ = ks_2samp(x[t - window:t], x[t - 2 * window:t - window])
        out[t] = stat
    return _forward_fill(out)


SIGNALS = {
    "ewacf_mean": lambda x, seed=0: ewacf_forget(x, agg="mean"),
    "ewacf_max": lambda x, seed=0: ewacf_forget(x, agg="max"),
    "mi_knn": lambda x, seed=0: mi_forget(x, seed=seed),
    "ks_shift": lambda x, seed=0: ks_forget(x),
}
