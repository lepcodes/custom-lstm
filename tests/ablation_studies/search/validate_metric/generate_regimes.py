"""
Synthetic labeled regime-switching series — the "ruler" for forget-signal validation.
====================================================================================

Every scenario returns a series plus per-timestep ground truth so any candidate
forget signal (EW-ACF, mutual information, distribution shift, ...) can be scored
objectively instead of eyeballed. Two ground-truth channels, because "forget" is
two different questions (see CLAUDE.md §4 / handoff):

  forget_label  — 1 while remembering the past would NOT improve the best possible
                  guess of the next value (conditional-mean relevance). For linear
                  processes this coincides with "dependence is gone"; the nonlinear
                  scenarios are exactly where the two definitions split (`arch_switch`:
                  dependence present, forecast value absent). Sustained, not impulsive.
  change_points — indices where the generating process *switches*. This is what
                  distribution-shift metrics (KS) should detect. Impulsive.

Design note: in the dependence-switch scenarios (`ar_noise_switch`, `phi_switch`,
`periodic_noise_switch`) each regime is scaled to unit *marginal* variance, so the
marginal distribution is (nearly) identical across regimes and only the temporal
dependence changes. A distribution-shift metric is *expected to fail* there — that
failure is informative, it separates "dependence" from "regime change" empirically.
Conversely `mean_shift` / `var_shift` keep the dependence constant, so dependence
metrics should stay flat while shift metrics fire.
"""

from dataclasses import dataclass, field

import numpy as np


@dataclass
class RegimeSeries:
    """A synthetic series with ground truth for scoring forget signals."""

    name: str
    x: np.ndarray              # (T,) the series
    forget_label: np.ndarray   # (T,) 1 = past irrelevant (dependence gone), 0 = past useful
    change_points: np.ndarray  # indices where the generating process switches
    description: str = ""
    regime_id: np.ndarray = field(default=None)  # (T,) which regime generated each step


def _segments(T: int, n_segments: int, rng: np.random.Generator, min_frac: float = 0.6):
    """Split [0, T) into n_segments of randomized lengths (each >= min_frac * T/n)."""
    base = T // n_segments
    lengths = rng.integers(int(base * min_frac), int(base * (2 - min_frac)), size=n_segments)
    lengths = np.round(lengths * T / lengths.sum()).astype(int)
    lengths[-1] = T - lengths[:-1].sum()
    bounds = np.concatenate([[0], np.cumsum(lengths)])
    return [(bounds[i], bounds[i + 1]) for i in range(n_segments)]


def _ar1(T: int, phi: float, rng: np.random.Generator, x0: float = 0.0) -> np.ndarray:
    """AR(1) with unit-variance innovations, scaled to unit *marginal* variance."""
    eps = rng.standard_normal(T)
    x = np.empty(T)
    prev = x0
    for t in range(T):
        prev = phi * prev + eps[t]
        x[t] = prev
    # marginal std of a stationary AR(1) is 1/sqrt(1 - phi^2); normalize it away
    return x * np.sqrt(1.0 - phi**2)


def ar_noise_switch(T: int = 4000, n_segments: int = 8, phi: float = 0.95,
                    seed: int = 0) -> RegimeSeries:
    """AR(1) phi=0.95 <-> white noise, both unit marginal variance (the canonical case)."""
    rng = np.random.default_rng(seed)
    x = np.empty(T)
    label = np.empty(T, dtype=int)
    regime = np.empty(T, dtype=int)
    segs = _segments(T, n_segments, rng)
    cps = []
    for i, (a, b) in enumerate(segs):
        if i % 2 == 0:  # persistent regime
            x[a:b] = _ar1(b - a, phi, rng)
            label[a:b], regime[a:b] = 0, 0
        else:           # noise regime: past irrelevant
            x[a:b] = rng.standard_normal(b - a)
            label[a:b], regime[a:b] = 1, 1
        if a > 0:
            cps.append(a)
    return RegimeSeries("ar_noise_switch", x, label, np.array(cps),
                        f"AR(1) phi={phi} <-> white noise, unit marginal variance", regime)


def phi_switch(T: int = 4000, n_segments: int = 8, phi_hi: float = 0.95,
               phi_lo: float = 0.3, seed: int = 0) -> RegimeSeries:
    """Graded version: strong vs weak (but nonzero) dependence. Harder than ar_noise_switch."""
    rng = np.random.default_rng(seed)
    x = np.empty(T)
    label = np.empty(T, dtype=int)
    regime = np.empty(T, dtype=int)
    segs = _segments(T, n_segments, rng)
    cps = []
    for i, (a, b) in enumerate(segs):
        phi = phi_hi if i % 2 == 0 else phi_lo
        x[a:b] = _ar1(b - a, phi, rng)
        label[a:b] = 0 if i % 2 == 0 else 1
        regime[a:b] = i % 2
        if a > 0:
            cps.append(a)
    return RegimeSeries("phi_switch", x, label, np.array(cps),
                        f"AR(1) phi={phi_hi} <-> phi={phi_lo}, unit marginal variance", regime)


def mean_shift(T: int = 4000, n_segments: int = 8, phi: float = 0.9,
               jump: float = 3.0, seed: int = 0) -> RegimeSeries:
    """AR(1) with constant dependence but jumping level. Dependence metrics should stay
    flat (forget_label is all-zero); only change-point detectors should fire."""
    rng = np.random.default_rng(seed)
    x = np.empty(T)
    segs = _segments(T, n_segments, rng)
    cps = []
    regime = np.empty(T, dtype=int)
    for i, (a, b) in enumerate(segs):
        mu = jump * (i % 2)
        x[a:b] = _ar1(b - a, phi, rng) + mu
        regime[a:b] = i % 2
        if a > 0:
            cps.append(a)
    return RegimeSeries("mean_shift", x, np.zeros(T, dtype=int), np.array(cps),
                        f"AR(1) phi={phi}, mean jumps by {jump}; dependence never breaks", regime)


def var_shift(T: int = 4000, n_segments: int = 8, phi: float = 0.9,
              scale: float = 3.0, seed: int = 0) -> RegimeSeries:
    """AR(1) with constant dependence but jumping scale. Same contract as mean_shift."""
    rng = np.random.default_rng(seed)
    x = np.empty(T)
    segs = _segments(T, n_segments, rng)
    cps = []
    regime = np.empty(T, dtype=int)
    for i, (a, b) in enumerate(segs):
        s = scale if i % 2 else 1.0
        x[a:b] = _ar1(b - a, phi, rng) * s
        regime[a:b] = i % 2
        if a > 0:
            cps.append(a)
    return RegimeSeries("var_shift", x, np.zeros(T, dtype=int), np.array(cps),
                        f"AR(1) phi={phi}, std jumps x{scale}; dependence never breaks", regime)


def periodic_noise_switch(T: int = 4000, n_segments: int = 8, period: int = 24,
                          snr: float = 4.0, seed: int = 0) -> RegimeSeries:
    """Sine+noise <-> pure noise, unit marginal variance. Mimics tidal series like
    aqua_alta losing/gaining periodic structure."""
    rng = np.random.default_rng(seed)
    x = np.empty(T)
    label = np.empty(T, dtype=int)
    regime = np.empty(T, dtype=int)
    segs = _segments(T, n_segments, rng)
    cps = []
    # amplitude/noise so that total marginal variance = 1 with the given SNR
    sig_var = snr / (1 + snr)
    amp = np.sqrt(2 * sig_var)
    noise_std = np.sqrt(1 - sig_var)
    for i, (a, b) in enumerate(segs):
        if i % 2 == 0:
            t = np.arange(a, b)
            x[a:b] = amp * np.sin(2 * np.pi * t / period) + noise_std * rng.standard_normal(b - a)
            label[a:b], regime[a:b] = 0, 0
        else:
            x[a:b] = rng.standard_normal(b - a)
            label[a:b], regime[a:b] = 1, 1
        if a > 0:
            cps.append(a)
    return RegimeSeries("periodic_noise_switch", x, label, np.array(cps),
                        f"sine(period={period})+noise <-> pure noise, unit variance", regime)


def _seasonal_ar(T: int, phi: float, s: int, rng: np.random.Generator) -> np.ndarray:
    """Seasonal AR: x_t = phi * x_{t-s} + eps. ACF lives only at multiples of s.
    Scaled to unit marginal variance (s interleaved AR(1) chains, each var 1/(1-phi^2))."""
    eps = rng.standard_normal(T)
    x = np.empty(T)
    for t in range(T):
        x[t] = (phi * x[t - s] if t >= s else 0.0) + eps[t]
    return x * np.sqrt(1.0 - phi**2)


def _lag_switch(name: str, s: int, T: int, n_segments: int, phi: float, seed: int,
                description: str) -> RegimeSeries:
    """Shared builder: seasonal-AR(s) regime <-> white noise, unit marginal variance."""
    rng = np.random.default_rng(seed)
    x = np.empty(T)
    label = np.empty(T, dtype=int)
    regime = np.empty(T, dtype=int)
    segs = _segments(T, n_segments, rng)
    cps = []
    for i, (a, b) in enumerate(segs):
        if i % 2 == 0:
            x[a:b] = _seasonal_ar(b - a, phi, s, rng)
            label[a:b], regime[a:b] = 0, 0
        else:
            x[a:b] = rng.standard_normal(b - a)
            label[a:b], regime[a:b] = 1, 1
        if a > 0:
            cps.append(a)
    return RegimeSeries(name, x, label, np.array(cps), description, regime)


def seasonal_ar_switch(T: int = 4000, n_segments: int = 8, phi: float = 0.9,
                       seed: int = 0) -> RegimeSeries:
    """Dependence ONLY at lag 8 (on the monitored grid): stresses lag aggregation.
    Prediction: max(|acf|) correctly retains; mean(|acf|) dilutes the one live lag."""
    return _lag_switch("seasonal_ar_switch", 8, T, n_segments, phi, seed,
                       f"seasonal AR lag 8 (phi={phi}) <-> white noise, unit variance")


def off_grid_lag_switch(T: int = 4000, n_segments: int = 8, phi: float = 0.9,
                        seed: int = 0) -> RegimeSeries:
    """Dependence ONLY at lag 6 — absent from the monitored grid [1,2,4,8].
    Prediction: BOTH EW-ACF aggregations fail; the lag grid is a load-bearing choice."""
    return _lag_switch("off_grid_lag_switch", 6, T, n_segments, phi, seed,
                       f"seasonal AR lag 6 (phi={phi}, off-grid) <-> white noise, unit variance")


def nonlinear_map_switch(T: int = 4000, n_segments: int = 8, seed: int = 0) -> RegimeSeries:
    """Chaotic logistic map (x -> 4x(1-x)): the next value is fully determined by the
    previous one, yet the theoretical ACF is zero at every lag — correlation is blind by
    construction. The noise regime draws i.i.d. samples from the map's own invariant
    (arcsine) distribution, so the marginal is identical and ONLY the dependence switches.
    Prediction: MI wins; EW-ACF ~ chance (the linearity limitation, made measurable)."""
    rng = np.random.default_rng(seed)
    u = np.empty(T)
    label = np.empty(T, dtype=int)
    regime = np.empty(T, dtype=int)
    segs = _segments(T, n_segments, rng)
    cps = []
    state = rng.uniform(0.1, 0.9)
    for i, (a, b) in enumerate(segs):
        if i % 2 == 0:
            for t in range(a, b):
                state = 4.0 * state * (1.0 - state)
                # tiny dynamical noise keeps float64 iterates off the map's unstable
                # fixed points without disturbing the invariant distribution
                state = np.clip(state + rng.uniform(-1e-6, 1e-6), 1e-9, 1 - 1e-9)
                u[t] = state
            label[a:b], regime[a:b] = 0, 0
        else:
            u[a:b] = np.sin(np.pi * rng.uniform(0, 1, b - a) / 2) ** 2  # i.i.d. arcsine
            label[a:b], regime[a:b] = 1, 1
        if a > 0:
            cps.append(a)
    x = (u - 0.5) / np.sqrt(0.125)  # arcsine(0,1): mean 1/2, variance 1/8
    return RegimeSeries("nonlinear_map_switch", x, label, np.array(cps),
                        "logistic map (deterministic, ACF=0) <-> i.i.d. arcsine noise", regime)


def arch_switch(T: int = 4000, n_segments: int = 8, alpha: float = 0.5,
                seed: int = 0) -> RegimeSeries:
    """ARCH(1) <-> white noise, both unit unconditional variance. Dependence exists in the
    ARCH regime (the past predicts the SIZE of the next move) but the conditional mean is 0
    everywhere, so the past never improves an MSE point forecast: forget_label = 1 at every
    timestep. Prediction: the correct signal stays flat across regimes; MI dropping during
    the ARCH regime is a false alarm under the thesis's prediction task."""
    rng = np.random.default_rng(seed)
    x = np.empty(T)
    regime = np.empty(T, dtype=int)
    segs = _segments(T, n_segments, rng)
    cps = []
    omega = 1.0 - alpha
    prev = 0.0
    for i, (a, b) in enumerate(segs):
        if i % 2 == 0:
            for t in range(a, b):
                prev = rng.standard_normal() * np.sqrt(omega + alpha * prev**2)
                x[t] = prev
            regime[a:b] = 0
        else:
            x[a:b] = rng.standard_normal(b - a)
            regime[a:b] = 1
        if a > 0:
            cps.append(a)
    return RegimeSeries("arch_switch", x, np.ones(T, dtype=int), np.array(cps),
                        f"ARCH(1) alpha={alpha} <-> white noise; conditional mean 0 everywhere", regime)


SCENARIOS = {
    "ar_noise_switch": ar_noise_switch,
    "phi_switch": phi_switch,
    "mean_shift": mean_shift,
    "var_shift": var_shift,
    "periodic_noise_switch": periodic_noise_switch,
    "seasonal_ar_switch": seasonal_ar_switch,
    "off_grid_lag_switch": off_grid_lag_switch,
    "nonlinear_map_switch": nonlinear_map_switch,
    "arch_switch": arch_switch,
}


def generate_all(T: int = 4000, seed: int = 0) -> list[RegimeSeries]:
    return [gen(T=T, seed=seed) for gen in SCENARIOS.values()]
