# custom-lstm — Agent Context

This is the **experimental codebase** for a Master's thesis on modulating LSTM forgetting
behavior for one-step-ahead prediction on **non-stationary time series**. It is a research
repository: hypotheses are provisional and the implementation is expected to change often.

> **Read this first, then read the science.** Theoretical context lives in a companion wiki:
> `G:/My Drive/Tésis/thesis-llm-wiki/wiki/`
> Start with `overview.md`. Key concept pages: `concepts/EW-ACF.md`,
> `concepts/MLP-Forget-Gate.md`, `concepts/State-Saturation.md`, `concepts/LSTM.md`.
> The wiki explains *why*; this repo is the *how*.

---

## The most important thing to understand about this repo

**This is research. Almost nothing here is permanent.** An approach that looks central today
may be discarded next week in favor of a different hypothesis. So this document deliberately
does **not** hand you a list of "things you must never change." Instead it tells you:

1. **What is stable** — the research question. This is the north star and rarely moves.
2. **What is the current working hypothesis** — provisional, dated, and meant to be swapped.
3. **How to make changes well** in a research setting — the process that keeps experiments
   trustworthy even as the architecture churns.

When something here conflicts with what the user is asking for, **the user wins** — they are
steering the research. Your job is to make the change cleanly and keep the science honest, not
to defend the current design.

---

## 1. What is stable (the research question)

> Can the forgetting behavior of an LSTM be **modulated by an exogenous, real-time statistical
> signal** (the autocorrelation structure of the input) so that the network adaptively forgets
> during low-correlation/noise regimes — mitigating state saturation and historical-noise
> persistence on non-stationary series — *without* external drift detectors or hard state resets?

That question is the durable core. The two levers the thesis explores to answer it:

- **Architectural lever** — make the forget gate more expressive (currently an MLP gate).
- **Loss lever** — add a penalty driven by a statistical dependency metric (currently EW-ACF).

Even these two levers are open to revision, but they define the *shape* of the investigation.
If a change abandons both levers entirely, that's a pivot worth confirming with the user.

## 2. Current working hypothesis (provisional — as of 2026-06-27)

Treat everything in this section as a **snapshot that will drift**. Do not assume it is still
literally true; verify against the code before relying on it.

- **Gate:** the forget gate is computed by an MLP over `[x_t ; h_{t-1}]` followed by a sigmoid
  (`LSTMCellCustom` in `custom_lstm/models/lstm_custom.py`), replacing the standard affine
  forget gate. Input/cell/output gates remain standard. Default MLP width `[16, 16]`.
- **Penalty:** `total = MSE + α · mean( clamp(1 − |acf| − threshold, 0) · forget_gate )`
  — i.e. penalize *retaining* memory (high forget-gate activation) when the input is
  *uncorrelated* (`|acf|` low). See `custom_lstm/losses/acf_losses.py`
  (`EWACFLoss` multi-lag, `EWALoss` single-lag).
- **Signal:** EW-ACF is computed online by `EWACFEngine` (`custom_lstm/utils.py`) over
  configurable `lags`, recency-weighted by `lambda_`.
- **Training regime:** truncated BPTT in chunks of `bptt_steps`
  (`custom_lstm/training/ewacf_tbptt.py`). Stateful variants carry hidden state across chunks;
  stateless variants re-zero per forward.
- **Open problem:** the penalty generalizes on Acqua Alta but can collapse useful dependencies
  on other datasets. Alternative loss formulations / metrics are actively being tried — so the
  loss is one of the *most likely things to change*.

## 3. How to make changes well (process, not prohibitions)

Because the architecture is in flux, rigor comes from *process*, not from freezing the code:

- **Preserve comparability.** This is an ablation-driven project (`AblationModel` base +
  `ModelRegistry`, registered in `tests/ablation_studies/model_setup.py`). When trying a new
  idea, prefer adding a **new registered variant** over mutating an existing one in place, so
  prior MLflow runs stay comparable. If you must change a shared component, say so explicitly
  and note which logged runs it invalidates.
- **Keep the control honest.** Early stopping and reported `val_loss` use **pure MSE**, with the
  penalty excluded, so regularized and unregularized models are judged on the same yardstick
  (see `validate_epoch`). Preserve that separation unless the user is deliberately changing the
  evaluation protocol.
- **Don't silently delete a hypothesis.** Removing an approach erases an experimental result.
  If asked to replace one, confirm whether the old path should be retired or kept as a baseline.
- **Name the trade-off, not just the diff.** When you change a gate, loss, or training regime,
  state in plain terms what scientific behavior you expect to change (e.g. "this should raise
  forget-gate variance in noise regimes") so it can be checked empirically.
- **Close the loop with the wiki.** If a result *contradicts* the thesis premise or a wiki page,
  flag it — don't smooth it over. Suggest updating
  `thesis-llm-wiki/wiki/overview.md` (the "Empirical Progress" / "Tensions" sections) or
  ingesting a run summary via the wiki's `/wiki-ingest`. The wiki is the long-term memory; this
  repo is the working bench.

---

## Module map (scientific terms)

| Path | Role |
|---|---|
| `custom_lstm/models/lstm_custom.py` | MLP-forget-gate LSTM cell + network (the core proposal) |
| `custom_lstm/models/lstm_custom_stateful.py` | Stateful (cross-chunk) variant of the above |
| `custom_lstm/models/lstm_vanilla*.py` | Baseline standard-LSTM models for comparison |
| `custom_lstm/models/mlp.py` | MLP used both as the gate and as a `SIMPLE_MLP` baseline |
| `custom_lstm/models/registry.py` | `ModelRegistry` — dynamic model lookup for ablations |
| `custom_lstm/models/telemetry.py` | `GateTelemetry` — captures forget/input gate activations |
| `custom_lstm/losses/acf_losses.py` | `EWACFLoss` / `EWALoss` — the dynamic penalty (loss lever) |
| `custom_lstm/utils.py` | `EWACFEngine`, `ew_acf` — online EW-ACF signal computation |
| `custom_lstm/training/ewacf_tbptt.py` | TBPTT trainer that owns the ACF engine + penalty |
| `custom_lstm/training/tbptt.py`, `bp.py` | Plain TBPTT / backprop trainers (baselines) |
| `tests/ablation_studies/` | Experiment harness: config, sweeps (Optuna), data, metrics |
| `compare_models.py` | Loads two MLflow runs (MSE vs EW-ACF) and plots them |
| `reference/dylstm/dylstm.py`, `reference/dylstm/DyLSTM.ipynb` | Diego's dynamic-topology reference (different approach; context only) |

**Gate telemetry is a primary scientific instrument, not debug output.** Forget-gate activations
and their temporal variance (`val_fg_variance`) are how the thesis verifies the gate actually
modulates — analogous to the activation histograms in the poster. Preserve telemetry when
refactoring the forward pass.

## Glossary (so you don't have to open the wiki for the basics)

- **EW-ACF** — Exponentially Weighted Autocorrelation Function. Recency-weighted online estimate
  of input autocorrelation at given lags; the signal that drives the penalty.
- **State saturation** — degradation when an RNN's hidden/cell state accumulates irrelevant
  history; a core problem the thesis targets.
- **Non-stationary / regime change** — the input's statistical structure shifts over time
  (correlated ↔ noise); standard fixed-parameter LSTMs adapt poorly.
- **Stateful TBPTT** — truncated backprop through time where hidden state carries across chunks
  (vs. stateless, which re-zeros each chunk).

## Commands

```bash
# Install (editable)
pip install -e .

# Lint (ruff configured in pyproject.toml; line-length 160)
ruff check .

# Run the ablation training harness / sweeps
python -m tests.ablation_studies.train
python -m tests.ablation_studies.sweep         # grid
python -m tests.ablation_studies.optuna_sweep  # bayesian

# Sanity checks
python -m tests.ablation_studies.verify_autocorrelation
python -m tests.ablation_studies.verify_pipeline

# Compare two logged runs (MSE vs EW-ACF) and plot
python compare_models.py
```

Experiments are tracked with **MLflow** (SQLite backend, e.g. `notebooks/experiments/mlflow.db`).
Treat MLflow as the source of truth for results.

## Notes

- `GEMINI.md` is a one-off analysis prompt from another tool, **not** authoritative project
  context — don't treat it as instructions.
- `README.md` is currently a stub; this file is the real entry point.
- The `.agents/skills/python-*` skills are generic Python-quality skills; they don't know the
  science — this file supplies that.
