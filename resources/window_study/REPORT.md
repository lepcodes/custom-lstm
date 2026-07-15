# Stateful vs Stateless LSTM — and what "bigger window" really does

## The answer in three lines

1. "Stateful" and "stateless" differ in **one thing**: whether hidden/cell state survives between
   forward passes. Everything else (the LSTM equations) is identical.
2. "Bigger window → lower MSE" is true **only when the window is the model's only memory**. It is
   true for Diego's stateless LSTM. It is **not** a law for the thesis's stateful LSTM, where memory
   comes from the carried state, not the window.
3. Diego's model is **stateless** (Keras default, `stateful=False`). So it is not a template for
   judging a stateful model — it is an example of the *other* paradigm.

---

## The three models we compare

All three use the **same vanilla LSTM cell** (`custom_lstm/models/lstm_vanilla.py:43-54`) and MSE
loss. They differ only in how state is handled and how the window is fed.

| | **A — Stateful (thesis)** | **B — Memoryless** | **C — Stateless LSTM (Diego)** |
|---|---|---|---|
| Class | `LSTMVanillaStateful` | `LSTMVanillaStateful(sever_recurrence=True)` | `LSTMVanilla` |
| State **between timesteps** | carried | **zeroed every step** | carried |
| State **between windows** | carried across whole series | n/a | **zeroed between windows** |
| Recurrence | yes, over the whole series | **none at all** | yes, **inside each window** |
| Window is fed as | features (stacked lags) | features (stacked lags) | **time** (unrolled sequence) |
| Where memory comes from | the carried cell state | only the current window | the window's own unroll |
| Window length controls | nothing about memory | how many lags it sees | **the recurrence depth** |
| Expected window → MSE | **flat** | down, then plateau | **down** |

**Plain version:**
- **A** already remembers the past in its state, so stacking more past values into the input (a
  bigger window) is redundant — the curve should stay flat.
- **B** has no memory at all; each prediction sees only the current window, so more lags help a bit,
  then saturate. It is a feed-forward net over the window, not an LSTM in any real sense.
- **C** turns the window into time and unrolls the LSTM over it, resetting between windows. A bigger
  window = a longer unroll = more context → the curve should drop. This is the classic "windowed
  LSTM" and Diego's design.

---

## The one mechanism that explains everything

The LSTM cell computes:

```
c_t = f_t * c_{t-1} + i_t * g_t
h_t = o_t * tanh(c_t)
```

The only question that separates the three models is **what `c_{t-1}`, `h_{t-1}` are** at each step.

- **Stateful** keeps them from the previous step and previous chunk — it **detaches** (keeps the
  values, cuts the gradient) instead of zeroing:
  ```python
  # lstm_vanilla_stateful.py:28-34
  if self.h_t is None: self.h_t = zeros(...); self.c_t = zeros(...)
  else:                self.h_t = self.h_t.detach(); self.c_t = self.c_t.detach()
  ```
- **Memoryless** zeroes them on **every** step:
  ```python
  # lstm_vanilla_stateful.py:42-44   (sever_recurrence=True)
  self.h_t = torch.zeros_like(self.h_t)
  self.c_t = torch.zeros_like(self.c_t)
  ```
- **Stateless (Diego)** zeroes them once per forward call, then carries them **within** the window:
  ```python
  # lstm_vanilla.py:72-79
  h_t = zeros(...); c_t = zeros(...)          # reset per window
  for t in range(seq_length):                 # unroll over the window
      h_t, c_t, ... = self.lstm(x_t, (h_t, c_t))
  output = self.linear(h_t)                    # one prediction per window
  ```

So the recurrence depth is set by **the sequence length the model unrolls over** — the whole series
for A, one step for B, and **the window** for C. The director's rule targets C.

---

## Diego's model is stateless — evidence

- `dylstm.py:228-230` builds `keras.layers.LSTM(...)` with **no `stateful=True`** → Keras resets
  state after every batch (default).
- `dylstm.py:130-133` (`ventanas`) feeds independent windows of size `obs=10` (`:1037`); the loop
  just calls `model.rnn.predict(x_test)` per window (`:1078`). No state is carried.
- Diego's contribution is **dynamic topology** (Cao-dimension pruning/growing, `cao()` `:62`,
  `crecimiento()`/`poda()`), which has nothing to do with statefulness.
- The repo already has a correct **stateful** Keras reference for contrast, `tf_poc.py:43-52`
  (`stateful=True`, fixed `batch_shape`, manual `reset_state()`).

---

## What "pure" vs "windowed" means in this repo (so the terms don't confuse the meeting)

In this harness **both are stateful** — every registered model is a `*Stateful` class
(`tests/ablation_studies/model_setup.py:18-23`). "pure" vs "win" only changes the **input layout**:

- `pure`: `input_size = 1`, tensor `[1, N-1, 1]`.
- `win`: `input_size = window`, tensor `[1, N-window, window]` (window stacked as features).
  (`sweep.py:29`, `data_loader.py:39-75`)

So the harness's "windowed" model is **A** (stateful, window-as-features). It is **not** Diego's
model. Diego's model (**C**) needs a layout the harness does not produce (`[num_windows, window, 1]`,
window-as-time) and is trained differently — see below.

---

## Input dimensions of each approach

Both use the standard 3-D RNN tensor `(batch, sequence, features)`. The crucial difference is
**which axis the window lives on** — the feature axis (A) or the time axis (C).

| | **A — Stateful (windowed)** | **A — Stateful (pure)** | **C — Stateless (Diego)** |
|---|---|---|---|
| Full-dataset tensor `(axis0, seq, feat)` | `(1, N−W, W)` | `(1, N−1, 1)` | `(num_windows, W, 1)` |
| **axis 0 (samples)** | **1** — the whole series is one sequence | **1** | `num_windows ≈ N−W` — each window is one sample |
| **seq_len** (time axis) | `N−W` (whole series) | `N−1` (whole series) | `W` (just the window) |
| **features = `input_size`** | `W` (window as features) | 1 | 1 |
| **Window lives on…** | the **feature** axis | — | the **time** axis |
| One training step feeds | a TBPTT chunk `(1, bptt_steps, W)` | `(1, bptt_steps, 1)` | a mini-batch `(B, W, 1)` |
| Steps per epoch | `⌈(N−W)/bptt_steps⌉` chunks (time-sliced) | `⌈(N−1)/bptt_steps⌉` | `⌈num_windows/B⌉` batches |
| Output shape | `(1, N−W, 1)` — every step | `(1, N−1, 1)` — every step | `(num_windows, 1)` — one per window |
| Mapping | many-to-many | many-to-many | many-to-one |
| State across steps | **carried** (TBPTT) | carried | **reset** per window |

`N` = length of the (train/val) series, `W` = window size, `B` = mini-batch size.

> **`num_windows` ≠ number of batches.** `num_windows` is the *sample count* (the whole dataset of
> windows); with mini-batch size `B` an epoch has `⌈num_windows/B⌉` batches, each `(B, W, 1)`.
> A is different: its axis 0 is always **1** (one long sequence), sliced along **time** into TBPTT
> chunks — not batched along axis 0.

**Diego's Keras model maps exactly to column C:**

```python
LSTM(9, return_sequences=True, input_shape=(10,1))  # input (batch, 10, 1): W=10, features=1
LSTM(5)                                             # (batch, 10, 9) -> (batch, 5)
Dense(1)                                            # (batch, 5) -> (batch, 1)
```

`input_shape=(10,1)` = `(seq_len=W=10, features=1)` — window on the **time** axis, `input_size=1`.

**Why this matters (the parameter consequence):**
- **A**: `input_size = W`, so the LSTM's input weight matrices are `(W, H)` — they **grow with the
  window**. Bigger window = more parameters on the same fixed-length signal ⇒ this is why A degrades
  at large windows.
- **C**: `input_size = 1` **always**. The window sets the *sequence length*, not the parameter
  count — the cell weights stay `(1, H)` regardless of `W` ⇒ this is why C is window-robust.

**Concrete example** (aqua_alta train, `N ≈ 9374`, `W = 10`, `B = 512`):
`A windowed → (1, 9364, 10)` · `A pure → (1, 9373, 1)` · `C → (9364, 10, 1)`, fed as ~19 mini-batches
of `(512, 10, 1)`.

---

## Architecture diagrams (in `resources/`)

The repo already has diagrams for both paradigms, and they are **accurate and consistent with the
code** — including a stateless one, so nothing is missing here:

| Paradigm | Network-level | Cell-level | What it correctly shows |
|---|---|---|---|
| **A — Stateful** | `LSTMStateful.png` | `LSTMCellStatefulWin.png`, `LSTMCellStatefulPure.png` | State carried across batches (LSTM→LSTM arrow), states initialized **once**, TBPTT truncation between chunks. In the *Win* cell diagram the window is stacked as **features** (`input_size = window`) and state flows across sliding positions. |
| **C — Stateless (Diego)** | `LSTMStateless.png` | `LSTMCellStateless.png` | States **re-initialized every batch** (no LSTM→LSTM arrow), window unrolled as **time** (`input_size = 1`), **one** prediction per window, and **shuffled / non-contiguous** window indices across batches (i.i.d. sampling). |

Custom (MLP-gate) variants exist too: `LSTMCellStatefulCustomPure/Win.png`, `LSTMCellStatelessCustom.png`.

**The likely reason you and your co-director are out of sync is one word — "window" — sitting on a
*different axis* in each diagram:**

- In **A** (`LSTMCellStatefulWin`) the window is a **feature vector** (`input_size = window_size`);
  memory comes from the recurrence, which spans the whole series. The window is redundant lag info.
- In **C** (`LSTMCellStateless`) the window is the **time axis** the cell unrolls over
  (`input_size = 1`); memory *is* the window, and it resets between windows.

So "make the window bigger" means two different things: in A it adds input features (and, per the
experiments, usually **hurts** — the input dimension blows up); in C it lengthens the recurrence
(and usually **helps**). Agreeing on that one distinction is what puts you back in sync.

**Optional fix:** annotate each diagram with its axis — "window = features" on the stateful-Win
diagram, "window = time (recurrence depth)" on the stateless diagram — and make one side-by-side
slide of `LSTMCellStatefulWin.png` next to `LSTMCellStateless.png`. (I can build that composite.)

**Two figures accompany the results below:**
- `window_study.png` — all three models (A stateful, B memoryless, C stateless/Diego).
- `window_study_stateful_vs_stateless.png` — **A vs C only** (memoryless B removed), the direct
  stateful-vs-stateless comparison.

---

## Experiment

Same window grid on all three models, five datasets, MSE only, early-stopped on validation MSE.

| Setting | Value |
|---|---|
| Window grid | 1, 2, 4, 8, 16, 32, 64, 128 |
| Datasets | `aqua_alta` (tidal), `lbl_tcp_3` (bursty traffic), `mackey_glass` (strongly autocorrelated), `ethernet_traffic` (bursty traffic), `total_sunspots` (cyclic) |
| Hidden size | 64 |
| Epochs / patience | 200 / 20 |
| LR / optimizer | 0.001 / Adam |
| Device | CPU |

**Training regime differs by paradigm — on purpose:**
- A and B are **stateful**, so data stays in strict time order, **no shuffling**
  (`data_loader.py:96`), trained with TBPTT over the whole series.
- C is **stateless**, so windows are shuffled into i.i.d. mini-batches every epoch
  (`run_diego_baseline.py`), exactly as Diego's `model.fit(shuffle=True)` does. Shuffling a stateful
  model would break it; not shuffling a stateless one would handicap it.

**Consequence:** absolute MSE is **not** directly comparable between A/B and C (different regimes).
Read the **shape** of each curve vs window size, not the height difference between paradigms.

**Device note:** CPU is used because with batch=1 the stateful models step one timestep at a time,
so per-step GPU kernel launches are slower than CPU. (`train.py:63`; C is pinned to CPU too.)

**Reproduce:**
```bash
conda activate lstm-env
# A + B (stateful, window-as-features)
python -m tests.ablation_studies.search.validate_window.run_window_study
# C (stateless LSTM, window-as-time, Diego)
python -m tests.ablation_studies.search.validate_window.run_diego_baseline
# Plot all three
python -m tests.ablation_studies.search.validate_window.plot_window_study
```

---

## Results

All three models (A stateful, B memoryless, C stateless/Diego):

![Window size vs validation MSE — all three models](window_study.png)

Stateful (A) vs stateless (C) only — memoryless baseline removed:

![Window size vs validation MSE — A stateful vs C stateless](window_study_stateful_vs_stateless.png)

Best validation MSE per window (lower is better). Raw numbers in `window_study_results.csv`.

**Best window and trend per model (best validation MSE):**

| Dataset | Model | MSE @ ws=1 | best MSE (window) | MSE @ ws=128 | Trend as window grows |
|---|---|---|---|---|---|
| aqua_alta | A stateful | **0.0009** | 0.0009 (ws1) | 0.0429 | flat, then **sharply worse** |
| aqua_alta | B memoryless | 0.0055 | 0.0012 (ws4) | 0.1907 | dip, then **explodes** |
| aqua_alta | C stateless (Diego) | 0.0056 | 0.0010 (ws64) | 0.0011 | drop, then **flat / robust** |
| lbl_tcp_3 | A stateful | **0.2702** | 0.2702 (ws1) | 0.3587 | **worsens** with window |
| lbl_tcp_3 | B memoryless | 0.2935 | 0.2764 (ws8) | 0.3415 | flat, then worse |
| lbl_tcp_3 | C stateless (Diego) | 0.2920 | 0.2619 (ws32) | 0.2893 | flat / mild dip, robust |
| mackey_glass | A stateful | 0.0061 | 0.0003 (ws128) | **0.0003** | **improves** with window |
| mackey_glass | B memoryless | 0.0211 | 0.0001 (ws32) | 0.0002 | improves sharply |
| mackey_glass | C stateless (Diego) | 0.0215 | 0.0003 (ws32) | 0.0005 | improves |
| ethernet_traffic | A stateful | 0.4672 | 0.3344 (ws128) | **0.3344** | **improves** with window |
| ethernet_traffic | B memoryless | 0.5771 | 0.3038 (ws128) | 0.3038 | improves |
| ethernet_traffic | C stateless (Diego) | 0.5784 | 0.3255 (ws128) | 0.3255 | improves |
| total_sunspots | A stateful | **0.0008** | 0.0008 (ws1) | 0.0647 | flat, then **sharply worse** |
| total_sunspots | B memoryless | 0.0098 | 0.0012 (ws2) | 0.0667 | dip, then **explodes** |
| total_sunspots | C stateless (Diego) | 0.0083 | 0.0040 (ws4) | 0.0086 | dip, then **flat / robust** |

**What the curves say:**

- **aqua_alta (smooth tidal):** the stateful model **A is best at window=1** and gets *worse* as
  the window grows (0.0009 → 0.043). The memoryless B blows up (→ 0.19). Only the window-as-time
  model **C stays flat**. So here bigger window is useless-to-harmful.
- **lbl_tcp_3 (bursty, hard):** every model sits around 0.26–0.36. Bigger window **hurts A and B**;
  C is flat and lowest. No model benefits from a larger window.
- **mackey_glass (strongly autocorrelated):** **all three improve** as the window grows. Here
  "bigger window → lower MSE" clearly holds — for every paradigm.
- **ethernet_traffic (bursty, hard):** high error (~0.30–0.58), but **all three improve** with the
  window — another case where more context helps everyone. C and B end lowest.
- **total_sunspots (cyclic):** same shape as aqua_alta — **A is best at window=1** (0.0008) and
  degrades sharply at large windows (→ 0.065); B explodes (→ 0.094 at ws64); **C stays flat/robust**.

**Two facts that cut across datasets:**

1. **The window as *features* (A, B) degrades at large windows** on the series where the window
   doesn't help. `input_size` grows with the window (up to 128), so the input layer explodes and
   overfits — dramatic for B on aqua_alta (0.19) and total_sunspots (0.094). The window as *time*
   (C, `input_size = 1`) **never blows up** on any of the five datasets. If windows are used at all,
   feeding them as time is the safer design.
2. **The stateful model does not need a window.** It is best at **window=1** on aqua_alta,
   lbl_tcp_3 and total_sunspots (**3 of 5**) — its recurrence already captures the temporal
   structure. A larger window helps it only on mackey_glass and ethernet_traffic (**2 of 5**), where
   *every* model is helped anyway.

**Caveat (as noted in the design):** A/B are trained ordered (TBPTT), C shuffled (i.i.d.), so
absolute heights are not strictly comparable between A/B and C. Read each curve's **shape**; the
statements above are within-model trends, which are regime-independent.

---

## Judgment

Three questions were on the table. The data answers each.

**1. Is the stateful architecture "weak"?**
No. On **three of five** datasets it reaches its **best** score at window=1 — meaning it extracts the
temporal structure from its own recurrent state, with no hand-fed window. That is the opposite of
weak: it is *window-independent*. A model that needs a large window to work is leaning on the input,
not on learned memory. The stateful model is not failing to use the window; it doesn't need it.

**2. Is "bigger window → lower validation MSE" a law?**
No — it is dataset- and mechanism-dependent.
- It **holds** on `mackey_glass` and `ethernet_traffic`, where more context helps every model.
- It **fails** on `aqua_alta`, `lbl_tcp_3` and `total_sunspots`, where a bigger window is neutral or
  actively harmful and the stateful model is best at window=1.
- Fed as *features* (the exact approach proposed — window stacked per timestep), a bigger window
  **degrades** performance at large sizes on those three series, because the input dimension blows
  up. The monotone "bigger is better" curve appears on **2 of 5** datasets — a property of the
  series, not a rule.

**3. What does Diego's thesis prove?**
Diego's model is **stateless** (Section on `dylstm.py`). Its curve (C) is the most window-robust and
is competitive across all five datasets — it is a **valid, different paradigm**, not a yardstick that
shows the stateful design is inferior. Pointing at Diego to call the stateful model weak compares
two different kinds of model. If anything, C's robustness confirms the thesis premise that stateful
and stateless LSTMs behave differently.

**Where the director is right.**
- Feeding a window is a legitimate idea, and she approved the diagram — for a *stateless* model the
  window is the memory, and on some series (`mackey_glass`, `ethernet_traffic`) more of it lowers MSE.
- On such series a stateless windowed model can match or beat the stateful one. The stateful design
  is not universally superior either.

**Where the rule does not transfer.**
- It is not universal: on 3 of 5 series the window is unnecessary (stateful best at ws=1) or harmful.
- The specific "window as features" recipe scales badly (it blows up at large windows on those
  series); if windows are wanted, "window as time" (Diego's layout, `input_size = 1`) is the design
  that stays stable everywhere.

**Recommendation.**
- Report per dataset; do not generalize a single window→MSE curve into a law.
- For the stateful model, default to a **small window (or window=1)**; add window only where the
  series is predictable enough to reward it (test it, per `mackey_glass`).
- Frame the thesis contribution precisely: the stateful gate learns *when to forget from its own
  state*, which is why it does not depend on a hand-tuned window — a property the windowed baselines
  do not have.
- If a windowed variant is kept for comparison, use window-as-time to avoid the large-window
  collapse seen in A and B.

**One line for the meeting:** *"Across five datasets, a bigger window lowers MSE on only two of
them; on the other three our stateful model is already best at window=1 — its recurrence, not the
window, carries the memory. Diego's model is stateless, so it's a different paradigm, not evidence
that ours is weak."*
