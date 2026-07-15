# LSTM con estado vs. sin estado — y qué hace realmente una "ventana más grande"

## La respuesta en tres líneas

1. "Con estado" ("stateful") y "sin estado" ("stateless") se diferencian en **una sola cosa**: si el
   estado oculto/de celda sobrevive entre pasadas hacia adelante (*forward passes*). Todo lo demás
   (las ecuaciones de la LSTM) es idéntico.
2. "Ventana más grande → MSE más bajo" es cierto **solo cuando la ventana es la única memoria del
   modelo**. Es cierto para la LSTM sin estado de Diego. **No** es una ley para la LSTM con estado de
   la tesis, donde la memoria proviene del estado arrastrado, no de la ventana.
3. El modelo de Diego **no tiene estado** (por defecto en Keras, `stateful=False`). Así que no es una
   plantilla para juzgar un modelo con estado — es un ejemplo del *otro* paradigma.

---

## Los tres modelos que comparamos

Los tres usan la **misma celda LSTM vanilla** (`custom_lstm/models/lstm_vanilla.py:43-54`) y pérdida
MSE. Se diferencian solo en cómo se maneja el estado y cómo se alimenta la ventana.

| | **A — Con estado (tesis)** | **B — Sin memoria** | **C — LSTM sin estado (Diego)** |
|---|---|---|---|
| Clase | `LSTMVanillaStateful` | `LSTMVanillaStateful(sever_recurrence=True)` | `LSTMVanilla` |
| Estado **entre pasos temporales** | se arrastra | **se pone a cero en cada paso** | se arrastra |
| Estado **entre ventanas** | se arrastra a lo largo de toda la serie | n/a | **se pone a cero entre ventanas** |
| Recurrencia | sí, sobre toda la serie | **ninguna** | sí, **dentro de cada ventana** |
| La ventana se alimenta como | características (lags apilados) | características (lags apilados) | **tiempo** (secuencia desenrollada) |
| De dónde viene la memoria | el estado de celda arrastrado | solo la ventana actual | el propio desenrollado de la ventana |
| La longitud de ventana controla | nada relacionado con la memoria | cuántos lags ve | **la profundidad de la recurrencia** |
| Tendencia esperada ventana → MSE | **plana** | baja, luego se estanca | **baja** |

**Versión simple:**
- **A** ya recuerda el pasado en su estado, así que apilar más valores pasados en la entrada (una
  ventana más grande) es redundante — la curva debería mantenerse plana.
- **B** no tiene memoria en absoluto; cada predicción ve solo la ventana actual, así que más lags
  ayudan un poco, y luego se saturan. Es una red feed-forward sobre la ventana, no una LSTM en ningún
  sentido real.
- **C** convierte la ventana en tiempo y desenrolla la LSTM sobre ella, reiniciando entre ventanas.
  Una ventana más grande = un desenrollado más largo = más contexto → la curva debería bajar. Este es
  el clásico "LSTM con ventana" y el diseño de Diego.

---

## El único mecanismo que lo explica todo

La celda LSTM calcula:

```
c_t = f_t * c_{t-1} + i_t * g_t
h_t = o_t * tanh(c_t)
```

La única pregunta que separa a los tres modelos es **qué son `c_{t-1}`, `h_{t-1}`** en cada paso.

- **Con estado** los mantiene del paso anterior y del chunk anterior — **desconecta** el gradiente
  (mantiene los valores, corta el gradiente) en lugar de ponerlos a cero:
  ```python
  # lstm_vanilla_stateful.py:28-34
  if self.h_t is None: self.h_t = zeros(...); self.c_t = zeros(...)
  else:                self.h_t = self.h_t.detach(); self.c_t = self.c_t.detach()
  ```
- **Sin memoria** los pone a cero en **cada** paso:
  ```python
  # lstm_vanilla_stateful.py:42-44   (sever_recurrence=True)
  self.h_t = torch.zeros_like(self.h_t)
  self.c_t = torch.zeros_like(self.c_t)
  ```
- **Sin estado (Diego)** los pone a cero una vez por llamada al forward, y luego los arrastra
  **dentro** de la ventana:
  ```python
  # lstm_vanilla.py:72-79
  h_t = zeros(...); c_t = zeros(...)          # reinicio por ventana
  for t in range(seq_length):                 # desenrollado sobre la ventana
      h_t, c_t, ... = self.lstm(x_t, (h_t, c_t))
  output = self.linear(h_t)                    # una predicción por ventana
  ```

Entonces la profundidad de la recurrencia la determina **la longitud de secuencia sobre la que el
modelo se desenrolla** — toda la serie para A, un solo paso para B, y **la ventana** para C. La regla
de la directora apunta a C.

---

## El modelo de Diego no tiene estado — evidencia

- `dylstm.py:228-230` construye `keras.layers.LSTM(...)` **sin `stateful=True`** → Keras reinicia el
  estado después de cada batch (comportamiento por defecto).
- `dylstm.py:130-133` (`ventanas`) alimenta ventanas independientes de tamaño `obs=10` (`:1037`); el
  bucle solo llama a `model.rnn.predict(x_test)` por ventana (`:1078`). No se arrastra ningún estado.
- La contribución de Diego es la **topología dinámica** (poda/crecimiento basado en la dimensión de
  Cao, `cao()` `:62`, `crecimiento()`/`poda()`), lo cual no tiene nada que ver con tener o no estado.
- El repo ya tiene una referencia correcta de Keras **con estado** para contrastar, `tf_poc.py:43-52`
  (`stateful=True`, `batch_shape` fijo, `reset_state()` manual).

---

## Qué significan "pure" y "windowed" en este repo (para que los términos no confundan la reunión)

En este harness **ambos tienen estado** — todo modelo registrado es una clase `*Stateful`
(`tests/ablation_studies/model_setup.py:18-23`). "pure" vs "win" solo cambia el **formato de la
entrada**:

- `pure`: `input_size = 1`, tensor `[1, N-1, 1]`.
- `win`: `input_size = window`, tensor `[1, N-window, window]` (la ventana apilada como características).
  (`sweep.py:29`, `data_loader.py:39-75`)

Entonces el modelo "windowed" del harness es **A** (con estado, ventana-como-características). **No**
es el modelo de Diego. El modelo de Diego (**C**) necesita un formato que el harness no produce
(`[num_windows, window, 1]`, ventana-como-tiempo) y se entrena de forma distinta — ver más abajo.

---

## Dimensiones de entrada de cada enfoque

Ambos usan el tensor estándar 3-D de RNN `(batch, secuencia, características)`. La diferencia crucial
es **en qué eje vive la ventana** — el eje de características (A) o el eje temporal (C).

| | **A — Con estado (windowed)** | **A — Con estado (pure)** | **C — Sin estado (Diego)** |
|---|---|---|---|
| Forma de entrada `(batch, seq, feat)` | `(1, N−W, W)` | `(1, N−1, 1)` | `(num_windows, W, 1)` |
| **batch** | 1 (toda la serie) | 1 (toda la serie) | `num_windows ≈ N−W`, en mini-batches de `B` |
| **seq_len** (eje temporal) | `N−W` (toda la serie) | `N−1` (toda la serie) | `W` (solo la ventana) |
| **características = `input_size`** | `W` (ventana como características) | 1 | 1 |
| **La ventana vive en…** | el eje de **características** | — | el eje **temporal** |
| Forma de salida | `(1, N−W, 1)` — cada paso | `(1, N−1, 1)` — cada paso | `(num_windows, 1)` — una por ventana |
| Mapeo | muchos-a-muchos | muchos-a-muchos | muchos-a-uno |
| Se alimenta como | chunks TBPTT de `bptt_steps`, estado **arrastrado** | igual | mini-batches mezclados, estado **reiniciado** por ventana |

`N` = longitud de la serie (train/val), `W` = tamaño de ventana, `B` = tamaño de mini-batch.

**El modelo Keras de Diego mapea exactamente a la columna C:**

```python
LSTM(9, return_sequences=True, input_shape=(10,1))  # entrada (batch, 10, 1): W=10, features=1
LSTM(5)                                             # (batch, 10, 9) -> (batch, 5)
Dense(1)                                            # (batch, 5) -> (batch, 1)
```

`input_shape=(10,1)` = `(seq_len=W=10, features=1)` — ventana en el eje **temporal**, `input_size=1`.

**Por qué esto importa (la consecuencia en parámetros):**
- **A**: `input_size = W`, así que las matrices de peso de entrada de la LSTM son `(W, H)` — **crecen
  con la ventana**. Ventana más grande = más parámetros sobre la misma señal de longitud fija ⇒ por
  esto A empeora con ventanas grandes.
- **C**: `input_size = 1` **siempre**. La ventana fija la *longitud de secuencia*, no la cantidad de
  parámetros — los pesos de la celda se mantienen en `(1, H)` sin importar `W` ⇒ por esto C es robusto
  ante el tamaño de ventana.

**Ejemplo concreto** (train de aqua_alta, `N ≈ 9374`, `W = 10`, `B = 512`):
`A windowed → (1, 9364, 10)` · `A pure → (1, 9373, 1)` · `C → (9364, 10, 1)`, alimentado en ~19
mini-batches de `(512, 10, 1)`.

---

## Diagramas de arquitectura (en `resources/`)

El repo ya tiene diagramas para ambos paradigmas, y son **precisos y consistentes con el código** —
incluyendo uno sin estado, así que aquí no falta nada:

| Paradigma | Nivel de red | Nivel de celda | Qué muestra correctamente |
|---|---|---|---|
| **A — Con estado** | `LSTMStateful.png` | `LSTMCellStatefulWin.png`, `LSTMCellStatefulPure.png` | Estado arrastrado entre batches (flecha LSTM→LSTM), estados inicializados **una sola vez**, truncamiento TBPTT entre chunks. En el diagrama de celda *Win* la ventana se apila como **características** (`input_size = window`) y el estado fluye a través de las posiciones deslizantes. |
| **C — Sin estado (Diego)** | `LSTMStateless.png` | `LSTMCellStateless.png` | Estados **reinicializados en cada batch** (sin flecha LSTM→LSTM), ventana desenrollada como **tiempo** (`input_size = 1`), **una** predicción por ventana, e índices de ventana **mezclados / no contiguos** entre batches (muestreo i.i.d.). |

También existen variantes personalizadas (con gate MLP): `LSTMCellStatefulCustomPure/Win.png`,
`LSTMCellStatelessCustom.png`.

**El motivo probable de que tú y tu co-directora estén desalineados es una palabra —
"ventana"— que se ubica en un *eje distinto* en cada diagrama:**

- En **A** (`LSTMCellStatefulWin`) la ventana es un **vector de características**
  (`input_size = window_size`); la memoria viene de la recurrencia, que abarca toda la serie. La
  ventana es información de lag redundante.
- En **C** (`LSTMCellStateless`) la ventana es el **eje temporal** sobre el que la celda se desenrolla
  (`input_size = 1`); la memoria *es* la ventana, y se reinicia entre ventanas.

Así que "hacer la ventana más grande" significa dos cosas distintas: en A añade características de
entrada (y, según los experimentos, generalmente **perjudica** — la dimensión de entrada se dispara);
en C alarga la recurrencia (y generalmente **ayuda**). Acordar esa única distinción es lo que los
pondría de nuevo en sintonía.

**Corrección opcional:** anotar cada diagrama con su eje — "ventana = características" en el diagrama
stateful-Win, "ventana = tiempo (profundidad de recurrencia)" en el diagrama sin estado — y hacer una
diapositiva comparativa de `LSTMCellStatefulWin.png` junto a `LSTMCellStateless.png`. (Puedo construir
ese compuesto.)

**Dos figuras acompañan los resultados a continuación:**
- `window_study.png` — los tres modelos (A con estado, B sin memoria, C sin estado/Diego).
- `window_study_stateful_vs_stateless.png` — **solo A vs C** (se elimina B sin memoria), la
  comparación directa con-estado vs. sin-estado.

---

## Experimento

Misma malla de ventanas en los tres modelos, cinco datasets, solo MSE, parada temprana según MSE de
validación.

| Configuración | Valor |
|---|---|
| Malla de ventanas | 1, 2, 4, 8, 16, 32, 64, 128 |
| Datasets | `aqua_alta` (mareas), `lbl_tcp_3` (tráfico a ráfagas), `mackey_glass` (fuertemente autocorrelacionado), `ethernet_traffic` (tráfico a ráfagas), `total_sunspots` (cíclico) |
| Tamaño oculto | 64 |
| Épocas / paciencia | 200 / 20 |
| LR / optimizador | 0.001 / Adam |
| Dispositivo | CPU |

**El régimen de entrenamiento difiere por paradigma — a propósito:**
- A y B tienen **estado**, así que los datos se mantienen en orden temporal estricto, **sin mezclar**
  (`data_loader.py:96`), entrenados con TBPTT sobre toda la serie.
- C no tiene estado, así que las ventanas se mezclan en mini-batches i.i.d. en cada época
  (`run_diego_baseline.py`), exactamente como hace el `model.fit(shuffle=True)` de Diego. Mezclar un
  modelo con estado lo rompería; no mezclar uno sin estado lo perjudicaría.

**Consecuencia:** el MSE absoluto **no** es directamente comparable entre A/B y C (regímenes
distintos). Hay que leer la **forma** de cada curva respecto al tamaño de ventana, no la diferencia de
altura entre paradigmas.

**Nota sobre el dispositivo:** se usa CPU porque con batch=1 los modelos con estado avanzan un paso
temporal a la vez, así que los lanzamientos de kernel de GPU por paso son más lentos que en CPU.
(`train.py:63`; C también está fijado a CPU.)

**Reproducir:**
```bash
conda activate lstm-env
# A + B (con estado, ventana-como-características)
python -m tests.ablation_studies.search.validate_window.run_window_study
# C (LSTM sin estado, ventana-como-tiempo, Diego)
python -m tests.ablation_studies.search.validate_window.run_diego_baseline
# Graficar los tres
python -m tests.ablation_studies.search.validate_window.plot_window_study
```

---

## Resultados

Los tres modelos (A con estado, B sin memoria, C sin estado/Diego):

![Tamaño de ventana vs. MSE de validación — los tres modelos](window_study.png)

Con estado (A) vs. sin estado (C) únicamente — se elimina la línea base sin memoria:

![Tamaño de ventana vs. MSE de validación — A con estado vs. C sin estado](window_study_stateful_vs_stateless.png)

Mejor MSE de validación por ventana (menor es mejor). Números crudos en `window_study_results.csv`.

**Mejor ventana y tendencia por modelo (mejor MSE de validación):**

| Dataset | Modelo | MSE @ ws=1 | mejor MSE (ventana) | MSE @ ws=128 | Tendencia al crecer la ventana |
|---|---|---|---|---|---|
| aqua_alta | A con estado | **0.0009** | 0.0009 (ws1) | 0.0429 | plana, luego **empeora bruscamente** |
| aqua_alta | B sin memoria | 0.0055 | 0.0012 (ws4) | 0.1907 | baja, luego **explota** |
| aqua_alta | C sin estado (Diego) | 0.0056 | 0.0010 (ws64) | 0.0011 | baja, luego **plana / robusta** |
| lbl_tcp_3 | A con estado | **0.2702** | 0.2702 (ws1) | 0.3587 | **empeora** con la ventana |
| lbl_tcp_3 | B sin memoria | 0.2935 | 0.2764 (ws8) | 0.3415 | plana, luego peor |
| lbl_tcp_3 | C sin estado (Diego) | 0.2920 | 0.2619 (ws32) | 0.2893 | plana / leve caída, robusta |
| mackey_glass | A con estado | 0.0061 | 0.0003 (ws128) | **0.0003** | **mejora** con la ventana |
| mackey_glass | B sin memoria | 0.0211 | 0.0001 (ws32) | 0.0002 | mejora marcadamente |
| mackey_glass | C sin estado (Diego) | 0.0215 | 0.0003 (ws32) | 0.0005 | mejora |
| ethernet_traffic | A con estado | 0.4672 | 0.3344 (ws128) | **0.3344** | **mejora** con la ventana |
| ethernet_traffic | B sin memoria | 0.5771 | 0.3038 (ws128) | 0.3038 | mejora |
| ethernet_traffic | C sin estado (Diego) | 0.5784 | 0.3255 (ws128) | 0.3255 | mejora |
| total_sunspots | A con estado | **0.0008** | 0.0008 (ws1) | 0.0647 | plana, luego **empeora bruscamente** |
| total_sunspots | B sin memoria | 0.0098 | 0.0012 (ws2) | 0.0667 | baja, luego **explota** |
| total_sunspots | C sin estado (Diego) | 0.0083 | 0.0040 (ws4) | 0.0086 | baja, luego **plana / robusta** |

**Lo que dicen las curvas:**

- **aqua_alta (mareas, suave):** el modelo con estado **A es mejor en ventana=1** y *empeora* a medida
  que crece la ventana (0.0009 → 0.043). El B sin memoria explota (→ 0.19). Solo el modelo
  ventana-como-tiempo **C se mantiene plano**. Aquí una ventana más grande es inútil o perjudicial.
- **lbl_tcp_3 (a ráfagas, difícil):** todos los modelos rondan 0.26–0.36. Una ventana más grande
  **perjudica a A y B**; C es plano y el más bajo. Ningún modelo se beneficia de una ventana más
  grande.
- **mackey_glass (fuertemente autocorrelacionado):** **los tres mejoran** al crecer la ventana. Aquí
  "ventana más grande → MSE más bajo" claramente se cumple — para todos los paradigmas.
- **ethernet_traffic (a ráfagas, difícil):** error alto (~0.30–0.58), pero **los tres mejoran** con la
  ventana — otro caso donde más contexto ayuda a todos. C y B terminan más bajos.
- **total_sunspots (cíclico):** misma forma que aqua_alta — **A es mejor en ventana=1** (0.0008) y
  se degrada bruscamente con ventanas grandes (→ 0.065); B explota (→ 0.094 en ws64); **C se mantiene
  plano/robusto**.

**Dos hechos que se repiten en todos los datasets:**

1. **La ventana como *características* (A, B) se degrada con ventanas grandes** en las series donde
   la ventana no ayuda. `input_size` crece con la ventana (hasta 128), así que la capa de entrada se
   dispara y sobreajusta — de forma dramática para B en aqua_alta (0.19) y total_sunspots (0.094). La
   ventana como *tiempo* (C, `input_size = 1`) **nunca explota** en ninguno de los cinco datasets. Si
   se van a usar ventanas, alimentarlas como tiempo es el diseño más seguro.
2. **El modelo con estado no necesita ventana.** Es mejor en **ventana=1** en aqua_alta, lbl_tcp_3 y
   total_sunspots (**3 de 5**) — su recurrencia ya captura la estructura temporal. Una ventana más
   grande solo lo ayuda en mackey_glass y ethernet_traffic (**2 de 5**), donde *todos* los modelos se
   benefician de todos modos.

**Salvedad (según lo señalado en el diseño):** A/B se entrenan de forma ordenada (TBPTT), C se mezcla
(i.i.d.), así que las alturas absolutas no son estrictamente comparables entre A/B y C. Hay que leer
la **forma** de cada curva; las afirmaciones anteriores son tendencias dentro de cada modelo, que son
independientes del régimen.

---

## Conclusión

Había tres preguntas sobre la mesa. Los datos responden a cada una.

**1. ¿Es "débil" la arquitectura con estado?**
No. En **tres de cinco** datasets alcanza su **mejor** puntaje en ventana=1 — es decir, extrae la
estructura temporal de su propio estado recurrente, sin ventana alimentada a mano. Eso es lo opuesto
a débil: es *independiente de la ventana*. Un modelo que necesita una ventana grande para funcionar se
está apoyando en la entrada, no en la memoria aprendida. El modelo con estado no está fallando en usar
la ventana; simplemente no la necesita.

**2. ¿Es "ventana más grande → MSE de validación más bajo" una ley?**
No — depende del dataset y del mecanismo.
- Se **cumple** en `mackey_glass` y `ethernet_traffic`, donde más contexto ayuda a todos los modelos.
- **Falla** en `aqua_alta`, `lbl_tcp_3` y `total_sunspots`, donde una ventana más grande es neutra o
  activamente perjudicial y el modelo con estado es mejor en ventana=1.
- Alimentada como *características* (el enfoque exacto propuesto — la ventana apilada por paso
  temporal), una ventana más grande **degrada** el desempeño en tamaños grandes en esas tres series,
  porque la dimensión de entrada se dispara. La curva monótona de "más grande es mejor" aparece en
  **2 de 5** datasets — una propiedad de la serie, no una regla.

**3. ¿Qué demuestra la tesis de Diego?**
El modelo de Diego **no tiene estado** (sección sobre `dylstm.py`). Su curva (C) es la más robusta
ante el tamaño de ventana y es competitiva en los cinco datasets — es un **paradigma válido y
distinto**, no una vara de medir que muestre que el diseño con estado es inferior. Señalar a Diego
para llamar débil al modelo con estado compara dos tipos de modelo diferentes. Si acaso, la robustez
de C confirma la premisa de la tesis de que las LSTM con y sin estado se comportan de manera distinta.

**Dónde tiene razón la directora.**
- Alimentar una ventana es una idea legítima, y ella aprobó el diagrama — para un modelo *sin estado*
  la ventana es la memoria, y en algunas series (`mackey_glass`, `ethernet_traffic`) más ventana reduce
  el MSE.
- En esas series, un modelo sin estado con ventana puede igualar o superar al modelo con estado. El
  diseño con estado tampoco es universalmente superior.

**Dónde la regla no se transfiere.**
- No es universal: en 3 de 5 series la ventana es innecesaria (con estado mejor en ws=1) o
  perjudicial.
- La receta específica de "ventana como características" escala mal (se dispara con ventanas grandes
  en esas series); si se quieren ventanas, "ventana como tiempo" (el formato de Diego,
  `input_size = 1`) es el diseño que se mantiene estable en todos los casos.

**Recomendación.**
- Reportar por dataset; no generalizar una sola curva ventana→MSE como ley.
- Para el modelo con estado, usar por defecto una **ventana pequeña (o ventana=1)**; añadir ventana
  solo donde la serie sea lo suficientemente predecible como para recompensarla (probarlo, según
  `mackey_glass`).
- Enmarcar la contribución de la tesis con precisión: el gate con estado aprende *cuándo olvidar a
  partir de su propio estado*, por eso no depende de una ventana ajustada a mano — una propiedad que
  las líneas base con ventana no tienen.
- Si se conserva una variante con ventana para comparación, usar ventana-como-tiempo para evitar el
  colapso en ventanas grandes visto en A y B.

**Una línea para la reunión:** *"En cinco datasets, una ventana más grande reduce el MSE solo en dos
de ellos; en los otros tres nuestro modelo con estado ya es el mejor en ventana=1 — su recurrencia, no
la ventana, es la que carga la memoria. El modelo de Diego no tiene estado, así que es un paradigma
distinto, no evidencia de que el nuestro sea débil."*
