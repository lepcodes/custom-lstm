# %% [markdown]
# ## Blibliotecas necesarias
# 

# %%
# Bibliotecas base
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Bibliotecas sklearn
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error

# Bibliotecas Tensorflow
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.framework import type_spec
from tensorflow.python.util import _pywrap_utils
from tensorflow.python.util import nest
from tensorflow.python.framework import composite_tensor

# Bibliotecas Keras
from keras.models import Sequential, load_model
from keras.layers import Activation, Dropout, Dense
from keras.layers import LSTM
from keras.callbacks import EarlyStopping, ModelCheckpoint

# %%
!pip install tensorflow_model_optimization

# %%
# Biblioteca para pruning
import tensorflow_model_optimization as tfmot

# Otras bibliotecas
import tempfile

# %% [markdown]
# Redefinit Magnitud del pruner
# 

# %%
# parche version original
prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude

# %% [markdown]
# # Funcionalidades
# 

# %% [markdown]
# Función de cálculo del orden usando Cao
# 

# %%
# Calcular complejidad del Dataset

"FUNCION PARA EL CALCULO DEL ORDEN POR CAO"


def cao(x):
    maxd = 6
    tau = 1
    a = np.zeros(maxd)
    e1 = np.zeros(maxd - 1)
    ndp = np.size(x)

    for d in range(maxd):
        a[d] = 0
        ng = ndp - (d + 1) * tau
        for i in range(ng):
            n0 = None
            v0 = 1.0e30
            for j in range(ng):
                if j != i:
                    v = 0
                    for k in range(d + 1):
                        v = v + (x[i + (k * tau)] - x[j + (k * tau)]) ** 2
                    v = np.sqrt(v)
                    vij = v
                    if vij < v0 and vij != 0:
                        n0 = j
                        v0 = vij
            if n0 == None and i < ng - 1:
                n0 = i + 1
            elif n0 == None and i >= ng - 1:
                n0 = i - 1
            a[d] = a[d] + (np.sqrt(v0**2 + (x[i + (d + 1) * tau] - x[n0 + (d + 1) * tau]) ** 2)) / v0
        a[d] = a[d] / ng
        if d >= 1:
            e1[d - 1] = a[d] / a[d - 1]
    dim = 0
    for k in range(len(e1) - 1):
        if e1[k] > 0.8:
            if (e1[k + 1] - e1[k]) < 0.17 or (e1[k + 1] - e1[k]) < 0:
                dim = k + 1
                break
    if dim == 0:
        dim = maxd - 1
    return dim

# %% [markdown]
# Funcion de compulado del modelo
# 

# %%
# Actualizar cada que va a cambiar la estrucura de la red
def comp(modelo):
    modelo.compile(optimizer="adam", loss="mean_squared_error", metrics="RootMeanSquaredError")

# %% [markdown]
# Métrica de desempeño sMAPE
# 

# %%
# Buena metrica sin importar el tamaño de los datos
def sMAPE(actual, forecast):
    if not all([isinstance(actual, np.ndarray), isinstance(forecast, np.ndarray)]):
        actual, forecast = np.array(actual), np.array(forecast)

    return round(np.mean(np.abs(actual - forecast) / ((np.abs(actual) + np.abs(forecast)) / 2)) * 100, 2)

# %% [markdown]
# Segmentador en ventanas
# 

# %%
# Cuantos pasos a futuro predice y cuantos al pasado toma en cuenta (muestras)
def ventanas(n_datos, train_set):
    x = np.atleast_3d([train_set[start : start + n_datos] for start in range(0, train_set.shape[0] - n_datos)])
    y = train_set[n_datos:]
    return (x, y)

# %% [markdown]
# Impresión de función de costo
# 

# %%
# historial de entrenamiento
def imprcosto(modelo, nombre):
    plt.plot(modelo.history.history["loss"], label="train")
    plt.plot(modelo.history.history["val_loss"], label="test")
    plt.title("Entrenamiento " + nombre)
    plt.xlabel("Epocas")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

# %% [markdown]
# Impresión de predicción contra test
# 

# %%
# Comparativa entre train y test
def impr(test, obs, predict, title):
    plt.plot(test[obs:], label="test")
    plt.plot(predict, color="green", label="predict")
    plt.title(title)
    plt.legend()
    plt.show()

# %%
# Modificacion Diego Timestep
class CustomModel(keras.Model):
    """Modelo personalizado

    Implementación de la función "train_step" personalizada,
    la cual aplica una mascara de entrada a los gradientes antes de
    actualizar los pesos de la red.

    Actualizando solo los elementos de la matriz de pesos que presenten
    "True" en su respectiva posicion de la mascara.

    Nota: esta función es llamada por la función "model.fit" por lo que
    fue modificado el programa "training.py" de keras para agregar el
    parametro "mascara" y poder utilizar normalmente la funcion "fit"

    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_tracker = keras.metrics.Mean(name="loss")
        self.mae_metric = keras.metrics.MeanAbsoluteError(name="mae")
        self.loss_fn = keras.losses.MeanSquaredError()
        self.mascara = None

    def train_step(self, data):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        x, y = data

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)  # Forward pass
            # Compute the loss value
            # (the loss function is configured in `compile()`)
            loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        # GRAD * MASK
        if self.mascara == None:
            newg = gradients
        else:
            newg = [a * tf.cast(b, tf.float32) for a, b in zip(gradients, self.mascara)]
            # newg=[a*b for a, b in zip(gradients,mascara)]

        # Update weights
        self.optimizer.apply_gradients(zip(newg, trainable_vars))

        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(y, y_pred)

        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}

# %% [markdown]
# Función para Crear modelos de LSTM
# 

# %%
def armado(in_shape, capas):
    inputs = keras.Input(shape=in_shape)
    for i, neuronas in enumerate(capas):
        if i == 0:
            primera = keras.layers.LSTM(units=capas[i], return_sequences=True, name="primera")(inputs)
        elif i == 1 and i != len(capas) - 1:
            segunda = keras.layers.LSTM(units=capas[i], name="segunda")(primera)
        elif i == 2 and i != len(capas) - 1:
            tercera = keras.layers.Dense(units=capas[i], name="tercera", activation="relu")(segunda)
        else:
            if len(capas) == 2:
                outputs = keras.layers.Dense(units=capas[i], name="salida")(primera)
            elif len(capas) == 3:
                outputs = keras.layers.Dense(units=capas[i], name="salida")(segunda)
            else:
                outputs = keras.layers.Dense(units=capas[i], name="salida")(tercera)
    modelo = CustomModel(inputs, outputs)
    comp(modelo)
    return modelo

# %% [markdown]
# Función mara enmascarar modelo (permite poda virtual)
# 

# %%
# Funcion para poda controlada
def enmascarar(modelo):
    mascara = []
    # Weight Matrix
    pesos = modelo.get_weights()

    for p in range(len(pesos)):
        if len(pesos[p].shape) > 1:  # No enmascarar bias
            msk = np.logical_or(pesos[p], np.zeros(pesos[p].shape))
        else:
            msk = np.logical_or(pesos[p], np.ones(pesos[p].shape))
        mascara.append(msk)
    # Return a bool mask
    return mascara

# %% [markdown]
# Función para agregar neuronas
# 

# %%
def agregar(modelo, x, y, obs, ue, caso):
    pesos_1 = modelo.layers[1].get_weights()
    pesos_2 = modelo.layers[2].get_weights()
    pesos_s = modelo.layers[3].get_weights()

    p1_n = pesos_1.copy()
    p2_n = pesos_2.copy()
    ps_n = pesos_s.copy()

    # Aumentar capa 1
    if caso == 1:
        # Expandir Kernel capa 1
        # p1_n[0]=np.append(pesos_1[0],np.random.rand(ue*4)).reshape(1,(ue+modelo.layers[1].units)*4)
        "random dentro de rangos"
        p1_n[0] = np.append(pesos_1[0], np.random.uniform(np.min(pesos_1[0]), np.max(pesos_1[0]), size=(ue * 4))).reshape(1, (ue + modelo.layers[1].units) * 4)

        # Expandir Re-Kernel capa 1
        new = []
        # new=np.array([np.append(new,np.append(p,np.random.rand(ue*4)))for p in pesos_1[1]])
        "random dentro de rangos"
        new = np.array([np.append(new, np.append(p, np.random.uniform(np.min(p), np.max(p), size=(ue * 4)))) for p in pesos_1[1]])

        # p1_n[1]=np.block([[new],[np.random.rand(ue,(ue+modelo.layers[1].units)*4)]])
        "random dentro de rangos"
        p1_n[1] = np.block([[new], [np.random.uniform(np.min(new), np.max(new), size=(ue, (ue + modelo.layers[1].units) * 4))]])

        # Expandir bias capa 1
        # p1_n[2]=np.append(pesos_1[2],np.random.rand(ue*4))
        "random dentro de rangos"
        p1_n[2] = np.append(pesos_1[2], np.random.uniform(np.min(pesos_1[2]), np.max(pesos_1[2]), size=(ue * 4)))

        # Expandir Kernel capa 2
        # p2_n[0]=np.block([[pesos_2[0]],[np.random.rand(ue,modelo.layers[2].units*4)]])
        "random dentro de rangos"
        p2_n[0] = np.block([[pesos_2[0]], [np.random.uniform(np.min(pesos_2[0]), np.max(pesos_2[0]), size=(ue, modelo.layers[2].units * 4))]])

        # Actualizar modelo
        inputs = keras.Input(shape=(obs, 1))
        primera = keras.layers.LSTM(units=(modelo.layers[1].units + ue), return_sequences=True, name="primera")(inputs)
        segunda = keras.layers.LSTM(units=modelo.layers[2].units, name="segunda")(primera)
        outputs = keras.layers.Dense(units=1, name="salida")(segunda)
        modelo = CustomModel(inputs, outputs)
        comp(modelo)

    # Aumentar capa 2
    elif caso == 2:
        # Expandir Kernel capa 2
        new = []
        # p2_n[0]=np.array([np.append(new,np.append(p,np.random.rand(ue*4)))for p in pesos_2[0]])
        "random dentro de rangos"
        p2_n[0] = np.array([np.append(new, np.append(p, np.random.uniform(np.min(p), np.max(p), size=(ue * 4)))) for p in pesos_2[0]])

        # Expandir Re-Kernel capa 2
        new = []
        # new=np.array([np.append(new,np.append(p,np.random.rand(ue*4)))for p in pesos_2[1]])
        "random dentro de rangos"
        new = np.array([np.append(new, np.append(p, np.random.uniform(np.min(p), np.max(p), size=(ue * 4)))) for p in pesos_2[1]])

        # p2_n[1]=np.block([[new],[np.random.rand(ue,(ue+modelo.layers[2].units)*4)]])
        "random dentro de rangos"
        p2_n[1] = np.block([[new], [np.random.uniform(np.min(new), np.max(new), size=(ue, (ue + modelo.layers[2].units) * 4))]])

        # Expandir bias capa 2
        # p2_n[2]=np.append(pesos_2[2],np.random.rand(ue*4))
        "random dentro de rangos"
        p2_n[2] = np.append(pesos_2[2], np.random.uniform(np.min(pesos_2[2]), np.max(pesos_2[2]), size=(ue * 4)))

        # Expandir Kernel capa salida
        # ps_n[0]=np.block([[pesos_s[0]],[np.random.rand(ue,modelo.layers[3].units)]])
        "random dentro de rangos"
        ps_n[0] = np.block([[pesos_s[0]], [np.random.uniform(np.min(pesos_s[0]), np.max(pesos_s[0]), size=(ue, modelo.layers[3].units))]])

        # Actualizar modelo
        inputs = keras.Input(shape=(obs, 1))
        primera = keras.layers.LSTM(units=modelo.layers[1].units, return_sequences=True, name="primera")(inputs)
        segunda = keras.layers.LSTM(units=(modelo.layers[2].units + ue), name="segunda")(primera)
        outputs = keras.layers.Dense(units=1, name="salida")(segunda)
        modelo = CustomModel(inputs, outputs)
        comp(modelo)

    # Transferencia de pesos originales (transferencia de "conocimiento")
    modelo.layers[1].set_weights(p1_n)
    modelo.layers[2].set_weights(p2_n)
    modelo.layers[3].set_weights(ps_n)

    comp(modelo)
    mascara = enmascarar(modelo)
    modelo.mascara = mascara
    modelo.fit(x, y, epochs=10, verbose=0)

    return modelo

# %%
# llama a mascara para poda controloada
def poda(modelo, ratio, x, y):
    # Parameters
    pruning_params = {"pruning_schedule": tfmot.sparsity.keras.ConstantSparsity(target_sparsity=ratio, begin_step=0), "block_pooling_type": "AVG"}

    # Callbacks
    callbacks = [tfmot.sparsity.keras.UpdatePruningStep()]

    # Model's Config.
    modelo_pruning = prune_low_magnitude(modelo, **pruning_params)

    # Compile
    modelo_pruning.compile(loss="mean_squared_error", optimizer="adam", metrics=["accuracy"])

    # The model is pruned during training step
    history = modelo_pruning.fit(x, y, epochs=90, verbose=0, callbacks=callbacks)
    # Ones de model is pruned, the model must be compiled again
    comp(modelo)
    mascara = enmascarar(modelo)
    # Return mask and model
    return (modelo_pruning, mascara, history)

# %%
def crecimiento(modelo, rate, x, y, obs, path):
    # Guarda el modelo anterior para comparar
    cp = ModelCheckpoint(path, save_weights_only=True, save_best_only=True)
    es = EarlyStopping(monitor="val_loss", mode="min", verbose=1, patience=15)

    vari = modelo.trainable_variables
    pesos = modelo.get_weights().copy()
    mascara = enmascarar(modelo)
    mascara = [~mascara[p] for p in range(len(mascara))]
    grads = [np.zeros(pesos[L].shape) for L in range(len(pesos))]

    aumentar = 1
    despertar = 0

    if rate > 1:
        revivir = len(pesos[1].flatten()) - np.count_nonzero(pesos[1].flatten())

        if aumentar == 1:
            ue = 1  # nuevas unidades a agregar (unidades extras)

            for u in range(ue):
                vari = modelo.trainable_variables
                pesos = modelo.get_weights().copy()
                mascara = enmascarar(modelo)
                mascara = [~mascara[p] for p in range(len(mascara))]
                grads = [np.zeros(pesos[L].shape) for L in range(len(pesos))]

                with tf.GradientTape() as tape:
                    loss = tf.keras.losses.mse(modelo(x), y)
                grads = [a + b for a, b in zip(grads, tape.gradient(loss, vari))]

                if revivir != 0:
                    grads = [(a * b) / len(x) for a, b in zip(grads, mascara)]
                else:
                    grads = [a / len(x) for a in grads]

                grad_c1 = abs((np.matrix(grads[0]).sum() / np.matrix(grads[0]).size) + (np.matrix(grads[1]).sum() / np.matrix(grads[1]).size)) / 2
                grad_c2 = abs((np.matrix(grads[3]).sum() / np.matrix(grads[3]).size) + (np.matrix(grads[4]).sum() / np.matrix(grads[4]).size)) / 2

                if grad_c1 > grad_c2:
                    modelo = agregar(modelo, x, y, obs, 1, 1)
                elif grad_c2 > grad_c1:
                    modelo = agregar(modelo, x, y, obs, 1, 2)
                "PRUEBA PRUNING"
                modelo_pr, mascara, history = poda(modelo, 0.10, x, y)

        else:
            if revivir != 0:
                despertar = 1

    if rate <= 1 or despertar == 1:
        for p in range(len(pesos)):
            if len(pesos[p].shape) > 1:
                vari = modelo.trainable_variables
                pesos = modelo.get_weights().copy()
                mascara = enmascarar(modelo)
                mascara = [~mascara[l] for l in range(len(mascara))]
                grads = [np.zeros(pesos[L].shape) for L in range(len(pesos))]

                if rate != 1:
                    revivir = int(rate * (len(pesos[p].flatten()))) - np.count_nonzero(pesos[p].flatten())
                else:
                    revivir = len(pesos[p].flatten()) - np.count_nonzero(pesos[p].flatten())

                with tf.GradientTape() as tape:
                    loss = tf.keras.losses.mse(modelo(x), y)
                grads = [a + b for a, b in zip(grads, tape.gradient(loss, vari))]

                grads = [(a * b) / len(x) for a, b in zip(grads, mascara)]
                grads = grads[p] + (~mascara[p] * np.full(pesos[p].shape, -1.0e30))

                forma = pesos[p].shape
                pesos[p] = pesos[p].flatten()
                gradiente = np.array(grads).flatten()

                for i in range(revivir):
                    pesos[p][np.argmax(gradiente)] = (-0.01) * np.amax(gradiente)
                    gradiente[np.argmax(gradiente)] = -1.0e30

                pesos[p] = pesos[p].reshape(forma)

                inicio = 0
                for c, capas in enumerate(modelo.layers):
                    if c != 0:
                        nuevo = []
                        [nuevo.append(pesos[inicio + ii]) for ii in range(len(modelo.layers[c].get_weights()))]
                        inicio += len(modelo.layers[c].get_weights())
                        modelo.layers[c].set_weights(nuevo)

                comp(modelo)
                modelo.mascara = enmascarar(modelo)
                modelo.fit(x, y, epochs=5, verbose=0)

    comp(modelo)
    modelo.mascara = enmascarar(modelo)
    history = modelo.fit(x, y, epochs=10, verbose=0, validation_split=0.1)  # ,callbacks=[cp])#,es])
    # modelo.load_weights(path)

    return (modelo, mascara, history)

# %%
cmap_data = plt.cm.Paired
cmap_cv = plt.cm.coolwarm


def plot_cv_indices(cv, X, ax, n_splits, lw=10):
    """Create a sample plot for indices of a cross-validation object."""

    # Generate the training/testing visualizations for each CV split

    for ii, (tr, tt) in enumerate(cv.split(X=X)):
        # Fill in indices with the training/test groups
        indices = np.array([np.nan] * len(X))
        indices[tt] = 1
        indices[tr] = 0

        # Visualize the results
        ax.scatter(
            range(len(indices)),
            [ii + 0.5] * len(indices),
            s=0.5,
            c=indices,
            marker="_",
            lw=lw,
            cmap=cmap_cv,
            vmin=-0.2,
            vmax=1.2,
        )

    yticklabels = list(range(n_splits))
    ax.set(
        yticks=np.arange(n_splits) + 0.5,
        yticklabels=yticklabels,
        xlabel="Sample index",
        ylabel="CV iteration",
        ylim=[n_splits, -0.2],
        xlim=[0, 1006],
    )
    plt.show()
    return ax

# %% [markdown]
# Politicas de recomposición
# 

# %%
def politic(i, train, test, pre_train, dim_reg, modelo, obs):
    tresh = 1.5e-5

    def cas_prn(pre_train, train, modelo, obs, dim_reg):
        print("Pruning")
        # es = EarlyStopping(monitor='val_loss', mode='min', verbose=1,patience=15)
        modelo.tr_reg.append(1)
        if (dim_reg[i - 1] - dim_reg[i]) > 1:
            modelo.rate = modelo.rate - 0.5
        else:
            modelo.rate = modelo.rate - 0.25
        if modelo.rate < 0.25:
            modelo.rate = 0.25

        for ind, tr_set in enumerate([pre_train, train]):
            if ind == 0:
                cp = ModelCheckpoint(modelo.path, save_weights_only=True, save_best_only=True)
                modelo.mascara = enmascarar(modelo.rnn)
                x, y = ventanas(obs, tr_set)
                modelo.history = modelo.rnn.fit(x, y, epochs=20, verbose=0, validation_split=0.1, callbacks=[cp])  # ,es])
                modelo.rnn.load_weights(modelo.path)
            else:
                cp = ModelCheckpoint(modelo.path, save_weights_only=True, save_best_only=True)
                modelo.mascara = enmascarar(modelo.rnn)
                x, y = ventanas(obs, tr_set)
                modelo_pr, modelo.mascara, modelo.history = poda(modelo.rnn, abs(modelo.rate - 1), x, y)
                modelo.mascara = enmascarar(modelo.rnn)
                modelo.history = modelo.rnn.fit(x, y, epochs=10, verbose=0, validation_split=0.1, callbacks=[cp])  # ,es])
                modelo.rnn.load_weights(modelo.path)
        return (modelo_pr, modelo.mascara, modelo.history)

    def cas_grw(pre_train, train, modelo, obs, dim_reg):
        print("Growing")
        # es = EarlyStopping(monitor='val_loss', mode='min', verbose=1,patience=15)
        modelo.tr_reg.append(2)
        modelo.rate = modelo.rate + 0.25 * (dim_reg[i] - dim_reg[i - 1])

        for ind, tr_set in enumerate([pre_train, train]):
            if ind == 0:
                cp = ModelCheckpoint(modelo.path, save_weights_only=True, save_best_only=True)
                modelo.mascara = enmascarar(modelo.rnn)
                x, y = ventanas(obs, tr_set)
                modelo.rnn, modelo.mascara, modelo.history = crecimiento(modelo.rnn, modelo.rate, x, y, obs, modelo.path)
            else:
                cp = ModelCheckpoint(modelo.path, save_weights_only=True, save_best_only=True)
                modelo.mascara = enmascarar(modelo.rnn)
                x, y = ventanas(obs, tr_set)
                modelo.history = modelo.rnn.fit(x, y, epochs=80, verbose=0, validation_split=0.1, callbacks=[cp])  # ,es])
                modelo.rnn.load_weights(modelo.path)
                # modelo.rnn, modelo.mascara, modelo.history = funciones.crecimiento(modelo.rnn,modelo.rate,x,y,obs,modelo.path)
        if modelo.rate > 1:
            modelo.rate = 1
        return (modelo.rnn, modelo.mascara, modelo.history)

    def cas_nrm(pre_train, train, modelo, obs, dim_reg):
        print("Normal")
        # cp=ModelCheckpoint(modelo.path,save_weights_only=True,save_best_only=True)
        # es = EarlyStopping(monitor='val_loss', mode='min', verbose=1,patience=25)
        modelo.tr_reg.append(0)
        x, y = ventanas(obs, train)
        modelo.mascara = enmascarar(modelo.rnn)
        modelo.history = modelo.rnn.fit(x, y, epochs=90, verbose=0, validation_split=0.1)  # ,callbacks=[cp]),#es])
        # modelo.rnn.load_weights(modelo.path)
        # modelo.rnn = load_model(modelo.path)
        return (modelo.rnn, modelo.mascara, modelo.history)

    # Entrenamiento DyLSTM
    if i == 0:
        # "NORMAL"
        print("Zero")
        modelo.mascara = None
        x, y = ventanas(obs, train)
        modelo.history = modelo.rnn.fit(x, y, epochs=90, verbose=0, validation_split=0.1)

    elif i == 1:
        # "PRUNING"
        if i > 0 and dim_reg[i] < dim_reg[i - 1] and modelo.rate > 0.25:
            modelo_pr, modelo.mascara, modelo.history = cas_prn(pre_train, train, modelo, obs, dim_reg)
        # "GROWING"
        elif i > 0 and dim_reg[i] > dim_reg[i - 1]:
            modelo.rnn, modelo.mascara, modelo.history = cas_grw(pre_train, train, modelo, obs, dim_reg)
        # "NORMAL"
        else:
            modelo.rnn, modelo.mascara, modelo.history = cas_nrm(pre_train, train, modelo, obs, dim_reg)

    else:
        # "PRUNING"
        if modelo.tr_reg[i - 1] == 1:
            # "UP"
            if modelo.serie[i - 1] > modelo.serie[i - 2] and (modelo.serie[i - 1] - modelo.serie[i - 2]) > tresh:
                if dim_reg[i] < dim_reg[i - 1]:
                    # modelo_pr, modelo.mascara, modelo.history = cas_prn(pre_train,train,modelo,obs,dim_reg)
                    modelo.rnn, modelo.mascara, modelo.history = cas_nrm(pre_train, train, modelo, obs, dim_reg)
                    modelo.flag = 1
                elif dim_reg[i] > dim_reg[i - 1]:
                    modelo.rnn, modelo.mascara, modelo.history = cas_grw(pre_train, train, modelo, obs, dim_reg)
                else:
                    modelo.rnn, modelo.mascara, modelo.history = cas_nrm(pre_train, train, modelo, obs, dim_reg)
                    # modelo.rate=modelo.rate+0.25
                    # modelo.rnn, modelo.mascara, modelo.history = cas_grw(pre_train,train,modelo,obs,dim_reg)

            # "DWN"
            elif modelo.serie[i - 1] < modelo.serie[i - 2]:
                if dim_reg[i] < dim_reg[i - 1]:
                    modelo_pr, modelo.mascara, modelo.history = cas_prn(pre_train, train, modelo, obs, dim_reg)
                elif dim_reg[i] > dim_reg[i - 1]:
                    # modelo.rnn, modelo.mascara, modelo.history = cas_grw(pre_train,train,modelo,obs,dim_reg)
                    modelo.rnn, modelo.mascara, modelo.history = cas_nrm(pre_train, train, modelo, obs, dim_reg)
                    modelo.flag = 2
                else:
                    modelo.rnn, modelo.mascara, modelo.history = cas_nrm(pre_train, train, modelo, obs, dim_reg)
            # "N"
            else:
                if dim_reg[i] < dim_reg[i - 1]:
                    modelo_pr, modelo.mascara, modelo.history = cas_prn(pre_train, train, modelo, obs, dim_reg)
                elif dim_reg[i] > dim_reg[i - 1]:
                    # modelo.rnn, modelo.mascara, modelo.history = cas_grw(pre_train,train,modelo,obs,dim_reg)
                    modelo.rnn, modelo.mascara, modelo.history = cas_nrm(pre_train, train, modelo, obs, dim_reg)
                    modelo.flag = 2
                else:
                    modelo.rnn, modelo.mascara, modelo.history = cas_nrm(pre_train, train, modelo, obs, dim_reg)

        # "GROWING"
        elif modelo.tr_reg[i - 1] == 2:
            # "UP"
            if modelo.serie[i - 1] > modelo.serie[i - 2] and (modelo.serie[i - 1] - modelo.serie[i - 2]) > tresh:
                if dim_reg[i] < dim_reg[i - 1]:
                    modelo_pr, modelo.mascara, modelo.history = cas_prn(pre_train, train, modelo, obs, dim_reg)
                elif dim_reg[i] > dim_reg[i - 1]:
                    # modelo.rnn, modelo.mascara, modelo.history = cas_grw(pre_train,train,modelo,obs,dim_reg)
                    modelo.rnn, modelo.mascara, modelo.history = cas_nrm(pre_train, train, modelo, obs, dim_reg)
                    modelo.flag = 2
                else:
                    modelo.rnn, modelo.mascara, modelo.history = cas_nrm(pre_train, train, modelo, obs, dim_reg)
            # "DWN"
            elif modelo.serie[i - 1] < modelo.serie[i - 2]:
                if dim_reg[i] < dim_reg[i - 1]:
                    modelo_pr, modelo.mascara, modelo.history = cas_prn(pre_train, train, modelo, obs, dim_reg)
                    # modelo.rnn, modelo.mascara, modelo.history = cas_nrm(pre_train,train,modelo,obs,dim_reg)
                    # modelo.flag=1
                elif dim_reg[i] > dim_reg[i - 1]:
                    modelo.rnn, modelo.mascara, modelo.history = cas_grw(pre_train, train, modelo, obs, dim_reg)
                    # modelo.rnn, modelo.mascara, modelo.history = cas_nrm(pre_train,train,modelo,obs,dim_reg)
                else:
                    modelo.rnn, modelo.mascara, modelo.history = cas_nrm(pre_train, train, modelo, obs, dim_reg)
            # "N"
            else:
                if dim_reg[i] < dim_reg[i - 1]:
                    modelo_pr, modelo.mascara, modelo.history = cas_prn(pre_train, train, modelo, obs, dim_reg)
                elif dim_reg[i] > dim_reg[i - 1]:
                    # modelo.rnn, modelo.mascara, modelo.history = cas_grw(pre_train,train,modelo,obs,dim_reg)
                    modelo.rnn, modelo.mascara, modelo.history = cas_nrm(pre_train, train, modelo, obs, dim_reg)
                    modelo.flag = 2
                else:
                    modelo.rnn, modelo.mascara, modelo.history = cas_nrm(pre_train, train, modelo, obs, dim_reg)

        # "NORMAL"
        else:
            if np.any(np.nonzero(np.array(modelo.tr_reg))) == True:
                last = np.array(modelo.tr_reg)[np.max(np.nonzero(np.array(modelo.tr_reg)))]
            else:
                last = 0
            if last == 1:
                # "PRUNING"
                # "UP"
                if modelo.serie[i - 1] > modelo.serie[i - 2] and (modelo.serie[i - 1] - modelo.serie[i - 2]) > tresh:
                    if dim_reg[i] < dim_reg[i - 1]:
                        # modelo_pr, modelo.mascara, modelo.history = cas_prn(pre_train,train,modelo,obs,dim_reg)
                        modelo.rnn, modelo.mascara, modelo.history = cas_nrm(pre_train, train, modelo, obs, dim_reg)
                        modelo.flag = 1
                    elif dim_reg[i] > dim_reg[i - 1]:
                        modelo.rnn, modelo.mascara, modelo.history = cas_grw(pre_train, train, modelo, obs, dim_reg)
                    else:
                        modelo.rnn, modelo.mascara, modelo.history = cas_nrm(pre_train, train, modelo, obs, dim_reg)
                        # modelo.rate=modelo.rate+0.25
                        # modelo.rnn, modelo.mascara, modelo.history = cas_grw(pre_train,train,modelo,obs,dim_reg)
                # "DWN"
                elif modelo.serie[i - 1] < modelo.serie[i - 2]:
                    if dim_reg[i] < dim_reg[i - 1]:
                        modelo_pr, modelo.mascara, modelo.history = cas_prn(pre_train, train, modelo, obs, dim_reg)
                    elif dim_reg[i] > dim_reg[i - 1]:
                        # modelo.rnn, modelo.mascara, modelo.history = cas_grw(pre_train,train,modelo,obs,dim_reg)
                        modelo.rnn, modelo.mascara, modelo.history = cas_nrm(pre_train, train, modelo, obs, dim_reg)
                        modelo.flag = 2
                    else:
                        modelo.rnn, modelo.mascara, modelo.history = cas_nrm(pre_train, train, modelo, obs, dim_reg)
                # "N"
                else:
                    if dim_reg[i] < dim_reg[i - 1]:
                        modelo_pr, modelo.mascara, modelo.history = cas_prn(pre_train, train, modelo, obs, dim_reg)
                    elif dim_reg[i] > dim_reg[i - 1]:
                        # modelo.rnn, modelo.mascara, modelo.history = cas_grw(pre_train,train,modelo,obs,dim_reg)
                        modelo.rnn, modelo.mascara, modelo.history = cas_nrm(pre_train, train, modelo, obs, dim_reg)
                        modelo.flag = 2
                    else:
                        modelo.rnn, modelo.mascara, modelo.history = cas_nrm(pre_train, train, modelo, obs, dim_reg)
            elif last == 2:
                # "GROWING"
                # "UP"
                if modelo.serie[i - 1] > modelo.serie[i - 2] and (modelo.serie[i - 1] - modelo.serie[i - 2]) > tresh:
                    if dim_reg[i] < dim_reg[i - 1]:
                        modelo_pr, modelo.mascara, modelo.history = cas_prn(pre_train, train, modelo, obs, dim_reg)
                    elif dim_reg[i] > dim_reg[i - 1]:
                        # modelo.rnn, modelo.mascara, modelo.history = cas_grw(pre_train,train,modelo,obs,dim_reg)
                        modelo.rnn, modelo.mascara, modelo.history = cas_nrm(pre_train, train, modelo, obs, dim_reg)
                        modelo.flag = 2
                    else:
                        modelo.rnn, modelo.mascara, modelo.history = cas_nrm(pre_train, train, modelo, obs, dim_reg)
                # "DWN"
                elif modelo.serie[i - 1] < modelo.serie[i - 2]:
                    if dim_reg[i] < dim_reg[i - 1]:
                        # modelo_pr, modelo.mascara, modelo.history = cas_prn(pre_train,train,modelo,obs,dim_reg)
                        modelo.rnn, modelo.mascara, modelo.history = cas_nrm(pre_train, train, modelo, obs, dim_reg)
                        modelo.flag = 1
                    elif dim_reg[i] > dim_reg[i - 1]:
                        modelo.rnn, modelo.mascara, modelo.history = cas_grw(pre_train, train, modelo, obs, dim_reg)
                        # modelo.rnn, modelo.mascara, modelo.history = cas_nrm(pre_train,train,modelo,obs,dim_reg)
                    else:
                        modelo.rnn, modelo.mascara, modelo.history = cas_nrm(pre_train, train, modelo, obs, dim_reg)
                # "N"
                else:
                    if dim_reg[i] < dim_reg[i - 1]:
                        modelo_pr, modelo.mascara, modelo.history = cas_prn(pre_train, train, modelo, obs, dim_reg)
                    elif dim_reg[i] > dim_reg[i - 1]:
                        # modelo.rnn, modelo.mascara, modelo.history = cas_grw(pre_train,train,modelo,obs,dim_reg)
                        modelo.rnn, modelo.mascara, modelo.history = cas_nrm(pre_train, train, modelo, obs, dim_reg)
                        modelo.flag = 2
                    else:
                        modelo.rnn, modelo.mascara, modelo.history = cas_nrm(pre_train, train, modelo, obs, dim_reg)
            else:
                # "NORMAL"
                if dim_reg[i] < dim_reg[i - 1]:
                    modelo_pr, modelo.mascara, modelo.history = cas_prn(pre_train, train, modelo, obs, dim_reg)
                elif dim_reg[i] > dim_reg[i - 1]:
                    modelo.rnn, modelo.mascara, modelo.history = cas_grw(pre_train, train, modelo, obs, dim_reg)
                else:
                    modelo.rnn, modelo.mascara, modelo.history = cas_nrm(pre_train, train, modelo, obs, dim_reg)

    return modelo

# %%
def politic(i, train, test, pre_train, dim_reg, modelo, obs):
    tresh = 1.5e-5

    def cas_prn(pre_train, train, modelo, obs, dim_reg):
        print("Pruning")
        # es = EarlyStopping(monitor='val_loss', mode='min', verbose=1,patience=15)
        modelo.tr_reg.append(1)
        if (dim_reg[i - 1] - dim_reg[i]) > 1:
            modelo.rate = modelo.rate - 0.5
        else:
            modelo.rate = modelo.rate - 0.25
        if modelo.rate < 0.25:
            modelo.rate = 0.25

        for ind, tr_set in enumerate([pre_train, train]):
            if ind == 0:
                cp = ModelCheckpoint(modelo.path, save_weights_only=True, save_best_only=True)
                modelo.rnn.mascara = enmascarar(modelo.rnn)
                x, y = ventanas(obs, tr_set)
                modelo.history = modelo.rnn.fit(x, y, epochs=20, verbose=0, validation_split=0.1)
                # modelo.rnn.load_weights(modelo.path)
            else:
                cp = ModelCheckpoint(modelo.path, save_weights_only=True, save_best_only=True)
                modelo.rnn.mascara = enmascarar(modelo.rnn)
                x, y = ventanas(obs, tr_set)
                modelo_pr, modelo.rnn.mascara, modelo.history = poda(modelo.rnn, abs(modelo.rate - 1), x, y)
                modelo.rnn.mascara = enmascarar(modelo.rnn)
                modelo.history = modelo.rnn.fit(x, y, epochs=10, verbose=0, validation_split=0.1)
                # modelo.rnn.load_weights(modelo.path)
        return (modelo_pr, modelo.rnn.mascara, modelo.history)

    def cas_grw(pre_train, train, modelo, obs, dim_reg):
        print("Growing")
        # es = EarlyStopping(monitor='val_loss', mode='min', verbose=1,patience=15)
        modelo.tr_reg.append(2)
        modelo.rate = modelo.rate + 0.25 * (dim_reg[i] - dim_reg[i - 1])

        for ind, tr_set in enumerate([pre_train, train]):
            if ind == 0:
                cp = ModelCheckpoint(modelo.path, save_weights_only=True, save_best_only=True)
                modelo.rnn.mascara = enmascarar(modelo.rnn)
                x, y = ventanas(obs, tr_set)
                modelo.rnn, modelo.mascara, modelo.history = crecimiento(modelo.rnn, modelo.rate, x, y, obs, modelo.path)
            else:
                cp = ModelCheckpoint(modelo.path, save_weights_only=True, save_best_only=True)
                modelo.rnn.mascara = enmascarar(modelo.rnn)
                x, y = ventanas(obs, tr_set)
                modelo.history = modelo.rnn.fit(x, y, epochs=80, verbose=0, validation_split=0.1)
                # modelo.rnn.load_weights(modelo.path)
                # modelo.rnn, modelo.mascara, modelo.history = funciones.crecimiento(modelo.rnn,modelo.rate,x,y,obs,modelo.path)
        if modelo.rate > 1:
            modelo.rate = 1
        return (modelo.rnn, modelo.mascara, modelo.history)

    def cas_nrm(pre_train, train, modelo, obs, dim_reg):
        print("Normal")
        # cp=ModelCheckpoint(modelo.path,save_weights_only=True,save_best_only=True)
        # es = EarlyStopping(monitor='val_loss', mode='min', verbose=1,patience=25)
        modelo.tr_reg.append(0)
        x, y = ventanas(obs, train)
        modelo.mascara = enmascarar(modelo.rnn)
        modelo.history = modelo.rnn.fit(x, y, epochs=90, verbose=0, validation_split=0.1)  # ,callbacks=[cp]),#es])
        # modelo.rnn.load_weights(modelo.path)
        # modelo.rnn = load_model(modelo.path)
        return (modelo.rnn, modelo.mascara, modelo.history)

    # Entrenamiento DyLSTM
    if i == 0:
        # "NORMAL"
        print("Zero")
        modelo.mascara = None
        x, y = ventanas(obs, train)
        modelo.history = modelo.rnn.fit(x, y, epochs=90, verbose=0, validation_split=0.1)

    elif i == 1:
        # "PRUNING"
        if i > 0 and dim_reg[i] < dim_reg[i - 1] and modelo.rate > 0.25:
            modelo_pr, modelo.mascara, modelo.history = cas_prn(pre_train, train, modelo, obs, dim_reg)
        # "GROWING"
        elif i > 0 and dim_reg[i] > dim_reg[i - 1]:
            modelo.rnn, modelo.mascara, modelo.history = cas_grw(pre_train, train, modelo, obs, dim_reg)
        # "NORMAL"
        else:
            modelo.rnn, modelo.mascara, modelo.history = cas_nrm(pre_train, train, modelo, obs, dim_reg)

    else:
        # "PRUNING" -> paso anterior fue pruning
        if modelo.tr_reg[i - 1] == 1:
            # "UP"
            if modelo.serie[i - 1] > modelo.serie[i - 2] and (modelo.serie[i - 1] - modelo.serie[i - 2]) > tresh:
                if dim_reg[i] < dim_reg[i - 1]:
                    # modelo_pr, modelo.mascara, modelo.history = cas_prn(pre_train,train,modelo,obs,dim_reg)
                    modelo.rnn, modelo.mascara, modelo.history = cas_nrm(pre_train, train, modelo, obs, dim_reg)
                    modelo.flag = 1
                elif dim_reg[i] > dim_reg[i - 1]:
                    modelo.rnn, modelo.mascara, modelo.history = cas_grw(pre_train, train, modelo, obs, dim_reg)
                else:
                    modelo.rnn, modelo.mascara, modelo.history = cas_nrm(pre_train, train, modelo, obs, dim_reg)
                    # modelo.rate=modelo.rate+0.25
                    # modelo.rnn, modelo.mascara, modelo.history = cas_grw(pre_train,train,modelo,obs,dim_reg)

            # "DWN"
            elif modelo.serie[i - 1] < modelo.serie[i - 2]:
                if dim_reg[i] < dim_reg[i - 1]:
                    modelo_pr, modelo.mascara, modelo.history = cas_prn(pre_train, train, modelo, obs, dim_reg)
                elif dim_reg[i] > dim_reg[i - 1]:
                    # modelo.rnn, modelo.mascara, modelo.history = cas_grw(pre_train,train,modelo,obs,dim_reg)
                    modelo.rnn, modelo.mascara, modelo.history = cas_nrm(pre_train, train, modelo, obs, dim_reg)
                    modelo.flag = 2
                else:
                    modelo.rnn, modelo.mascara, modelo.history = cas_nrm(pre_train, train, modelo, obs, dim_reg)
            # "N"
            else:
                if dim_reg[i] < dim_reg[i - 1]:
                    modelo_pr, modelo.mascara, modelo.history = cas_prn(pre_train, train, modelo, obs, dim_reg)
                elif dim_reg[i] > dim_reg[i - 1]:
                    # modelo.rnn, modelo.mascara, modelo.history = cas_grw(pre_train,train,modelo,obs,dim_reg)
                    modelo.rnn, modelo.mascara, modelo.history = cas_nrm(pre_train, train, modelo, obs, dim_reg)
                    modelo.flag = 2
                else:
                    modelo.rnn, modelo.mascara, modelo.history = cas_nrm(pre_train, train, modelo, obs, dim_reg)

        # "GROWING" -> paso anterior fue growing
        elif modelo.tr_reg[i - 1] == 2:
            # "UP"
            if modelo.serie[i - 1] > modelo.serie[i - 2] and (modelo.serie[i - 1] - modelo.serie[i - 2]) > tresh:
                if dim_reg[i] < dim_reg[i - 1]:
                    modelo_pr, modelo.mascara, modelo.history = cas_prn(pre_train, train, modelo, obs, dim_reg)
                elif dim_reg[i] > dim_reg[i - 1]:
                    # modelo.rnn, modelo.mascara, modelo.history = cas_grw(pre_train,train,modelo,obs,dim_reg)
                    modelo.rnn, modelo.mascara, modelo.history = cas_nrm(pre_train, train, modelo, obs, dim_reg)
                    modelo.flag = 2
                else:
                    modelo.rnn, modelo.mascara, modelo.history = cas_nrm(pre_train, train, modelo, obs, dim_reg)
            # "DWN"
            elif modelo.serie[i - 1] < modelo.serie[i - 2]:
                if dim_reg[i] < dim_reg[i - 1]:
                    modelo_pr, modelo.mascara, modelo.history = cas_prn(pre_train, train, modelo, obs, dim_reg)
                    # modelo.rnn, modelo.mascara, modelo.history = cas_nrm(pre_train,train,modelo,obs,dim_reg)
                    # modelo.flag=1
                elif dim_reg[i] > dim_reg[i - 1]:
                    modelo.rnn, modelo.mascara, modelo.history = cas_grw(pre_train, train, modelo, obs, dim_reg)
                    # modelo.rnn, modelo.mascara, modelo.history = cas_nrm(pre_train,train,modelo,obs,dim_reg)
                else:
                    modelo.rnn, modelo.mascara, modelo.history = cas_nrm(pre_train, train, modelo, obs, dim_reg)
            # "N"
            else:
                if dim_reg[i] < dim_reg[i - 1]:
                    modelo_pr, modelo.mascara, modelo.history = cas_prn(pre_train, train, modelo, obs, dim_reg)
                elif dim_reg[i] > dim_reg[i - 1]:
                    # modelo.rnn, modelo.mascara, modelo.history = cas_grw(pre_train,train,modelo,obs,dim_reg)
                    modelo.rnn, modelo.mascara, modelo.history = cas_nrm(pre_train, train, modelo, obs, dim_reg)
                    modelo.flag = 2
                else:
                    modelo.rnn, modelo.mascara, modelo.history = cas_nrm(pre_train, train, modelo, obs, dim_reg)

        # "NORMAL"
        else:
            if np.any(np.nonzero(np.array(modelo.tr_reg))) == True:
                last = np.array(modelo.tr_reg)[np.max(np.nonzero(np.array(modelo.tr_reg)))]
            else:
                last = 0
            if last == 1:
                # "PRUNING"
                # "UP"
                if modelo.serie[i - 1] > modelo.serie[i - 2] and (modelo.serie[i - 1] - modelo.serie[i - 2]) > tresh:
                    if dim_reg[i] < dim_reg[i - 1]:
                        # modelo_pr, modelo.mascara, modelo.history = cas_prn(pre_train,train,modelo,obs,dim_reg)
                        modelo.rnn, modelo.mascara, modelo.history = cas_nrm(pre_train, train, modelo, obs, dim_reg)
                        modelo.flag = 1
                    elif dim_reg[i] > dim_reg[i - 1]:
                        modelo.rnn, modelo.mascara, modelo.history = cas_grw(pre_train, train, modelo, obs, dim_reg)
                    else:
                        modelo.rnn, modelo.mascara, modelo.history = cas_nrm(pre_train, train, modelo, obs, dim_reg)
                        # modelo.rate=modelo.rate+0.25
                        # modelo.rnn, modelo.mascara, modelo.history = cas_grw(pre_train,train,modelo,obs,dim_reg)
                # "DWN"
                elif modelo.serie[i - 1] < modelo.serie[i - 2]:
                    if dim_reg[i] < dim_reg[i - 1]:
                        modelo_pr, modelo.mascara, modelo.history = cas_prn(pre_train, train, modelo, obs, dim_reg)
                    elif dim_reg[i] > dim_reg[i - 1]:
                        # modelo.rnn, modelo.mascara, modelo.history = cas_grw(pre_train,train,modelo,obs,dim_reg)
                        modelo.rnn, modelo.mascara, modelo.history = cas_nrm(pre_train, train, modelo, obs, dim_reg)
                        modelo.flag = 2
                    else:
                        modelo.rnn, modelo.mascara, modelo.history = cas_nrm(pre_train, train, modelo, obs, dim_reg)
                # "N"
                else:
                    if dim_reg[i] < dim_reg[i - 1]:
                        modelo_pr, modelo.mascara, modelo.history = cas_prn(pre_train, train, modelo, obs, dim_reg)
                    elif dim_reg[i] > dim_reg[i - 1]:
                        # modelo.rnn, modelo.mascara, modelo.history = cas_grw(pre_train,train,modelo,obs,dim_reg)
                        modelo.rnn, modelo.mascara, modelo.history = cas_nrm(pre_train, train, modelo, obs, dim_reg)
                        modelo.flag = 2
                    else:
                        modelo.rnn, modelo.mascara, modelo.history = cas_nrm(pre_train, train, modelo, obs, dim_reg)
            elif last == 2:
                # "GROWING"
                # "UP"
                if modelo.serie[i - 1] > modelo.serie[i - 2] and (modelo.serie[i - 1] - modelo.serie[i - 2]) > tresh:
                    if dim_reg[i] < dim_reg[i - 1]:
                        modelo_pr, modelo.mascara, modelo.history = cas_prn(pre_train, train, modelo, obs, dim_reg)
                    elif dim_reg[i] > dim_reg[i - 1]:
                        # modelo.rnn, modelo.mascara, modelo.history = cas_grw(pre_train,train,modelo,obs,dim_reg)
                        modelo.rnn, modelo.mascara, modelo.history = cas_nrm(pre_train, train, modelo, obs, dim_reg)
                        modelo.flag = 2
                    else:
                        modelo.rnn, modelo.mascara, modelo.history = cas_nrm(pre_train, train, modelo, obs, dim_reg)
                # "DWN"
                elif modelo.serie[i - 1] < modelo.serie[i - 2]:
                    if dim_reg[i] < dim_reg[i - 1]:
                        # modelo_pr, modelo.mascara, modelo.history = cas_prn(pre_train,train,modelo,obs,dim_reg)
                        modelo.rnn, modelo.mascara, modelo.history = cas_nrm(pre_train, train, modelo, obs, dim_reg)
                        modelo.flag = 1
                    elif dim_reg[i] > dim_reg[i - 1]:
                        modelo.rnn, modelo.mascara, modelo.history = cas_grw(pre_train, train, modelo, obs, dim_reg)
                        # modelo.rnn, modelo.mascara, modelo.history = cas_nrm(pre_train,train,modelo,obs,dim_reg)
                    else:
                        modelo.rnn, modelo.mascara, modelo.history = cas_nrm(pre_train, train, modelo, obs, dim_reg)
                # "N"
                else:
                    if dim_reg[i] < dim_reg[i - 1]:
                        modelo_pr, modelo.mascara, modelo.history = cas_prn(pre_train, train, modelo, obs, dim_reg)
                    elif dim_reg[i] > dim_reg[i - 1]:
                        # modelo.rnn, modelo.mascara, modelo.history = cas_grw(pre_train,train,modelo,obs,dim_reg)
                        modelo.rnn, modelo.mascara, modelo.history = cas_nrm(pre_train, train, modelo, obs, dim_reg)
                        modelo.flag = 2
                    else:
                        modelo.rnn, modelo.mascara, modelo.history = cas_nrm(pre_train, train, modelo, obs, dim_reg)
            else:
                # "NORMAL"
                if dim_reg[i] < dim_reg[i - 1]:
                    modelo_pr, modelo.mascara, modelo.history = cas_prn(pre_train, train, modelo, obs, dim_reg)
                elif dim_reg[i] > dim_reg[i - 1]:
                    modelo.rnn, modelo.mascara, modelo.history = cas_grw(pre_train, train, modelo, obs, dim_reg)
                else:
                    modelo.rnn, modelo.mascara, modelo.history = cas_nrm(pre_train, train, modelo, obs, dim_reg)

    return modelo

# %% [markdown]
# # Experimentación
# 

# %% [markdown]
# Crear modelo
# 

# %%
class Modelo:
    def __init__(self, rnn):
        self.mse = []
        self.smape = []
        self.cv_smape = []
        self.rnn = rnn
        self.serie = []
        self.serie_smape = []
        self.rate = 1
        self.tr_reg = [0]
        self.mascara = []
        self.path = None
        self.history = []
        self.flag = 0


obs = 10

# %%
data = pd.read_csv(r"daily-min-temperatures.csv")
data = np.asarray(data[["Temp"]])

# %%
sc = MinMaxScaler(feature_range=(0, 1))
data = sc.fit_transform(data.reshape(-1, 1))

# %%
model = armado([obs, 1], [9, 5, 1])
model = Modelo(model)
model.path = "model.weights.h5"

# %%
"PROCESS"
n_splits = 9
tscv = TimeSeriesSplit(n_splits=n_splits)
pre_train = []
dim_reg = np.zeros(n_splits + 1)
dim_reg[0] = cao(data[0:100])

for i, (train_index, test_index) in enumerate(tscv.split(data)):
    train = data[train_index]
    test = data[test_index]

    # Train--->
    if i == 0:
        # primera iteracion sin considera cao
        ventana = np.append(train, test)
    else:
        ventana = np.append(pre_train, test)

    dim_reg[i + 1] = cao(ventana)
    model = politic(i, train, test, pre_train, dim_reg, model, obs)

    # Test--->
    x_test, y = ventanas(obs, test)

    #  DyLSTM Test
    pred = model.rnn.predict(x_test)

    model.serie.append(mean_squared_error(test[obs:], pred))
    model.serie_smape.append(sMAPE(test[obs:], pred))

    pre_train = test


# "ERRORS"
model.mse = round(np.mean(model.serie), 4)
model.cv_smape = round(np.mean(model.serie_smape), 4)

# %%
print(model.mse)
print(model.cv_smape)

# %%
# 0 Normal
# 1 Prunning
# 2 Growing
Cell_MD  = 0
MLP()

for i in range(epochs):

  D = np.argmax(MLP(data))

  if D == 1:
    Cell_MD = 1
  elif D==2
    Cell_MD = 2


  if D==1:
    podar()
  elif



