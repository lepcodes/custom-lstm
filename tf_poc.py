import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tensorflow import keras


# 1. DATA PREPARATION (TBPTT Slicer)
def load_and_slice_data(csv_path, window_size, bptt_steps, train_ratio=0.8):
    df = pd.read_csv(csv_path).dropna().reset_index(drop=True)  
    raw_data = df.iloc[:, 0].values.reshape(-1, 1)

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(raw_data).flatten()

    split_idx = int(len(scaled_data) * train_ratio)
    train_data = scaled_data[:split_idx]
    val_data = scaled_data[split_idx:]

    def create_blocks(data, M, B):
        # M: window_size (Features)
        # B: bptt_steps (Sequence Length)

        # Sliding window for features
        windows = np.lib.stride_tricks.sliding_window_view(data, M)
        X = windows[:-1]  # Inputs
        Y = data[M:].reshape(-1, 1)  # Targets

        # Slice into contiguous blocks of B steps
        num_blocks = len(X) // B
        X = X[: num_blocks * B].reshape(num_blocks, B, M)
        Y = Y[: num_blocks * B].reshape(num_blocks, B, 1)

        return X, Y

    X_train, Y_train = create_blocks(train_data, window_size, bptt_steps)
    X_val, Y_val = create_blocks(val_data, window_size, bptt_steps)

    return X_train, Y_train, X_val, Y_val


# 2. MODEL FACTORY
def build_stateful_model(window_size, bptt_steps, hidden_size=64):
    model = keras.Sequential(
        [
            keras.layers.Input(batch_shape=(1, bptt_steps, window_size)),
            keras.layers.LSTM(hidden_size, stateful=True, return_sequences=True),
            keras.layers.Dense(1),
        ]
    )
    model.compile(optimizer="adam", loss="mse")
    return model


# 3. EXPERIMENT RUNNER
def run_experiment(name, window_size, bptt_steps, epochs=50):
    print(f"\n>>> Running Experiment: {name} (Window Size M={window_size})")

    csv_path = "data/preprocessed/lbl_tcp_3.csv"
    X_tr, Y_tr, X_val, Y_val = load_and_slice_data(csv_path, window_size, bptt_steps)

    model = build_stateful_model(window_size, bptt_steps)

    history = {"loss": [], "val_loss": []}

    for epoch in range(epochs):
        # Train on sequence blocks
        h = model.fit(X_tr, Y_tr, batch_size=1, epochs=1, shuffle=False, verbose=0)
        # Validate on sequence blocks
        v = model.evaluate(X_val, Y_val, batch_size=1, verbose=0)

        history["loss"].append(h.history["loss"][0])
        history["val_loss"].append(v)

        # Reset state after full sequence pass
        for layer in model.layers:
            if hasattr(layer, "reset_state"):
                layer.reset_state()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/{epochs} - loss: {h.history['loss'][0]:.6f} - val_loss: {v:.6f}")

    return history


if __name__ == "__main__":
    bptt = 50
    epochs = 60

    # Experiment A: Pure (M=1)
    hist_pure = run_experiment("PURE", window_size=1, bptt_steps=bptt, epochs=epochs)

    # Experiment B: Windowed (M=30)
    hist_win = run_experiment("WINDOWED", window_size=30, bptt_steps=bptt, epochs=epochs)

    # Plotting
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(hist_pure["loss"], label="Train MSE")
    plt.plot(hist_pure["val_loss"], label="Val MSE")
    plt.title("Pure Model (M=1, Stateful)")
    plt.xlabel("Epochs")
    plt.ylabel("MSE")
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(hist_win["loss"], label="Train MSE")
    plt.plot(hist_win["val_loss"], label="Val MSE")
    plt.title("Windowed Model (M=30, Stateful)")
    plt.xlabel("Epochs")
    plt.ylabel("MSE")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("tf_poc_results.png")
    print("\nResults saved to tf_poc_results.png")
