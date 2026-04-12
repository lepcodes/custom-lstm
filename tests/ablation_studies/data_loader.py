from dataclasses import dataclass
from typing import Tuple
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from tests.ablation_studies.config import DataMode


@dataclass
class TensorPair:
    """Holds the input and target tensors for a specific configuration."""
    X: torch.Tensor
    Y: torch.Tensor


@dataclass
class ModeTensors:
    """Holds the data for both pure and windowed variants for a specific split."""
    pure: TensorPair
    win: TensorPair

    def get_by_mode(self, mode: DataMode) -> TensorPair:
        if mode == DataMode.PURE:
            return self.pure
        elif mode == DataMode.WINDOWED:
            return self.win
        raise ValueError(f"Unknown mode: {mode}")


@dataclass
class DataSplits:
    """Holds the train, validation, and test splits."""
    train: ModeTensors
    val: ModeTensors
    test: ModeTensors


def create_decoupled_tbptt_tensors(data_array: np.ndarray, window_size: int) -> ModeTensors:
    """
    Generates the exact [1, Sequence, Features] tensors required for manual TBPTT.
    Respects the distinct timelines of the Pure LSTM vs. Windowed MLP/LSTM architectures.
    """
    # ---------------------------------------------------------
    # 1. PURE STATEFUL TENSORS (Sequence Length = N - 1)
    # ---------------------------------------------------------
    # The pure LSTM predicts the very next step immediately.
    # Input: x_0 to x_{N-2}
    # Target: x_1 to x_{N-1}
    X_pure = data_array[:-1].reshape(-1, 1)
    Y_pure = data_array[1:].reshape(-1, 1)

    X_pure_tensor = torch.tensor(X_pure, dtype=torch.float32).unsqueeze(0)
    Y_pure_tensor = torch.tensor(Y_pure, dtype=torch.float32).unsqueeze(0)

    # ---------------------------------------------------------
    # 2. WINDOWED TENSORS (Sequence Length = N - window_size)
    # ---------------------------------------------------------
    # The windowed model must consume 'window_size' steps before its first prediction.
    # Fast sliding window using numpy memory strides
    windows = np.lib.stride_tricks.sliding_window_view(data_array, window_size)

    # Drop the last window because it has no future target to predict against
    X_win = windows[:-1]

    # Targets start strictly at index 'window_size'
    Y_win = data_array[window_size:].reshape(-1, 1)

    X_win_tensor = torch.tensor(X_win, dtype=torch.float32).unsqueeze(0)
    Y_win_tensor = torch.tensor(Y_win, dtype=torch.float32).unsqueeze(0)

    return ModeTensors(
        pure=TensorPair(X=X_pure_tensor, Y=Y_pure_tensor),
        win=TensorPair(X=X_win_tensor, Y=Y_win_tensor)
    )


def load_data(
    csv_path: str,
    window_size: int = 4,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15
) -> Tuple[DataSplits, StandardScaler]:
    """
    End-to-end ingestion, strict chronological splitting, and tensor generation.
    """
    # 1. Load and clean
    df = pd.read_csv(csv_path).dropna().reset_index(drop=True)
    raw_data = df.iloc[:, 0].values  # Assuming univariate target is in column 0

    # 2. Calculate explicit chronological indices
    N = len(raw_data)
    train_end = int(N * train_ratio)
    val_end = train_end + int(N * val_ratio)

    # 3. Split strictly by time (No random shuffling allowed for stateful RNNs)
    train_raw = raw_data[:train_end]
    val_raw = raw_data[train_end:val_end]
    test_raw = raw_data[val_end:]

    # 4. Strict Scaling (Fit ONLY on train to prevent data leakage)
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_raw.reshape(-1, 1)).flatten()
    val_scaled = scaler.transform(val_raw.reshape(-1, 1)).flatten()
    test_scaled = scaler.transform(test_raw.reshape(-1, 1)).flatten()

    # 5. Generate the decoupled TBPTT Tensors
    train_tensors = create_decoupled_tbptt_tensors(train_scaled, window_size)
    val_tensors = create_decoupled_tbptt_tensors(val_scaled, window_size)
    test_tensors = create_decoupled_tbptt_tensors(test_scaled, window_size)

    # 6. Pack into a clean dataclass
    splits = DataSplits(train=train_tensors, val=val_tensors, test=test_tensors)

    return splits, scaler


if __name__ == "__main__":
    # Replace with your actual dataset path
    data_dict, scaler = load_data("../../data/preprocessed/exchange_rate.csv", window_size=10)

    print("--- Pure LSTM Shapes ---")
    print(f"X_pure: {data_dict.train.pure.X.shape}")
    print(f"Y_pure: {data_dict.train.pure.Y.shape}")

    print("\n--- Windowed Model Shapes ---")
    print(f"X_win: {data_dict.train.win.X.shape}")
    print(f"Y_win: {data_dict.train.win.Y.shape}")

    print("\n--- Test Shapes ---")
    print(f"X_pure: {data_dict.test.pure.X.shape}")
    print(f"Y_pure: {data_dict.test.pure.Y.shape}")
    print(f"X_win: {data_dict.test.win.X.shape}")
    print(f"Y_win: {data_dict.test.win.Y.shape}")
