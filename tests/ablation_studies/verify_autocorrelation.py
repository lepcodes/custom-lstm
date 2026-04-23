import numpy as np
import torch

from custom_lstm.losses.acf_losses import EWACFLoss
from custom_lstm.utils import ew_acf


def verify_alignment():
    print(">>> Starting EW-ACF Alignment Verification")

    # 1. Generate random test data
    n_points = 500
    lag = 10
    lambda_ = 0.8  # Higher lambda to test longer EMA memory
    data_tensor = torch.randn(n_points, 1)
    data_np = data_tensor.numpy().flatten()

    # 2. Compute via utils.ew_acf (Trimmed)
    utils_acf = ew_acf(data_np, lag=lag, lambda_=lambda_, last_only=False)

    # 3. Compute via EWACFLoss (Stateful/Trimmed)
    # We pass dummy inputs for predictions/targets since we only care about penalty logic
    loss_fn = EWACFLoss(lambda_=lambda_, lag=lag, alpha=1.0, threshold=0.0)

    # Format data for loss forward: [Batch, Seq, Features]
    sequence = data_tensor.unsqueeze(0)
    dummy_pred = torch.randn(1, n_points, 1)
    dummy_target = torch.randn(1, n_points, 1)

    # Randomize gates (simulating sigmoid [0, 1])
    dummy_gates = torch.rand(1, n_points, 1)
    gates_np = dummy_gates.numpy().flatten()

    # In the synchronized forward, the loss handles slicing internally.
    _, _, penalty_val = loss_fn(dummy_pred, dummy_target, sequence, dummy_gates)

    # Manual calculation from utils:
    # 1. We must slice the gates to match the autocorrelation sequence (the last N-lag steps)
    active_gates_np = gates_np[lag:]

    # 2. Compute weighted penalty: mean( (1 - abs(acf)) * gates )
    expected_penalty = np.mean((1 - np.abs(utils_acf)) * active_gates_np)

    print(f"Utils ACF Length: {len(utils_acf)}")
    print(f"Utils Expected Penalty (Weighted): {expected_penalty:.8f}")
    print(f"Loss Actual Penalty:              {penalty_val.item():.8f}")

    diff = abs(expected_penalty - penalty_val.item())
    print(f"Absolute Difference:   {diff:.8e}")

    if diff < 1e-7:
        print("\nSUCCESS: Implementation logic is identical!")
    else:
        print("\nFAILURE: Implementations still drift.")


if __name__ == "__main__":
    verify_alignment()
