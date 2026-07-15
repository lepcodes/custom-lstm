import numpy as np
import torch

from custom_lstm.losses.acf_losses import EWACFLoss
from custom_lstm.utils import EWACFEngine, ew_acf


def verify_alignment():
    print(">>> Starting EW-ACF Alignment Verification (Multi-Lag Architecture)")

    # 1. Generate random test data
    n_points = 500
    lags = [1, 5, 10]
    lambda_ = 0.8  # Higher lambda to test longer EMA memory
    data_tensor = torch.randn(n_points, 1)
    data_np = data_tensor.numpy().flatten()

    # 2. Compute via utils.ew_acf (Legacy comparison for each lag)
    # We compare the first lag specifically to ensure no drift in core math
    utils_acf_lag1 = ew_acf(data_np, lag=lags[0], lambda_=lambda_, last_only=False)

    # 3. Compute via new Multi-Lag Engine
    acf_engine = EWACFEngine(lambda_=lambda_, lags=lags)
    loss_fn = EWACFLoss(alpha=1.0, threshold=0.0, aggregation_strategy="average")

    # Format data for loss forward: [Batch, Seq, Features]
    sequence = data_tensor.unsqueeze(0)
    dummy_pred = torch.randn(1, n_points, 1)
    dummy_target = torch.randn(1, n_points, 1)

    # Randomize gates (simulating sigmoid [0, 1])
    dummy_gates = torch.rand(1, n_points, 1)
    gates_np = dummy_gates.numpy().flatten()

    # In the new architecture:
    # ACF Engine returns (Batch, Seq, Feat, Lags)
    autocorrelation = acf_engine(sequence)

    # Loss aggregates and calculates penalty
    _, _, penalty_val = loss_fn(dummy_pred, dummy_target, dummy_gates, autocorrelation)

    # 4. Manual calculation for verification:
    # We check the internal engine output for the first lag against legacy utils.ew_acf
    engine_acf_lag1 = autocorrelation[0, :, 0, 0].numpy()

    # The engine returns a sequence starting from max(lags)
    max_lag = max(lags)
    # Align legacy util output to match engine's valid window
    # utils_acf_lag1 starts from lags[0]
    # engine_acf_lag1 starts from max_lag
    offset = max_lag - lags[0]
    aligned_utils_acf = utils_acf_lag1[offset:]

    diff_acf = np.abs(engine_acf_lag1 - aligned_utils_acf).max()
    print(f"Max difference in raw ACF (Lag {lags[0]}): {diff_acf:.8e}")

    # 5. Verify Aggregated Penalty (Average)
    # Manual Average aggregation
    # autocorrelation is [1, valid_seq, 1, 3]
    abs_acf_all = np.abs(autocorrelation.numpy()[0, :, 0, :])
    avg_acf = np.mean(abs_acf_all, axis=-1)

    # Slice gates to match valid_seq
    active_gates_np = gates_np[max_lag:]
    expected_penalty = np.mean((1 - avg_acf) * active_gates_np)

    print(f"Expected Penalty (Weighted Average): {expected_penalty:.8f}")
    print(f"Loss Actual Penalty:                {penalty_val.item():.8f}")

    diff_penalty = abs(expected_penalty - penalty_val.item())
    print(f"Absolute Penalty Difference:        {diff_penalty:.8e}")

    if diff_acf < 1e-7 and diff_penalty < 1e-7:
        print("\nSUCCESS: Multi-Lag implementation logic is identical to expectations!")
    else:
        print("\nFAILURE: Multi-Lag implementation drifts.")


if __name__ == "__main__":
    verify_alignment()
