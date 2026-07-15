import torch
from torch import nn


class EWALoss(nn.Module):
    """
    EW-ACF Loss with precomputed autocorrelation values.
    Uses 'broadcast' strategy by default (direct penalty on forget gates).
    """

    def __init__(self, alpha=0.5, threshold=0.0):
        super(EWALoss, self).__init__()
        self.mse_loss = nn.MSELoss()
        self.alpha = alpha
        self.threshold = threshold

    def forward(self, predictions, targets, forget_gates, precomputed_ewa):
        mse_val = self.mse_loss(predictions, targets)

        irrelevance = 1 - torch.abs(precomputed_ewa)
        active_irrelevance = torch.clamp(irrelevance - self.threshold, min=0.0)

        penalty_tensor = torch.mul(active_irrelevance, forget_gates)

        penalty_val = torch.mean(penalty_tensor)
        total_loss = mse_val + (self.alpha * penalty_val)
        return total_loss, mse_val, penalty_val


class EWACFLoss(nn.Module):
    """
    EW-ACF Loss that accepts multi-lag autocorrelation values.
    Supports 'average' and 'max_pooling' aggregation strategies.
    """

    VALID_STRATEGIES = {"average", "max_pooling"}

    def __init__(self, alpha=0.5, threshold=0.1, aggregation_strategy="average"):
        super(EWACFLoss, self).__init__()
        if aggregation_strategy not in self.VALID_STRATEGIES:
            raise ValueError(f"Invalid aggregation_strategy '{aggregation_strategy}'. Must be one of {self.VALID_STRATEGIES}")

        self.mse_loss = nn.MSELoss()
        self.alpha = alpha
        self.threshold = threshold
        self.aggregation_strategy = aggregation_strategy

    def forward(self, predictions, targets, forget_gates, autocorrelation):
        """
        Calculates the EW-ACF penalty using a multi-lag autocorrelation tensor.

        Args:
            predictions: Model output
            targets: Ground truth
            forget_gates: (batch, seq, hidden) forget gate activations
            autocorrelation: (batch, seq_valid, features, num_lags) precomputed ACF signal
        """
        if autocorrelation.numel() == 0:
            mse_val = self.mse_loss(predictions, targets)
            return mse_val, mse_val, torch.zeros(1, device=predictions.device)

        # 1. Aggregate across lags: (batch, seq_valid, features, num_lags) -> (batch, seq_valid, features)
        abs_acf = torch.abs(autocorrelation)

        if self.aggregation_strategy == "average":
            aggregated_acf = torch.mean(abs_acf, dim=-1)
        elif self.aggregation_strategy == "max_pooling":
            # Max pooling behaves as an OR gate: if ANY lag is correlated, assume relevance
            aggregated_acf = torch.max(abs_acf, dim=-1).values

        # 2. Align lengths: compute penalty only where acf exists
        num_acf_steps = aggregated_acf.shape[1]
        active_forget_gates = forget_gates[:, -num_acf_steps:, :]

        # 3. Calculate irrelevance and penalty
        irrelevance = 1 - aggregated_acf
        # Average irrelevance across input features to get a per-step penalty scalar per batch/unit
        irrelevance = torch.mean(irrelevance, dim=2, keepdim=True)
        active_irrelevance = torch.clamp(irrelevance - self.threshold, min=0.0)

        penalty_tensor = torch.mul(active_irrelevance, active_forget_gates)

        penalty_val = torch.mean(penalty_tensor)
        mse_val = self.mse_loss(predictions, targets)
        total_loss = mse_val + (self.alpha * penalty_val)
        return total_loss, mse_val, penalty_val
