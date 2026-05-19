import warnings

import torch
from torch import nn

from custom_lstm.utils import EWACFEngine


class EWALoss(nn.Module):
    """
    EW-ACF Loss with precomputed autocorrelation values.
    Supports routing the penalty via 'broadcast' or 'input_gate' strategy.
    """

    VALID_STRATEGIES = {"broadcast", "input_gate"}

    def __init__(self, alpha=0.5, threshold=0.0, routing_strategy="broadcast"):
        super(EWALoss, self).__init__()
        if routing_strategy not in self.VALID_STRATEGIES:
            raise ValueError(f"Invalid routing_strategy '{routing_strategy}'. Must be one of {self.VALID_STRATEGIES}")
        self.mse_loss = nn.MSELoss()
        self.alpha = alpha
        self.threshold = threshold
        self.routing_strategy = routing_strategy

    def forward(self, predictions, targets, forget_gates, precomputed_ewa, input_gates=None):
        mse_val = self.mse_loss(predictions, targets)

        irrelevance = 1 - torch.abs(precomputed_ewa)
        active_irrelevance = torch.clamp(irrelevance - self.threshold, min=0.0)

        penalty_tensor = torch.mul(active_irrelevance, forget_gates)

        if self.routing_strategy == "input_gate":
            if input_gates is None:
                raise ValueError("routing_strategy='input_gate' requires input_gates tensor, got None.")
            penalty_tensor = penalty_tensor * input_gates.detach()

        penalty_val = torch.mean(penalty_tensor)
        total_loss = mse_val + (self.alpha * penalty_val)
        return total_loss, mse_val, penalty_val


class EWACFLoss(nn.Module):
    """
    EW-ACF Loss with online autocorrelation computation.
    Uses the shared mathematical engine from utils.py.
    """

    VALID_STRATEGIES = {"broadcast", "input_gate"}

    def __init__(self, lambda_=0.5, lag=1, alpha=0.5, threshold=0.1, routing_strategy="broadcast"):
        super(EWACFLoss, self).__init__()
        if routing_strategy not in self.VALID_STRATEGIES:
            raise ValueError(f"Invalid routing_strategy '{routing_strategy}'. Must be one of {self.VALID_STRATEGIES}")
        self.mse_loss = nn.MSELoss()
        self.alpha = alpha
        self.epsilon = 1e-8
        self.threshold = threshold
        self.routing_strategy = routing_strategy

        self.acf_engine = EWACFEngine(lambda_=lambda_, lag=lag, epsilon=self.epsilon)

    @property
    def lag(self):
        return self.acf_engine.lag

    @lag.setter
    def lag(self, value):
        self.acf_engine.lag = value

    def enforce_min_lag(self, input_size: int):
        if self.lag < input_size:
            warnings.warn(f"EWACFLoss: lag={self.lag} < input_size={input_size}. Clamping lag to {input_size} to enforce non-overlapping windows.")
            self.lag = input_size

    def forward(self, predictions, targets, sequence, forget_gates, input_gates=None):
        autocorrelation = self.acf_engine(sequence)

        if autocorrelation.numel() == 0:
            mse_val = self.mse_loss(predictions, targets)
            return mse_val, mse_val, torch.zeros(1, device=predictions.device)

        # Align lengths: compute penalty only where acf exists
        num_acf_steps = autocorrelation.shape[1]
        active_forget_gates = forget_gates[:, -num_acf_steps:, :]

        irrelevance = 1 - torch.abs(autocorrelation)
        irrelevance = torch.mean(irrelevance, dim=2, keepdim=True)
        active_irrelevance = torch.clamp(irrelevance - self.threshold, min=0.0)

        penalty_tensor = torch.mul(active_irrelevance, active_forget_gates)

        if self.routing_strategy == "input_gate":
            if input_gates is None:
                raise ValueError("routing_strategy='input_gate' requires input_gates tensor")
            active_input_gates = input_gates[:, -num_acf_steps:, :].detach()
            penalty_tensor = penalty_tensor * active_input_gates

        penalty_val = torch.mean(penalty_tensor)
        mse_val = self.mse_loss(predictions, targets)
        total_loss = mse_val + self.alpha * penalty_val
        return total_loss, mse_val, penalty_val

    def reset_state(self):
        self.acf_engine.reset_state()
