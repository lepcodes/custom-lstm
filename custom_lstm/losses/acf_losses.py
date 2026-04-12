import warnings

import torch
from torch import nn


class EWALoss(nn.Module):
    """
    EW-ACF Loss with precomputed autocorrelation values.
    Supports routing the penalty via 'broadcast' or 'input_gate' strategy.
    """

    VALID_STRATEGIES = {"broadcast", "input_gate"}

    def __init__(self, alpha=0.5, threshold=0.0, routing_strategy="broadcast"):
        super(EWALoss, self).__init__()
        if routing_strategy not in self.VALID_STRATEGIES:
            raise ValueError(
                f"Invalid routing_strategy '{routing_strategy}'. "
                f"Must be one of {self.VALID_STRATEGIES}"
            )
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
                raise ValueError(
                    "routing_strategy='input_gate' requires input_gates tensor, got None."
                )
            penalty_tensor = penalty_tensor * input_gates.detach()

        penalty_val = torch.mean(penalty_tensor)
        total_loss = mse_val + (self.alpha * penalty_val)
        return total_loss, mse_val, penalty_val


class EWACFLoss(nn.Module):
    """
    EW-ACF Loss with online autocorrelation computation.
    Supports routing the penalty via 'broadcast' or 'input_gate' strategy.
    Enforces lag >= input_size for windowed inputs via enforce_min_lag().
    """

    VALID_STRATEGIES = {"broadcast", "input_gate"}

    def __init__(self, lambda_=0.5, lag=1, alpha=0.5, threshold=0.1, routing_strategy="broadcast"):
        super(EWACFLoss, self).__init__()
        if routing_strategy not in self.VALID_STRATEGIES:
            raise ValueError(
                f"Invalid routing_strategy '{routing_strategy}'. "
                f"Must be one of {self.VALID_STRATEGIES}"
            )
        self.mse_loss = nn.MSELoss()
        self.alpha = alpha
        self.lambda_ = lambda_
        self.lag = lag
        self.epsilon = 1e-8
        self.threshold = threshold
        self.routing_strategy = routing_strategy
        self.register_buffer("mean", torch.zeros(1))
        self.register_buffer("variance", torch.zeros(1) + self.epsilon)
        self.register_buffer("covariance", torch.zeros(1))
        self.register_buffer("input_lag", torch.zeros(0))

    def enforce_min_lag(self, input_size: int):
        """
        For windowed inputs, enforce lag >= input_size so the autocorrelation
        measures genuine beyond-window dependency (Strategy B).
        """
        if self.lag < input_size:
            warnings.warn(
                f"EWACFLoss: lag={self.lag} < input_size={input_size}. "
                f"Clamping lag to {input_size} to enforce non-overlapping windows."
            )
            self.lag = input_size

    def forward(self, predictions, targets, sequence, forget_gates, input_gates=None):
        if self.input_lag.numel() == 0:
            full_sequence = sequence
            initial_padding = [torch.zeros(sequence.shape[0], sequence.shape[2], device=sequence.device)] * self.lag
        else:
            full_sequence = torch.cat((self.input_lag, sequence), dim=1)
            initial_padding = []

        self.input_lag = full_sequence[:, -self.lag :, :].detach()

        autocorrelation = [] + initial_padding
        for t in range(self.lag, full_sequence.shape[1]):
            x_t = full_sequence[:, t, :]
            x_t_lag = full_sequence[:, t - self.lag, :]

            self.mean = torch.mul(self.lambda_, self.mean) + torch.mul(1 - self.lambda_, x_t)
            self.variance = torch.mul(self.lambda_, self.variance) + torch.mul(1 - self.lambda_, (x_t - self.mean) ** 2)
            self.covariance = torch.mul(self.lambda_, self.covariance) + torch.mul(1 - self.lambda_, (x_t - self.mean) * (x_t_lag - self.mean))
            autocorrelation.append(self.covariance / torch.sqrt(self.variance * self.variance + self.epsilon))

        irrelevance = 1 - torch.abs(torch.stack(autocorrelation, dim=1))
        irrelevance = torch.mean(irrelevance, dim=2, keepdim=True)
        active_irrelevance = torch.clamp(irrelevance - self.threshold, min=0.0)

        penalty_tensor = torch.mul(active_irrelevance, forget_gates)

        if self.routing_strategy == "input_gate":
            if input_gates is None:
                raise ValueError(
                    "routing_strategy='input_gate' requires input_gates tensor, got None."
                )
            penalty_tensor = penalty_tensor * input_gates.detach()

        penalty_val = torch.mean(penalty_tensor)
        mse_val = self.mse_loss(predictions, targets)
        total_loss = mse_val + self.alpha * penalty_val
        return total_loss, mse_val, penalty_val

    def reset_state(self):
        self.mean.zero_()
        self.variance.fill_(self.epsilon)
        self.covariance.zero_()
        self.input_lag = torch.tensor([], device=self.mean.device)
