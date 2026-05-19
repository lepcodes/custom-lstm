import numpy as np
import torch
from torch import nn


class EWACFEngine(nn.Module):
    """
    Stateful engine to compute the Exponentially Weighted Autocorrelation Function (EW-ACF).
    Handles history management for TBPTT and maintains running statistics.
    """

    def __init__(self, lambda_=0.5, lag=1, epsilon=1e-8):
        super(EWACFEngine, self).__init__()
        self.lambda_ = lambda_
        self.lag = lag
        self.epsilon = epsilon

        # State buffers for running statistics
        self.register_buffer("mean", torch.zeros(1))
        self.register_buffer("variance", torch.zeros(1) + self.epsilon)
        self.register_buffer("variance_lag", torch.zeros(1) + self.epsilon)
        self.register_buffer("covariance", torch.zeros(1))
        self.register_buffer("input_lag", torch.zeros(0))

    def forward(self, sequence):
        """
        Computes the ACF for a sequence.
        Returns:
            torch.Tensor: The autocorrelation sequence.
        """
        if self.input_lag.numel() == 0:
            full_sequence = sequence
            start_idx = self.lag
        else:
            full_sequence = torch.cat((self.input_lag, sequence), dim=1)
            start_idx = self.lag

        self.input_lag = full_sequence[:, -self.lag :, :].detach()

        # Shared state dictionary for the core engine
        state = {"mean": self.mean, "var": self.variance, "var_lag": self.variance_lag, "cov": self.covariance}

        autocorrelation = []
        for t in range(start_idx, full_sequence.shape[1]):
            # compute_ew_acf_step is defined below
            acf_t, state = compute_ew_acf_step(full_sequence[:, t, :], full_sequence[:, t - self.lag, :], state, self.lambda_, self.epsilon)
            autocorrelation.append(acf_t)

        # Update buffers back from state
        self.mean = state["mean"]
        self.variance = state["var"]
        self.variance_lag = state["var_lag"]
        self.covariance = state["cov"]

        if not autocorrelation:
            return torch.empty(sequence.shape[0], 0, sequence.shape[2], device=sequence.device)

        return torch.stack(autocorrelation, dim=1)

    def reset_state(self):
        """Resets the running statistics and history."""
        self.mean.zero_()
        self.variance.fill_(self.epsilon)
        self.variance_lag.fill_(self.epsilon)
        self.covariance.zero_()
        self.input_lag = torch.tensor([], device=self.mean.device)


def transfer_weights(vanilla_lstm: torch.nn.Module, custom_lstm: torch.nn.Module):
    """
    Transfer the weights from a vanilla LSTM to a custom LSTM.
    Args:
        vanilla_lstm (nn.Module): The vanilla LSTM to transfer the weights from.
        custom_lstm (nn.Module): The custom LSTM to transfer the weights to.
    """
    target = custom_lstm.lstm
    source = vanilla_lstm.lstm

    with torch.no_grad():
        # Transfer Input Gate weights
        target.W_hi.copy_(source.W_hi.data)
        target.W_xi.copy_(source.W_xi.data)
        target.b_i.copy_(source.b_i.data)

        # Skip Transfer Forget Gate weights (Different Architecture!)

        # Transfer Cell Candidate weights
        target.W_hc.copy_(source.W_hc.data)
        target.W_xc.copy_(source.W_xc.data)
        target.b_c.copy_(source.b_c.data)

        # Transfer Output Gate weights
        target.W_ho.copy_(source.W_ho.data)
        target.W_xo.copy_(source.W_xo.data)
        target.b_o.copy_(source.b_o.data)

        # Transfer Linear Layer weights
        if hasattr(custom_lstm, "linear") and hasattr(vanilla_lstm, "linear"):
            custom_lstm.linear.weight.data.copy_(vanilla_lstm.linear.weight.data)
            custom_lstm.linear.bias.data.copy_(vanilla_lstm.linear.bias.data)


def compute_ew_acf_step(x_t, x_lag, state, lambda_, epsilon=1e-8):
    """
    Core mathematical engine for a single EW-ACF step.
    Works with both single values and batches.
    """
    # Update running stats
    state["mean"] = lambda_ * state["mean"] + (1 - lambda_) * x_t
    state["var"] = lambda_ * state["var"] + (1 - lambda_) * (x_t - state["mean"]) ** 2
    state["var_lag"] = lambda_ * state["var_lag"] + (1 - lambda_) * (x_lag - state["mean"]) ** 2
    state["cov"] = lambda_ * state["cov"] + (1 - lambda_) * (x_t - state["mean"]) * (x_lag - state["mean"])

    # Calculate Correlation
    acf = state["cov"] / torch.sqrt(state["var"] * state["var_lag"] + epsilon)
    return acf, state


def ew_acf(time_series, lag, lambda_=0.5, last_only=False):
    """
    Calculates the exponential weighted autocorrelation function of a time series.
    Maintains backward compatibility with NumPy-based analysis.
    """
    if len(time_series) <= lag:
        return np.nan

    # Convert to torch for consistent precision with the loss function
    ts = torch.tensor(time_series, dtype=torch.float32)
    epsilon = 1e-8

    state = {"mean": torch.zeros(1), "var": torch.zeros(1) + epsilon, "var_lag": torch.zeros(1) + epsilon, "cov": torch.zeros(1)}

    acf_list = []
    for i in range(lag, len(ts)):
        acf_t, state = compute_ew_acf_step(ts[i], ts[i - lag], state, lambda_, epsilon)
        if not last_only:
            acf_list.append(acf_t.item())

    if last_only:
        return acf_t.item()

    return np.array(acf_list)


def std_acf(series, lag, last_only=False):
    """Calculates the standardized autocorrelation function of a time series.
    Args:
        series (array-like): The time series to calculate the autocorrelation function of.
        lag (int): The lag to calculate the autocorrelation at.
    Returns:
        float: The autocorrelation value with specified lag.
    """
    mean = np.mean(series)
    denominator = np.sum((series - mean) ** 2)

    if lag == 0:
        return 1.0
    else:
        numerator = np.sum((series[lag:] - mean) * (series[:-lag] - mean))
        return numerator / denominator
