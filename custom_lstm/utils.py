import numpy as np
import torch
from torch import nn


class EWACFEngine(nn.Module):
    """
    Rigorously causal stateful engine to compute Multi-Lag EW-ACF.
    Maintains a 'Memory Bank' of signal history and its temporal context (mean, variance).
    Uses fully vectorized updates for maximum performance.
    """

    def __init__(self, lambda_=0.5, lags=[1], epsilon=1e-8):
        super(EWACFEngine, self).__init__()
        self.lambda_ = lambda_
        self.lags = sorted(lags)
        self.max_lag = max(lags)
        self.epsilon = epsilon

        # State: Cumulative statistics across the entire time series
        self.register_buffer("mean", torch.tensor([]))
        self.register_buffer("variance", torch.tensor([]))
        self.register_buffer("covariance", torch.tensor([]))  # (L, B, F)

        # Memory Bank: Rolling history of context for causal centering
        self.register_buffer("x_hist", torch.tensor([]))
        self.register_buffer("mu_hist", torch.tensor([]))
        self.register_buffer("v_hist", torch.tensor([]))

    def _lazy_init(self, B, F, device):
        """Sizes the buffers on the first forward pass based on batch/features."""
        L = len(self.lags)
        if self.mean.numel() == 0:
            self.register_buffer("mean", torch.zeros(B, F, device=device))
            self.register_buffer("variance", torch.full((B, F), self.epsilon, device=device))
            self.register_buffer("covariance", torch.zeros(L, B, F, device=device))

    def forward(self, sequence):
        """
        Computes the ACF for all lags in parallel.
        Returns:
            torch.Tensor: (batch, seq_valid, features, num_lags)
        """
        B, T, F = sequence.shape
        L = len(self.lags)
        device = sequence.device
        self._lazy_init(B, F, device)

        # 1. Context Preparation: Combine saved history with current chunk
        X = torch.cat([self.x_hist, sequence], dim=1) if self.x_hist.numel() > 0 else sequence

        # Memory Bank trackers for the current combined sequence
        MU = torch.zeros(B, X.shape[1], F, device=device)
        V = torch.zeros(B, X.shape[1], F, device=device)

        offset = self.x_hist.shape[1] if self.x_hist.numel() > 0 else 0
        if offset > 0:
            MU[:, :offset, :] = self.mu_hist
            V[:, :offset, :] = self.v_hist

        all_acf = []
        lag_indices = torch.tensor(self.lags, device=device)

        # 2. Main Temporal Loop
        for t in range(offset, X.shape[1]):
            x_t = X[:, t, :]

            # Update Today's context
            self.mean = self.lambda_ * self.mean + (1 - self.lambda_) * x_t
            self.variance = self.lambda_ * self.variance + (1 - self.lambda_) * (x_t - self.mean) ** 2

            MU[:, t, :] = self.mean
            V[:, t, :] = self.variance

            # --- FULLY VECTORIZED PARALLEL LAG UPDATES ---
            t_minus_lags = t - lag_indices
            valid_mask = (t_minus_lags >= 0).view(L, 1, 1)  # Causal validity mask
            safe_idx = torch.clamp(t_minus_lags, min=0)

            # Retrieve "Yesterday" context from the Memory Bank for all lags
            x_lag = X[:, safe_idx, :].permute(1, 0, 2)  # (L, B, F)
            mu_lag = MU[:, safe_idx, :].permute(1, 0, 2)  # (L, B, F)

            # Recursive Covariance Update: Centers current and historical surprises
            # update = (Surprise_today * Surprise_yesterday)
            update = (x_t - self.mean).unsqueeze(0) * (x_lag - mu_lag)

            # Apply masked update to covariance buffer
            self.covariance = self.lambda_ * self.covariance + (1 - self.lambda_) * torch.where(valid_mask, update, torch.zeros_like(self.covariance))

            # Record Output if we reach the validity threshold (All lags valid)
            if t >= self.max_lag:
                v_lag = V[:, safe_idx, :].permute(1, 0, 2)  # (L, B, F)
                # Correlation (Rho) = Cov / sqrt(Var_today * Var_yesterday)
                rho_t = self.covariance / torch.sqrt(self.variance * v_lag + self.epsilon)
                all_acf.append(rho_t.permute(1, 2, 0))  # (B, F, L)

        # 3. Persistence: Save the tail context for the next TBPTT chunk
        self.x_hist = X[:, -self.max_lag :, :].detach()
        self.mu_hist = MU[:, -self.max_lag :, :].detach()
        self.v_hist = V[:, -self.max_lag :, :].detach()

        if not all_acf:
            return torch.empty(B, 0, F, L, device=device)

        return torch.stack(all_acf, dim=1)

    def reset_state(self):
        """Full reset of statistics and memory bank."""
        if self.mean.numel() > 0:
            self.mean.zero_()
            self.variance.fill_(self.epsilon)
            self.covariance.zero_()
        self.x_hist = torch.tensor([], device=self.x_hist.device)
        self.mu_hist = torch.tensor([], device=self.mu_hist.device)
        self.v_hist = torch.tensor([], device=self.v_hist.device)


def transfer_weights(vanilla_lstm, custom_lstm):
    target = custom_lstm.lstm
    source = vanilla_lstm.lstm
    with torch.no_grad():
        target.W_hi.copy_(source.W_hi.data)
        target.W_xi.copy_(source.W_xi.data)
        target.b_i.copy_(source.b_i.data)
        target.W_hc.copy_(source.W_hc.data)
        target.W_xc.copy_(source.W_xc.data)
        target.b_c.copy_(source.b_c.data)
        target.W_ho.copy_(source.W_ho.data)
        target.W_xo.copy_(source.W_xo.data)
        target.b_o.copy_(source.b_o.data)
        if hasattr(custom_lstm, "linear") and hasattr(vanilla_lstm, "linear"):
            custom_lstm.linear.weight.data.copy_(vanilla_lstm.linear.weight.data)
            custom_lstm.linear.bias.data.copy_(vanilla_lstm.linear.bias.data)


def compute_ew_acf_step(x_t, x_lag, mu_t, mu_lag, var_t, var_lag, cov_prev, lambda_, epsilon=1e-8):
    cov_new = lambda_ * cov_prev + (1 - lambda_) * (x_t - mu_t) * (x_lag - mu_lag)
    acf = cov_new / torch.sqrt(var_t * var_lag + epsilon)
    return acf, cov_new


def ew_acf(time_series, lag, lambda_=0.5, last_only=False):
    if len(time_series) <= lag:
        return np.nan
    ts = torch.tensor(time_series, dtype=torch.float32)
    epsilon = 1e-8
    mu, var, cov = torch.zeros(1), torch.zeros(1) + epsilon, torch.zeros(1)
    mu_history, var_history, acf_list = [], [], []
    for t in range(len(ts)):
        x_t = ts[t]
        mu = lambda_ * mu + (1 - lambda_) * x_t
        var = lambda_ * var + (1 - lambda_) * (x_t - mu) ** 2
        mu_history.append(mu.clone())
        var_history.append(var.clone())
        if t >= lag:
            x_lag, m_lag, v_lag = ts[t - lag], mu_history[t - lag], var_history[t - lag]
            acf_t, cov = compute_ew_acf_step(x_t, x_lag, mu, m_lag, var, v_lag, cov, lambda_, epsilon)
            acf_list.append(acf_t.item())
    return acf_t.item() if last_only else np.array(acf_list)


def std_acf(series, lag, last_only=False):
    mean = np.mean(series)
    denominator = np.sum((series - mean) ** 2)
    if lag == 0:
        return 1.0
    numerator = np.sum((series[lag:] - mean) * (series[:-lag] - mean))
    return numerator / denominator
