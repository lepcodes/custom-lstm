import numpy as np
import torch


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


def ew_acf(time_series, lag, lambda_=0.5, last_only=False):
    """
    Calculates the exponential weighted autocorrelation function of a time series.
    Uses dual-variance tracking for robustness against regime shifts.
    """
    if len(time_series) <= lag:
        return np.nan

    epsilon = 1e-8
    mean = 0.0
    variance = epsilon
    variance_lag = epsilon
    autocovariance = 0

    if not last_only:
        autocorrelation_list = []

    for i in range(lag, len(time_series)):
        x_t = time_series[i]
        x_lag = time_series[i - lag]

        mean = lambda_ * mean + (1 - lambda_) * x_t

        autocovariance = lambda_ * autocovariance + (1 - lambda_) * (x_t - mean) * (x_lag - mean)
        variance = lambda_ * variance + (1 - lambda_) * (x_t - mean) ** 2
        variance_lag = lambda_ * variance_lag + (1 - lambda_) * (x_lag - mean) ** 2

        autocorrelation = autocovariance / np.sqrt(variance * variance_lag + epsilon)

        if not last_only:
            autocorrelation_list.append(autocorrelation)

    if not last_only:
        return np.array(autocorrelation_list)

    return autocorrelation


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


def generate_regime_data(n_points, sampling_rate=100, freq1=20, freq2=100, freq3=20, noise_level=0.1, random_seed=42):
    """
    Generates data where the duration SCALES with n_points,
    ensuring 'dt' (time step) remains constant.
    """
    np.random.seed(random_seed)

    duration = n_points / sampling_rate

    t = np.arange(0, duration, 1 / sampling_rate)
    t = t[:n_points]

    n1 = int(n_points / 3)
    n2 = int(n_points / 3)

    t1 = t[:n1]
    t2 = t[n1 : n1 + n2]
    t3 = t[n1 + n2 :]

    regime_1 = np.sin(2 * np.pi * freq1 * t1)
    regime_2 = np.sin(2 * np.pi * freq2 * t2)
    regime_3 = np.sin(2 * np.pi * freq3 * t3)

    # Concatenate
    clean_signal = np.concatenate((regime_1, regime_2, regime_3))

    # Add noise
    noise = np.random.normal(0, noise_level, clean_signal.shape)
    noisy_signal = clean_signal + noise

    data_tensor = torch.from_numpy(noisy_signal).float().reshape(-1, 1)

    return data_tensor, clean_signal
