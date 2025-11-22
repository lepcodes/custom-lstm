import numpy as np


def ew_acf(time_series, lag, lambda_=0.5, last_only=False):
    """
    Calculates the exponential weighted autocorrelation function of a time series.
    Args:
        time_series (array-like): The time series to calculate the autocorrelation function of.
        lag (int): The lag to calculate the autocorrelation at.
        lambda_ (float): The exponential decay parameter.
    Returns:
        float: The autocorrelation value with specified lag.
    """
    if len(time_series) <= lag:
        return np.nan

    mean = 0.5
    variance = 0
    variance_lag = 0
    autocovariance = 0

    if not last_only:
        autocorrerlation_list = []

    for i in range(lag, len(time_series)):
        mean = lambda_ * mean + (1 - lambda_) * time_series[i]
        autocovariance = lambda_ * autocovariance + (1 - lambda_) * (
            time_series[i] - mean
        ) * (time_series[i - lag] - mean)
        variance = lambda_ * variance + (1 - lambda_) * (time_series[i] - mean) ** 2
        variance_lag = (
            lambda_ * variance_lag + (1 - lambda_) * (time_series[i - lag] - mean) ** 2
        )
        autocorrelation = autocovariance / np.sqrt(variance * variance_lag)
        if not last_only:
            autocorrerlation_list.append(autocorrelation)
        # print(f"Mean({i}): {mean}, Autocovariance({i}): {autocovariance}, Variance({i}): {variance}, VarianceLag({i}): {variance_lag}, Autocorrelation({i}): {autocorrelation}")

    if not last_only:
        return np.array(autocorrerlation_list)

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
