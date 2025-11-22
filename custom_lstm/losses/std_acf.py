import numpy as np


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
