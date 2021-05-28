import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error


def normalized_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Computed the RMSE normalized by the standard deviation.

    Args:
        y_true: A [N, D] array of true values
        y_pred: A [N, D] array of predicted values
    Returns:
        The normalized RMSE
    """
    # Compute the RMSE entry-wise; [K, D] array
    errors = mean_squared_error(y_true=y_true,
                                y_pred=y_pred,
                                squared=False,
                                multioutput='raw_values')

    # Compute the average error over all elements; [D]
    avg_error = np.average(errors, axis=0)

    # Get the standard deviation for each feature; [D]
    std_dev = np.std(y_true, axis=0)
    
    # Normalize the error; [D]
    normalized_errors = avg_error / std_dev

    # Return the unweighted average error over all features
    return float(np.average(normalized_errors))


def normalized_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Computed the RMSE normalized by the standard deviation.

    Args:
        y_true: A [N, D] array of true values
        y_pred: A [N, D] array of predicted values
    Returns:
        The normalized RMSE
    """
    # Compute the RMSE entry-wise; [K, D] array
    errors = mean_absolute_error(y_true=y_true,
                                 y_pred=y_pred,
                                 multioutput='raw_values')

    # Compute the average error over all elements; [D]
    avg_error = np.average(errors, axis=0)

    # Compute the IQR for each feature
    third_quartile = np.percentile(y_true, 75, axis=0)
    first_quartile = np.percentile(y_true, 25, axis=0)
    iqr = third_quartile - first_quartile

    # Normalize the error; [D]
    normalized_errors = avg_error / iqr

    # Return the unweighted average error over all features
    return float(np.average(normalized_errors))
