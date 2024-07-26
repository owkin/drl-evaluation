"""Utils functions for metric computations."""
from typing import Any

import numpy as np


def compute_bootstrapped_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric: Any,
    seed: int = 0,
    n_bootstraps: int = 1000,
    sampling_rate: float = 1.0,
) -> list:
    """Compute metric on bootstrapped subsamples of the test set.

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth labels
    y_pred : np.ndarray
        Predicted labels
    metric : Any
        metric function
    seed : int, optional
        seed for random number generation, by default 0
    n_bootstraps : int, optional
        number of bootstraps, by default 1000
    sampling_rate : float, optional
        proportion of the dataset to sample per bootstrap, by default 1.0

    Returns
    -------
    list
        list of metric values
    """
    np.random.seed(seed)
    metric_values = []
    for _ in range(n_bootstraps):
        random_indices = np.random.choice(
            len(y_pred),
            size=int(len(y_pred) * sampling_rate),
            replace=True,
        )
        y_subsample_hat, y_subsample_test = (
            y_pred[random_indices],
            y_true[random_indices],
        )
        metric_values.append(metric(y_subsample_test, y_subsample_hat))
    return metric_values
