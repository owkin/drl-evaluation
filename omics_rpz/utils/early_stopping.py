"""Utils functions for early stopping in AE and MLP models."""

from typing import Any

import numpy as np
import pandas as pd


def convert_to_array(dataset: Any) -> np.ndarray:
    """Convert dataset to dataset.values if dataframe.

    Args:
        dataset (Any): the dataset we wish to convert

    Returns:
        np.ndarray: the np.ndarray version of the data
    """
    if isinstance(dataset, pd.DataFrame):
        dataset = dataset.values
    return dataset


def initialize_early_stopping(eval_loss: list, early_stopping_best: float) -> float:
    """Initialize the current best loss at the first epoch Only useful for the AE, as
    the MLP metrics can be initialized to their worst value, ie 0.

    Args:
        eval_loss (list): the list of loss at every epoch
        early_stopping_best (float): the best validation performance so far

    Returns:
        float, either the initialized early_stopping_best or the current one
    """
    if len(eval_loss) == 1:  # first epoch
        early_stopping_best = eval_loss[0]
    return early_stopping_best


def compute_metric(metric: Any, val_metric: list, y_val: list, val_preds: list) -> list:
    """Compute the validation metric, either C-index or Spearman.

    Args:
        metric (Any): the metric configured in the mlp.yaml
        val_metric (list): the list of validation metric updated at every epoch
        y_val (list): the list of validation ground truth
        val_preds (list): the list of validation predictions for this epoch

    Returns:
        list: the updated list of validation metric
    """
    if "spearmanr" in str(metric):
        val_metric.append(metric(y_val, val_preds)[0])
    else:
        val_metric.append(metric(y_val, val_preds))
    return val_metric


def update_early_stopping(
    eval_list: list,
    early_stopping_best: float,
    early_stopping_delta: int,
    early_stopping_patience_count: int,
    use_metric: bool = False,
):
    """Update the best loss/metric value epoch by epoch, along with the patience count.
    If we use a metric (MLP), we want it to increase. If we use the loss (AE), we want
    it to decrease.

    Args:
        eval_list (list): either the metric (for MLP) or the loss at each epoch
        early_stopping_best (float): the current best performance on the eval set
        early_stopping_delta (float): the threshold for which we consider the model
            hasn't improved enough
        early_stopping_patience_count (int): the current number of epochs the model
            hasn't improved enough
        use_metric (bool): whether a metric is used, or the loss

    Returns:
        (float, int): the updated early_stopping_best and early_stopping_patience_count
    """
    if use_metric and (eval_list[-1] > early_stopping_best + early_stopping_delta):
        early_stopping_best = eval_list[-1]
        early_stopping_patience_count = 0
    elif not use_metric and (
        eval_list[-1] < early_stopping_best - early_stopping_delta
    ):
        early_stopping_best = eval_list[-1]
        early_stopping_patience_count = 0
    else:
        early_stopping_patience_count += 1
    return (early_stopping_best, early_stopping_patience_count)
