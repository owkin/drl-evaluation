"""File to regroup gene essentiality specific metrics."""

import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import r2_score


def compute_regression_score(df_predictions: pd.DataFrame, metric: str = "r2") -> float:
    """Compute a regression score on a df_predictions containig y_true and y_pred.

    Args:
        df_predictions (pd.DataFrame): Dataframe with first row = y_true and second
            row = y_pred.
        metric (str, optional): Metric to compute. Defaults to "r2".

    Raises:
        ValueError: If the metric is not implemented.

    Returns:
        float: regression score.
    """

    y_true, y_pred = df_predictions.values[:, 0], df_predictions.values[:, 1]
    if metric == "r2":
        return r2_score(y_true, y_pred)
    if metric == "pearson":
        return pearsonr(y_true, y_pred)[0]
    if metric == "mse":
        return mse(y_true, y_pred)
    if metric == "mae":
        return mae(y_true, y_pred)
    if metric == "spearman":
        return spearmanr(y_true, y_pred)[0]

    raise ValueError("Metric not implemented")


def evaluated_per_depoi(
    y_true: pd.Series, y_pred: pd.Series, metric: str = 'r2'
) -> pd.Series:
    """Compute the metric of interest per DepOI.

    Args:
        y_true (pd.Series): True values to evaluate the predictions.
        y_pred (pd.Series): Predicted scores from the model.
        metric (str, optional): Metric to compute. Defaults to 'r2'.

    Returns:
        pd.Series: Series with DepOI in index and in values the corresponding metric
            score.
    """
    y_both = pd.merge(y_true, y_pred, how='inner', left_index=True, right_index=True)
    return y_both.groupby("DepOI").apply(
        lambda df_one_depoi_preds: compute_regression_score(df_one_depoi_preds, metric)
    )


def evaluated_per_cell_line(
    y_true: pd.Series, y_pred: pd.Series, metric: str = 'r2'
) -> pd.DataFrame:
    """Compute the metric of interest per cell line.

    Args:
        y_true (pd.Series): True values to evaluate the predictions.
        y_pred (pd.Series): Predicted scores from the model.
        metric (str, optional): Metric to compute. Defaults to 'r2'.

    Returns:
        pd.Series: Series with cell line in index and in values the corresponding metric
            score.
    """
    y_both = pd.merge(y_true, y_pred, how='inner', left_index=True, right_index=True)
    return y_both.groupby("DepMap_ID").apply(
        lambda df_one_cell_preds: compute_regression_score(df_one_cell_preds, metric)
    )
