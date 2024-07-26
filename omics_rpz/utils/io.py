"""Utils fonction to load and save pickles."""

import os
import pathlib
import pickle
from typing import Any, Optional, Union

import numpy as np
import pandas as pd


def save_predictions(
    y_true: Union[pd.Series, pd.DataFrame], y_pred: np.ndarray, path: os.PathLike
):
    """Save task predictions with index under the parquet format.

    Args:
        y_true (Union[pd.Series, pd.DataFrame]): True values
        y_pred (np.ndarray): Predictions
        path (os.PathLike): Where to store the result
    """
    # Save experiment results
    if isinstance(y_true, pd.Series):
        res = y_true.to_frame()
    else:
        res = y_true.copy()

    columns_true = (
        ["y_true"]
        if res.shape[1] == 1
        else ["y_true_" + str(i) for i in range(res.shape[1])]
    )
    column_pred = (
        "y_pred"
        if len(y_pred.shape) == 1
        else ["y_pred_" + str(i) for i in range(y_pred.shape[1])]
    )

    res.columns = columns_true
    res[column_pred] = y_pred
    res.to_parquet(path)


def load_pickle(path: Union[str, pathlib.Path]) -> Any:
    """Load an object from a .pkl file.

    Parameters
    ----------
    path : Union[str, pathlib.Path]
        Path to load the object.
    Returns
    -------
    Loaded object
    """
    with open(path, "rb") as file:
        obj = pickle.load(file)
    return obj


def save_pickle(path: Union[str, pathlib.Path], obj) -> None:
    """Save an object as a .pkl file.

    Parameters
    ----------
    path : Union[str, pathlib.Path]
        Path to save the object.
    obj : Object
        Object to save.
    """
    with open(path, "wb") as file:
        pickle.dump(obj, file, pickle.HIGHEST_PROTOCOL)


def define_remote_prefix(
    experiment_folder: os.PathLike, keep_tree_from: Optional[str]
) -> str:
    """If `keep_tree_from` is within the `experiment folder`, it will keep the relative
    path between itself and  the experiment folder as a prefix. Otherwise, the prefix
    will be `results`.

    Args:
        experiment_folder (os.PathLike): Given path of the results.
        keep_tree_from (str): Start point of the folder trees to keep

    Returns:
        str: Prefix
    """

    abs_remote_parent = str(experiment_folder.resolve().parent)
    if keep_tree_from is not None and keep_tree_from in abs_remote_parent:
        remote_prefix = (
            f"results{keep_tree_from}{abs_remote_parent.rsplit(keep_tree_from, 1)[-1]}"
        )
    else:
        remote_prefix = "results"

    return remote_prefix
