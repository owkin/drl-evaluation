"""Module implementing utility functions for preprocessing."""
import pandas as pd
from sklearn.preprocessing import LabelEncoder

VAR_TO_IMPUTE = {"cat": ["race", "stage", "grade", "TSS"], "num": ["age"]}


def handle_nan_values(dataframe: pd.DataFrame, var_to_impute: dict = None):
    """Handle nan values by replacing either by the median for continuous columns or by
    "Unknown" for categorical.

    Parameters
    ----------
    dataframe : pd.DataFrame
        The input dataframe
    var_to_impute: dict
        The variables to impute with two keys: cat (categorical variables), num
        (numerical variables), each containing a list of variables to impute

    Returns
    -------
    pd.DataFrame
        The transformed dataframe without NA values
    """
    if var_to_impute is None:
        var_to_impute = VAR_TO_IMPUTE

    # Fill NaN values with median (cont.) or "Unknown" (cat.)
    dataframe[var_to_impute["cat"]] = dataframe[var_to_impute["cat"]].fillna("Unknown")
    dataframe[var_to_impute["num"]] = dataframe[var_to_impute["num"]].fillna(
        dataframe[var_to_impute["num"]].median()
    )

    return dataframe


def encode(dataframe: pd.DataFrame, columns: list[str]):
    """Encodes labels at specific columns of a given dataframe.

    Parameters
    ----------
    dataframe : pd.DataFrame
        The input dataframe
    columns : list[str]
        list of columns to encode

    Returns
    -------
    pd.Dataframe
        Dataframe with encoded values at columns `columns`
    """
    encoders = {}
    for c in columns:
        if c in dataframe.columns:
            encoders[c] = LabelEncoder().fit(dataframe[c])
            dataframe[c] = encoders[c].transform(dataframe[c]).astype(int)
    return dataframe, encoders
