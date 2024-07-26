"""Wrapper functions for representation models' methods."""
from typing import Any

import pandas as pd


def rpz_transform_wrapper(
    rpz_model: Any,
    X_df: pd.DataFrame,
    columns_to_transform: list[str] = None,
    data_name: str = 'rna_seq',
) -> pd.DataFrame:
    """Wrapper for the transform method of all representation models, building a pd
    DataFrame from the np ndarray returned by the method.

    Args:
        rpz_model (Any): Any representation model object.
        X_df (pd.DataFrame): Dataframe from which a subset of columns to be
            transformed are input to the representation model.
        columns_to_transform (list[str]): Columns to be transformed.
        data_name (str, optional): String to appear in all transformed columns in the
            transformed DataFrame.

    Returns:
        pd.DataFrame: the transformed DataFrame.
    """
    if columns_to_transform is None:
        columns_to_transform = X_df.columns
    X_emb = rpz_model.transform(X_df[columns_to_transform])

    X_emb_df = pd.DataFrame(data=X_emb, index=X_df.index)
    X_emb_df = X_emb_df.add_prefix(f"{data_name}_{type(rpz_model).__name__}_")

    # Replaces columns_to_transform by the transformed columns
    return pd.concat([X_df, X_emb_df], axis=1).drop(columns=columns_to_transform)


def rpz_fit_transform_wrapper(
    rpz_model: Any,
    X_df: pd.DataFrame,
    columns_to_fit_transform: list[str] = None,
    data_name: str = 'rna_seq',
) -> pd.DataFrame:
    """Wrapper for the fit_transform method of all representation models.

    Fit model and then call rpz_transform_wrapper.

    Args:
        rpz_model (Any): Any representation model object.
        X_df (pd.DataFrame): Dataframe from which a subset of columns to be
            transformed are input to the representation model.
        columns_to_fit_transform (list[str]): Columns to be transformed.
        data_name (str, optional): String to appear in all transformed columns in the
            transformed DataFrame.

    Returns:
        pd.DataFrame: the transformed DataFrame.
    """
    if columns_to_fit_transform is None:
        columns_to_fit_transform = X_df.columns
    # Fit model.
    rpz_model.fit(X_df[columns_to_fit_transform])

    return rpz_transform_wrapper(rpz_model, X_df, columns_to_fit_transform, data_name)
