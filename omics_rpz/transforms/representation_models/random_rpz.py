"Benchmark for representation methods having random vectors as reduction methods"
import numpy as np
import pandas as pd
from typing_extensions import Self


class Random:
    """Class that implements the random selection method as a dimension reduction
    Parameters:
        repr_dim : number of dimensions composed by random vectors
            acting as a dimension reduction
    """

    def __init__(self, repr_dim: int):
        self.repr_dim = repr_dim

    def fit(self, X: pd.DataFrame, *args, **kwargs) -> Self:
        """Fits the random reduction method.

        Args:
            X (pd.DataFrame): training matrix of shape (n_samples, n_features)
            args: for compatibility reasons with deep rpz methods
            kwargs: for compatibility reasons with deep rpz methods

        Returns:
            self (object): returns the object itself
        """
        del X
        del args
        del kwargs
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Apply the dimensionality reduction on X.

        Args:
            X (np.ndarray, array-like of shape (n_samples, n_features)):
            Training data, where `n_samples` is the number of samples
            and `n_features` is the number of features.
        Returns:
            X_new (np.ndarray) : ndarray of shape (n_samples, repr_dim)
            Transformed values: random vectors.
        """
        return np.random.normal(size=(len(X), self.repr_dim))

    def fit_transform(self, X: pd.DataFrame, *args, **kwargs) -> np.ndarray:
        """Fit the model with X and apply the dimensionality reduction on X.

        Args:
            X (pd.DataFrame): training matrix of shape (n_samples, n_features)
            args: for compatibility reasons with deep rpz methods
            kwargs: for compatibility reasons with deep rpz methods

        Returns:
            self(object) : Returns ndarray of shape (n_samples, repr_dim)
        """
        del args
        del kwargs
        return self.transform(X)
