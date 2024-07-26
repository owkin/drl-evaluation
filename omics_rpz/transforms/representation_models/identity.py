"""Benchmark representation: identity function."""

import numpy as np
import pandas as pd
from typing_extensions import Self


class Identity:
    """Class that implements the identity function."""

    def __init__(self):
        pass

    def fit(self, X: pd.DataFrame, *args, **kwargs) -> Self:
        """Returning the same features that are given to the function.

        Args:
            X (pd.DataFrame): training matrix of shape (n_samples, n_features)
            args: for compatibility reasons with deep rpz methods
            kwargs: for compatibility reasons with deep rpz methods

        Returns:
            self (object):
            Returns the instance itself
        """
        del X
        del args
        del kwargs
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Returning the same features that are given to the function
        Args:
            X (array-like, sparse matrix of shape (n_samples, n_features)):
            Training vector where n_samples is the number
            of samples and n_features is the number of features
        Returns:
            X (np.ndarray of shape (n_samples, n_features)):
            Returns the initial matrix, without any change
        """
        return X.to_numpy()

    def fit_transform(self, X: pd.DataFrame, *args, **kwargs) -> np.ndarray:
        """Returning the same features that are given to the function
        Args:
            X (pd.DataFrame): training matrix of shape (n_samples, n_features)
            args: for compatibility reasons with deep rpz methods
            kwargs: for compatibility reasons with deep rpz methods
        """
        del args
        del kwargs
        self.fit(X)
        return self.transform(X)
