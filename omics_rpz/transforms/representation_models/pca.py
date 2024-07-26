"""Dimension reduction method based on principal component analysis."""
import numpy as np
import pandas as pd
import sklearn.decomposition
from typing_extensions import Self


class PCA(sklearn.decomposition.PCA):
    """Class implementing principal component analysis.

    Parameters:
        repr_dim (int, default = 36) : number of dimensions composed by
        random vectors acting as a dimension reduction
    Arguments:
        components_: Principal axes in feature space, representing the directions
        of maximum variance in the data.
        Equivalently, the right singular vectors of the centered input data,
        parallel to its eigenvectors.The components are sorted by explained_variance_
    """

    def __init__(self, repr_dim: int):
        super().__init__(n_components=repr_dim, random_state=42)
        self.repr_dim = repr_dim

    def metagenes(self, X: pd.DataFrame) -> np.ndarray:
        """To calculate the metagenes that are derived from the dim reduction
        Args:
            X (array-like, sparse matrix of shape (n_samples, n_features)):
            Training vector where n_samples is the number
            of samples and n_features is the number of features.
        Returns:
            self.components_ (object) : Returns the components derived,
            also called metagenes in this context
        """
        self.fit(X)
        return self.components_

    def fit(self, X: pd.DataFrame, *args, **kwargs) -> Self:
        """Fit the model with X.

        Args:
            X (pd.DataFrame): training matrix of shape (n_samples, n_features)
            args: for compatibility reasons with deep rpz methods
            kwargs: for compatibility reasons with deep rpz methods

        Returns
            self (object): The instance itself
        """
        del args
        del kwargs
        return super().fit(X)

    def fit_transform(self, X: pd.DataFrame, *args, **kwargs) -> np.ndarray:
        """Fit the model with X and apply the dimensionality reduction on X.

        Args:
            X (pd.DataFrame): training matrix of shape (n_samples, n_features)
            args: for compatibility reasons with deep rpz methods
            kwargs: for compatibility reasons with deep rpz methods

        Returns
            X_new (ndarray of shape (n_samples, n_rid_dimension)):
            Transformed values.
        """
        del args
        del kwargs
        return super().fit_transform(X)
