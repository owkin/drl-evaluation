"""Tests for the PCA Class."""
import numpy as np

from omics_rpz.transforms import PCA


def test_pca():
    """Basic test for the PCA class."""
    n_patients, n_rnaseq, repr_dim = 100, 1_000, 32

    X = np.random.normal(size=(n_patients, n_rnaseq))

    model = PCA(repr_dim=repr_dim)
    embedding = model.fit_transform(X)

    assert embedding.shape == (n_patients, repr_dim)
