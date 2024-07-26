"""Tests for the Random class."""
import numpy as np

from omics_rpz.transforms import Random


def test_random():
    """Basic test for the random class."""
    n_patients, n_rnaseq, repr_dim = 100, 1_000, 32

    X = np.random.normal(size=(n_patients, n_rnaseq))

    model = Random(repr_dim=repr_dim)
    embedding = model.fit_transform(X)

    assert embedding.shape == (n_patients, repr_dim)
