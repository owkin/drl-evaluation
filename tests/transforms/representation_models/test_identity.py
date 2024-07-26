"""Tests for the class Identity."""
import numpy as np
import pandas as pd

from omics_rpz.transforms import Identity


def test_identity():
    """Basic test of the class Identity."""
    n_patients, n_rnaseq = 100, 1_000

    X = pd.DataFrame(np.random.normal(size=(n_patients, n_rnaseq)))

    model = Identity()
    embedding = model.fit_transform(X)

    assert embedding.shape == (n_patients, n_rnaseq)
