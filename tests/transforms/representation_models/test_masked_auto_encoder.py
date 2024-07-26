"""Tests for the MaskedAutoencoder class."""
import numpy as np
import pandas as pd

from omics_rpz.transforms import MaskedAutoencoder


def test_vime():
    """Basic test of the class MaskedAutoencoder."""
    n_patients, n_rnaseq, repr_dim = 100, 1_000, 32
    num_epochs = 2

    X = np.random.normal(size=(n_patients, n_rnaseq))

    model = MaskedAutoencoder(
        repr_dim=repr_dim,
        hidden_n_layers=2,
        hidden_n_units_first=128,
        hidden_decrease_rate=0.5,
        dropout=0,
        bias=True,
        num_epochs=num_epochs,
        batch_size=32,
        learning_rate=0.001,
        early_stopping_use=False,
        device='cpu',
        use_vime_mask=True,
    )

    assert model.hidden == [128, 64]

    embedding = model.fit_transform(pd.DataFrame(X), f'rep{0} split{0}')

    assert embedding.shape == (n_patients, repr_dim)
