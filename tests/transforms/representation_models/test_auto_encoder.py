"""Tests of the class AutoEncoder."""
import numpy as np
import pandas as pd

from omics_rpz.transforms import AutoEncoder


def test_auto_encoder():
    """Basic test of the class AutoEncoder."""
    n_patients, n_rnaseq, repr_dim = 100, 1_000, 32

    X = np.random.normal(size=(n_patients, n_rnaseq))

    hidden_test_set = (
        {'n_layers': 0, 'hidden': []},
        {'n_layers': 1, 'n_units_1st': 512, 'hidden': [512]},
        {'n_layers': 1, 'n_units_1st': 256, 'hidden': [256]},
        {'n_layers': 2, 'n_units_1st': 512, 'rate': 0.5, 'hidden': [512, 256]},
        {'n_layers': 2, 'n_units_1st': 256, 'rate': 0.5, 'hidden': [256, 128]},
        {'n_layers': 2, 'n_units_1st': 256, 'rate': 1, 'hidden': [256, 256]},
        {'n_layers': 3, 'n_units_1st': 512, 'rate': 0.5, 'hidden': [512, 256, 128]},
        {'n_layers': 3, 'n_units_1st': 256, 'rate': 0.5, 'hidden': [256, 128, 64]},
        {'n_layers': 3, 'n_units_1st': 256, 'rate': 1, 'hidden': [256, 256, 256]},
        {'n_layers': 3, 'n_units_1st': 256, 'rate': 0.3, 'hidden': [256, 76, 22]},
    )

    for item in hidden_test_set:
        model = AutoEncoder(
            repr_dim=repr_dim,
            hidden_n_layers=item['n_layers'],
            hidden_n_units_first=item['n_units_1st'] if 'n_units_1st' in item else None,
            hidden_decrease_rate=item['rate'] if 'rate' in item else None,
        )
        assert model.convert_hidden_config() == item['hidden']

    embedding = model.fit_transform(pd.DataFrame(X), "metric suffix")

    assert embedding.shape == (n_patients, repr_dim)
