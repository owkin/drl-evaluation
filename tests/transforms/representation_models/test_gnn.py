"""Tests for the AutoEncoderMultiHead class."""

import os
import warnings

import numpy as np
import pandas as pd
import pytest

CI = os.environ.get("CI") in {"True", "true"}


@pytest.mark.skipif(CI, reason="not a test for CI")
def test_gnn():
    """Basic test of the class AutoEncoderMultiHead."""

    try:
        # pylint: disable-next=import-outside-toplevel
        from omics_rpz.data import load_graph, load_tcga

        # pylint: disable-next=import-outside-toplevel
        from omics_rpz.transforms import Gnn
    except ImportError:
        warnings.warn(
            UserWarning(
                "Omics_rpz was installed without GNN capabilities and user asked for"
                " GNN reprentation. Re-install with 'poetry install -E gnn'"
            )
        )
        return

    repr_dim = 256
    num_epochs = 10

    _, rnaseq_variables = load_tcga(cohort='BRCA')
    _, _, genes = load_graph(
        clustering='louvain', string_threshold=0.95, genes=rnaseq_variables[:100]
    )
    n_patients = 500
    label = np.random.uniform(size=n_patients)
    X = np.random.normal(size=(n_patients, len(genes)))
    X = pd.DataFrame(X, columns=genes)

    model = Gnn(
        repr_dim=repr_dim,
        hidden_channels=[4],
        out_dim=1,
        sigmoid=True,
        num_epochs=num_epochs,
        device='cuda',
    )

    embedding = model.fit_transform(
        X,
        pd.DataFrame(label, columns=['OS']),
        'test_gnn',
    )

    assert embedding.shape == (n_patients, repr_dim)
