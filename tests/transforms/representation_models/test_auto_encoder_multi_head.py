"""Tests for the AutoEncoderMultiHead class."""
import numpy as np
import pandas as pd

from omics_rpz.transforms import AutoEncoderMultiHead


def test_auto_encoder_multi_head():
    """Basic test of the class AutoEncoderMultiHead."""
    n_patients, n_rnaseq, n_clinical, repr_dim = 100, 1_000, 10, 32
    num_epochs = 100

    X_rnaseq = np.random.normal(size=(n_patients, n_rnaseq))
    X_clinical = np.random.normal(size=(n_patients, n_clinical))

    l_multi_heads = (
        ['gender'],
        ['age', 'gender'],
        ['gender', 'stage', 'age'],
    )
    l_betas = (
        [1, 0.1],
        [1, 0.8, 0.2],
        [1, 0.1, 1.5, 0.3],
    )
    l_losses = (
        ['BCE'],
        ['MSE', 'BCE'],
        ['BCE', 'BCE', 'MSE'],
    )
    l_hidden = (
        [[64]],
        [[64, 32], [128, 64, 32]],
        [[32, 16], [128, 64, 32, 16], [64, 32, 16]],
    )
    l_dropout = (
        [0.5],
        None,
        [0.2, 0.1, 0.2],
    )

    for multi_heads, betas, aux_losses, aux_hidden, aux_dropout in zip(
        l_multi_heads, l_betas, l_losses, l_hidden, l_dropout
    ):
        model = AutoEncoderMultiHead(
            repr_dim=repr_dim,
            hidden_n_layers=2,
            hidden_n_units_first=512,
            hidden_decrease_rate=0.5,
            dropout=0.5,
            bias=True,
            num_epochs=num_epochs,
            batch_size=32,
            learning_rate=0.0005,
            early_stopping_use=False,
            device='cpu',
            multi_heads=multi_heads,
            betas=betas,
            aux_losses=aux_losses,
            aux_hidden=aux_hidden,
            aux_dropout=aux_dropout,
        )
        df_clinical = pd.DataFrame(X_clinical)
        for i, head in enumerate(multi_heads):
            new_columns = list(df_clinical.columns.values)
            new_columns[i] = head
            df_clinical.columns = new_columns

        embedding = model.fit_transform(
            pd.DataFrame(X_rnaseq),
            f'rep{0} split{0}',
            df_labels=df_clinical,
        )

        assert model.hidden == [512, 256]
        assert embedding.shape == (n_patients, repr_dim)
        assert len(model.heads_nn) == len(model.betas) - 1 == len(model.aux_losses)
        assert len(model.train_loss) == num_epochs


if __name__ == "__main__":
    test_auto_encoder_multi_head()
