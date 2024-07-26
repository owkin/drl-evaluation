"""Wrapper class for the scVI model."""

from typing import Optional

import anndata as ad
import mlflow
import numpy as np
import pandas as pd
import scanpy as sc
import scvi
import torch


class ScVI:
    """Wrapper class for the Representation model: scVI. 'Deep generative modeling for
    single-cell transcriptomics'.  [Lopez et. al, 2018]
    https://pubmed.ncbi.nlm.nih.gov/30504886/

    Parameters:
         repr_dim: int, Size of the representation dimension.
         hidden_n_layers: int = 1, number of hidden layers
         hidden_n_units_first: int = 256, number of units of the first hidden layer
         dropout: float, Probability p corresponding to all hidden layers. During
            training, randomly zeroes (with p probability) some units of the hidden
            layers.
         num_epochs: Optional[int] = 300, Maximum number of epochs for the scVI trainer
         batch_size: Optional[int] = 256, Size of minibatches for stochastic optimizers.
             If the solver is "lbfgs", the classifier will not use minibatch.
         max_num_epochs: int = 300, Maximum number of iterations in the case of early
             stopping. Otherwise num_epochs is used.
        early_stopping_use: bool = True, Whether to split the data in train val during
             training of the autoencoder and perform early stopping with the val set
             (Done within the scVI Trainer class )
        gene_likelihood: str = 'nb', Distribution of the decoded data.
            choose between "nb" (Negative Binomial) and "poisson" (Poisson)
        device: Optional[str] = "cpu", Whether to use cpu or CUDA to run computations.
            random_state: int, by default set to 42, random seed for the model.
        random_state: int, by default set to 42, random seed for the model.

    Attributes:
        adata: Anndata, object where the training data is stored.
    """

    def __init__(
        self,
        repr_dim: int = 64,
        hidden_n_layers: int = 1,
        hidden_n_units_first: int = 256,
        dropout: float = 0.1,
        num_epochs: Optional[int] = 300,
        batch_size: Optional[int] = 256,
        max_num_epochs: int = 300,
        early_stopping_use: bool = True,
        gene_likelihood: str = "nb",
        device: Optional[str] = "cpu",
        random_state: int = 42,
    ):
        self.random_state = random_state
        torch.manual_seed(self.random_state)
        self.repr_dim = repr_dim
        self.hidden_n_layers = hidden_n_layers
        self.hidden_n_units_first = hidden_n_units_first
        self.dropout = dropout
        self.num_epochs = num_epochs
        self.max_num_epochs = max_num_epochs
        self.batch_size = batch_size
        self.early_stopping_use = early_stopping_use
        self.gene_likelihood = gene_likelihood
        self.device = device

        # Where the data will be stored at each training run
        self.adata = None

    def _init_models(self):
        """Init method to register the Anndata object and initializes the scVI model."""
        scvi.model.SCVI.setup_anndata(
            self.adata,
            layer="counts",
        )
        # pylint: disable=attribute-defined-outside-init
        self.model = scvi.model.SCVI(
            self.adata,
            n_hidden=self.hidden_n_units_first,
            n_latent=self.repr_dim,
            n_layers=self.hidden_n_layers,
            dropout_rate=self.dropout,
            dispersion="gene",
            latent_distribution="normal",
            gene_likelihood=self.gene_likelihood,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Computes the encoding and decodings of the input x.

        Args:
            x (torch.Tensor): input tensor.

        Returns:
            tensor (x_dec) reconstructed by the network.
        """
        return self.model.module.forward(x)

    def fit(
        self,
        X: pd.DataFrame,
        metrics_suffix: str = None,
        **kwargs,
    ):
        """Fit the scVI model according to the given training data X.

        Args:
            X (pd.DataFrame): training data.
            metrics_suffix (str): repeat/split information to be logged with the loss.
            **kwargs: variable number of keyword arguments, to be compatible with
                the method from parent class.
        Returns:
            self with fitted decoder/encoder.
        """
        del kwargs
        adata = ad.AnnData(
            X=X.values.astype(int),
            obs=list(X.index),
            var=list(X.columns),
        )
        # Preprocessing
        # Preserve counts
        adata.layers["counts"] = adata.X.copy()
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        adata.raw = adata  # freeze the state in `.raw`
        adata.obs.columns = ["Donor"]
        adata.var.columns = ["Gene"]
        assert (adata.layers["counts"] == np.nan).mean() == 0.0

        # Setup Anndata
        self.adata = adata
        # Initialize model
        self._init_models()
        # Fit model
        use_gpu = self.device == "cuda"

        self.model.train(
            log_every_n_steps=5,
            max_epochs=self.max_num_epochs,
            use_gpu=use_gpu,
            train_size=0.9,
            batch_size=self.batch_size,
            early_stopping=self.early_stopping_use,
        )

        # Log the losses
        # pylint: disable=unsubscriptable-object
        for key in self.model.history.keys():
            values = list(self.model.history[key].values.flatten())
            for epoch, value in enumerate(values):
                mlflow.log_metric(
                    f"train-{key}-{metrics_suffix} Loss",
                    value,
                    step=epoch,
                )

        return self.model.module

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Encode the data using the fitted scVI model.

        Args:
            X (pd.DataFrame): training data.

        Returns:
            np.ndarray: encoded data.
        """
        adata = ad.AnnData(
            X=X.values,
            obs=list(X.index),
            var=list(X.columns),
        )
        # Preprocessing
        adata.layers["counts"] = adata.X.copy()  # .astype(int)  # preserve counts
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        adata.raw = adata  # freeze the state in `.raw`
        adata.obs.columns = ["Donor"]
        adata.var.columns = ["Gene"]

        assert (adata.layers["counts"] == np.nan).mean() == 0.0

        latent = self.model.get_latent_representation(adata)

        return latent

    def fit_transform(
        self, X: pd.DataFrame, metrics_suffix: str, **kwargs
    ) -> np.ndarray:
        """Fit the model, then encode the data using the fitted model.

        Args:
            X (pd.DataFrame): training data.
            metrics_suffix (str): repeat/split information to be logged with the loss
            **kwargs: variable number of keyword arguments, to be compatible with
                method from parent class.

        Returns:
            np.ndarray: encoded data.
        """
        del kwargs
        _ = self.fit(X, metrics_suffix)
        return self.transform(X)
