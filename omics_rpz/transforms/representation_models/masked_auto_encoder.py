"""Class implementing the Masked Autoencoder representation model.

This module implements the methods for class MaskedAutoencoder which overrides methods
from the parent class AutoEncoder.
"""

import copy
from typing import Optional

import mlflow
import numpy as np
import pandas as pd
import torch
from loguru import logger
from sklearn.metrics import accuracy_score as acc
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from typing_extensions import Self

from omics_rpz.data import OmicsDataset
from omics_rpz.transforms.representation_models.auto_encoder import AutoEncoder
from omics_rpz.utils import initialize_early_stopping, update_early_stopping


class MaskedAutoencoder(AutoEncoder):
    """Implementation of Masked Autoencoder for self-supervised pre-training as
    described in the paper: "VIME: Extending the Success of Self- and- Semi-supervised
    Learning to Tabular Domain [202Ø]".

    Inherits from class Autoencoder and adds:
      1. Pretraining with masking
      2. Adds the option to train with reconstruction and mask prediction loss

    Parameters:
    -----------
        repr_dim (int) : dimensionality of the bottleneck
        hidden_n_layers (int): number of hidden layers
        hidden_n_units_first (int): number of units of the first hidden layer
        hidden_decrease_rate (float): decrease rate of the number of units / layer
        dropout (float): probability of the dropout at all hidden layers
        activation (torch.nn.Module): Activation layer
        bias (bool): If True, bias will be used
        num_epochs (int): Number of epochs for training
        batch_size (int): Batch size
        learning_rate (float): Learning rate
        early_stopping_use (bool): If True, the data will be split for evaluation
        max_num_epochs: int, Maximum number of iterations in the case of early
             stopping. Otherwise num_epochs is used.
        early_stopping_split: float, the train/val split proportion for the early
             stopping val to use
        early_stopping_patience: int, if the model doesn't improve for n_patience by
             more than early_stopping_delta, stop the training
        early_stopping_delta: float, if the model doesn't improve for n_patience by
             more than early_stopping_delta, stop the training
        device (str): String representing the type of device (GPU or CPU)
        random_state (int): Random seed
        alpha (float): Weight of the mask prediction loss (Binary CE)
        beta (float): Noise level (used when use_noise_mask set to True)
        corruption_proba (float): Masking probability
        compute_reconstruction (bool): If True, computes the reconstruction
             loss in forward pass
        compute_reconstruction_masked (bool): if True, computes the reconstruction
             loss in forward pass but only for the masked values
        predict_mask (bool): If True, predicts the mask in froward pass
        use_vime_mask (bool): if True, uses the VIME mask
        use_noise_mask (bool): if True, uses the Noise mask
        use_full_noise_mask (bool): if True, uses the full Noise mask
    Attributes:
    -----------
        mask_decoder (torch.nn.Module): Decoder network to reconstruct the mask
    """

    def __init__(
        self,
        repr_dim: int = 64,
        hidden_n_layers: int = 2,
        hidden_n_units_first: int = 512,
        hidden_decrease_rate: float = 0.5,
        dropout: float = None,
        activation: Optional[torch.nn.Module] = torch.nn.ReLU(),
        bias: bool = True,
        num_epochs: int = 10,
        batch_size: Optional[int] = 16,
        learning_rate: Optional[float] = 1.0e-3,
        early_stopping_use: bool = False,
        max_num_epochs: int = 300,
        early_stopping_split: float = 0.2,
        early_stopping_patience: int = 50,
        early_stopping_delta: float = 0.001,
        device: Optional[str] = "cpu",
        random_state: int = 42,
        alpha: int = 1.0,
        beta: int = 1.0,
        corruption_proba: float = 0.3,
        compute_reconstruction: bool = True,
        compute_reconstruction_masked: bool = False,
        predict_mask: bool = False,
        use_vime_mask: bool = False,
        use_noise_mask: bool = False,
        use_full_noise_mask: bool = False,
    ):
        super().__init__(
            repr_dim,
            hidden_n_layers,
            hidden_n_units_first,
            hidden_decrease_rate,
            dropout,
            activation,
            bias,
            num_epochs,
            batch_size,
            learning_rate,
            early_stopping_use,
            max_num_epochs,
            early_stopping_split,
            early_stopping_patience,
            early_stopping_delta,
            device,
            random_state,
        )

        # Other params
        self.p = corruption_proba
        self.alpha = alpha
        self.beta = beta
        self.compute_reconstruction = compute_reconstruction
        self.compute_reconstruction_masked = compute_reconstruction_masked
        self.predict_mask = predict_mask
        self.use_vime_mask = use_vime_mask
        self.use_noise_mask = use_noise_mask
        self.use_full_noise_mask = use_full_noise_mask

        # Logging
        if self.predict_mask:
            self.losses = {
                "train": {"full": [], "reconstruction": [], "mask": []},
                "eval": {"full": [], "reconstruction": [], "mask": []},
            }

            self.metrics = {
                "train": {"reconstruction": [], "mask": []},
                "eval": {"reconstruction": [], "mask": []},
            }
        else:
            self.losses = {
                "train": {"full": [], "reconstruction": []},
                "eval": {"full": [], "reconstruction": []},
            }

            self.metrics = {
                "train": {"reconstruction": []},
                "eval": {"reconstruction": []},
            }

        # Initialize architecture
        self.mask_decoder = None
        self.loss_functions = None

    def _init_models(self):
        super()._init_models()
        if self.predict_mask:
            self.mask_decoder = copy.deepcopy(self.decoder)
            self.mask_decoder.append(torch.nn.Sigmoid())
            self.mask_decoder.to(self.device)

        # Define objective functions
        self.loss_functions = {
            "reconstruction": torch.nn.MSELoss(),
            "mask": torch.nn.BCELoss(),
        }

    def mask_generator(self, x: torch.Tensor) -> torch.Tensor:
        """Generate mask vector.

        Args:
          x (torch.Tensor): input batch

        Returns:
          mask: binary mask matrix
        """
        tensor_p = torch.ones_like(x) * self.p
        mask = torch.bernoulli(tensor_p)
        return mask

    def pretext_generator(self, mask: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Generate corrupted samples.

        Args:
          mask (torch.Tensor): mask matrix
          x (torch.Tensor): input batch

        Returns:
          x_tilde: corrupted input
        """
        n, dim = x.shape

        if self.use_vime_mask:
            rng = np.random.default_rng()
            # Comment: this numpy function is much faster than the for loop of VIME
            # Need to detach from torch to use numpy function.
            x_bar = rng.permuted(x.detach().cpu(), axis=0)
            x_bar = torch.from_numpy(x_bar).to(self.device)
        elif self.use_noise_mask:
            # Gaussion noise
            x_noise = torch.randn(n, dim, device=self.device)
            x_bar = x + self.beta * x_noise
        elif self.use_full_noise_mask:
            # Full gaussion noise, to use with beta = 1.
            x_noise = torch.randn(n, dim, device=self.device)
            x_bar = self.beta * x_noise
        else:
            # Corrupt samples (Regular masking)
            x_bar = torch.zeros([n, dim])
        x_tilde = x * (1 - mask) + x_bar * mask

        return x_tilde

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Computes the reconstruction and predicts the mask applied to produce the
        corrupted sample.

        Args:
            x (torch.Tensor): input tensor

        Returns:
            rec_x (torch.Tensor): reconstruction of the input
            rec_mask (torch.Tensor): predicted mask
            mask (torch.Tensor): generated (groundtruth) mask
        """
        mask = self.mask_generator(x)
        x_tilde = self.pretext_generator(mask, x)

        z = self.encoder(x_tilde)

        rec_x = self.decoder(z)

        if self.predict_mask:
            rec_mask = self.mask_decoder(z)
        else:
            rec_mask = None

        return rec_x, rec_mask, mask

    def fit(
        self,
        X: pd.DataFrame,
        metrics_suffix: str = None,
        finetuning_rpz: bool = False,
        **kwargs,
    ) -> Self:
        """Fit the model according to the given training data and auxiliary labels.

        Args:
            X (pd.DataFrame): training data
            metrics_suffix (str): repeat/split information to be logged with the loss
            finetuning_rpz (bool): if true, don't re-initialise the model
            **kwargs: variable number of keyword arguments, to be compatible with
                the method from parent class.

        Returns:
            self
        """
        # pylint: disable=too-many-statements
        # pylint: disable=too-many-branches
        # pylint: disable=duplicate-code
        del kwargs

        # avoid printing "None" in MLflow
        metrics_suffix = metrics_suffix if metrics_suffix else ""

        if self.early_stopping_use:
            X_full = X.copy()
            X, _x_val = train_test_split(
                X, test_size=self.early_stopping_split, random_state=self.random_state
            )

        dataset = OmicsDataset(X.values)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
        )

        self.in_features = iter(dataloader).next().shape[1]

        # need to add "if not finetuning_rpz:" for the model to not be re-initialised
        # when doing finetuning_rpz, but too many if statements (13/12)
        self._init_models()

        self.train()

        params = (
            list(self.encoder.parameters())
            + (list(self.decoder.parameters()) if not self.predict_mask else [])
            + (list(self.mask_decoder.parameters()) if self.predict_mask else [])
        )

        optimizer = torch.optim.Adam(params, lr=self.learning_rate)
        early_stopping_best, early_stopping_patience_count = 0, 0
        for epoch in tqdm(range(self.num_epochs), total=self.num_epochs):
            # self.train() in the epoch loop because self.evaluate() calls self.eval()
            self.train()
            train_loss = {"full": [], "reconstruction": [], "mask": []}
            train_metrics = {"mask": []}
            for x in dataloader:
                x = x.to(self.device)
                # Forward pass
                x_hat, mask_hat, mask = self.forward(x)

                loss_rec = torch.zeros(1).to(self.device)
                if self.compute_reconstruction_masked:
                    loss_rec = self.loss_functions["reconstruction"](
                        x_hat * mask, x * mask
                    )
                else:
                    loss_rec = self.loss_functions["reconstruction"](x_hat, x)
                loss_mask = torch.zeros(1).to(self.device)
                if self.predict_mask:
                    loss_mask = self.alpha * self.loss_functions["mask"](mask_hat, mask)
                loss = (
                    loss_rec + loss_mask if self.compute_reconstruction else loss_mask
                )

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # Compute losses/metrics
                train_loss["full"].append(loss.detach().cpu().numpy())
                train_loss["reconstruction"].append(loss_rec.detach().cpu().numpy())
                if self.predict_mask:
                    train_loss["mask"].append(loss_mask.detach().cpu().numpy())
                    # Train metrics
                    y_pred = (mask_hat.cpu().detach().cpu().numpy() > 0.5).astype(float)
                    y_true = mask.cpu().detach().cpu().numpy()
                    train_acc = acc(y_true.flatten(), y_pred.flatten())
                    train_metrics["mask"].append(train_acc)
            # Early stopping
            if self.early_stopping_use:
                self.evaluate(_x_val)

                # Mlflow logging
                self.log_losses(epoch, train_loss, train_metrics, metrics_suffix)

                early_stopping_best = initialize_early_stopping(
                    self.losses["eval"]["reconstruction"], early_stopping_best
                )

                (
                    early_stopping_best,
                    early_stopping_patience_count,
                ) = update_early_stopping(
                    self.losses["eval"]["reconstruction"],
                    early_stopping_best,
                    self.early_stopping_delta,
                    early_stopping_patience_count,
                )

                if early_stopping_patience_count > self.early_stopping_patience:
                    logger.info(
                        f"AE training finished by early stopping at epoch {epoch + 1}"
                    )
                    self.early_stopping_epoch = (
                        epoch + 1  # to return it if someone wants to use this info
                    )
                    break

        else:  # it means I finished all my epochs
            logger.info(f"AE training finished with the max epoch number: {epoch + 1}")
            self.early_stopping_epoch = (
                self.num_epochs  # to return it if someone wants to use this info
            )

        if self.early_stopping_use:
            self.early_stopping_use = False
            self.num_epochs = self.early_stopping_epoch - self.early_stopping_patience
            self.fit(
                X_full,
                metrics_suffix=metrics_suffix + "retrain",
                finetuning_rpz=finetuning_rpz,
            )

        return self

    def log_losses(
        self, epoch: int, train_loss: dict, train_metrics: dict, metrics_suffix: str
    ):
        """Handles mlflow logging.

        Args:
            epoch (int): epoch index
            train_loss (Dict): Dictionnary containing train losses
            train_metrics (Dict): Dictionnary containing train metrics
            metrics_suffix (str): repeat/split information to be logged with the loss
        """
        # logging train losses & metrics
        self.losses["train"]["full"].append(np.mean(train_loss["full"]))
        self.losses["train"]["reconstruction"].append(
            np.mean(train_loss["reconstruction"])
        )
        self.metrics["train"]["reconstruction"].append(
            np.mean(train_loss["reconstruction"])
        )
        if self.predict_mask:
            self.losses["train"]["mask"].append(np.mean(train_loss["mask"]))
            self.metrics["train"]["mask"].append(np.mean(train_metrics["mask"]))

        # mlflow logging
        # Losses
        for task in self.losses["train"]:
            mlflow.log_metric(
                f"train-{task}-{metrics_suffix} Loss",
                self.losses["train"][task][-1],
                step=epoch,
            )
            if self.early_stopping_use:
                mlflow.log_metric(
                    f"eval-{task}-{metrics_suffix} Loss",
                    self.losses["eval"][task][-1],
                    step=epoch,
                )
        # Metrics
        for task in self.metrics["train"]:
            mlflow.log_metric(
                f"train-{task}-{metrics_suffix} Metric",
                self.metrics["train"][task][-1],
                step=epoch,
            )
            if self.early_stopping_use:
                mlflow.log_metric(
                    f"eval-{task}-{metrics_suffix} Metric",
                    self.metrics["eval"][task][-1],
                    step=epoch,
                )

    def evaluate(self, X: pd.DataFrame) -> Self:
        """Evaluation method.

        Args:
            X (pd.DataFrame): features matrix

        Returns:
            self
        """
        dataset = OmicsDataset(X.values)

        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
        )

        self.eval()

        eval_loss = {"full": [], "reconstruction": [], "mask": []}
        eval_metrics = {"mask": []}
        for x in dataloader:
            x = x.to(self.device)
            # Forward pass
            x_hat, mask_hat, mask = self.forward(x)

            loss_rec = torch.zeros(1).to(self.device)
            if self.compute_reconstruction_masked:
                loss_rec = self.loss_functions["reconstruction"](x_hat * mask, x * mask)
            else:
                loss_rec = self.loss_functions["reconstruction"](x_hat, x)
            loss_mask = torch.zeros(1).to(self.device)
            if self.predict_mask:
                loss_mask = self.alpha * self.loss_functions["mask"](mask_hat, mask)
            loss = loss_rec + loss_mask if self.compute_reconstruction else loss_mask

            # Compute Losses/Metrics
            eval_loss["full"].append(loss.detach().cpu().numpy())
            eval_loss["reconstruction"].append(loss_rec.detach().cpu().numpy())
            if self.predict_mask:
                # eval metrics
                eval_loss["mask"].append(loss_mask.detach().cpu().numpy())
                y_pred = (mask_hat.detach().cpu().numpy() > 1 - self.p).astype(float)
                y_true = mask.cpu().detach().cpu().numpy()
                eval_acc = np.mean(
                    [acc(y_true, y_pred) for i in range(y_true.shape[0])]
                )
                eval_metrics["mask"].append(eval_acc)

        self.losses["eval"]["full"].append(np.mean(eval_loss["full"]))
        self.losses["eval"]["reconstruction"].append(
            np.mean(eval_loss["reconstruction"])
        )
        self.metrics["eval"]["reconstruction"].append(
            np.mean(eval_loss["reconstruction"])
        )
        if self.predict_mask:
            self.losses["eval"]["mask"].append(np.mean(eval_loss["mask"]))
            self.metrics["eval"]["mask"].append(np.mean(eval_metrics["mask"]))

        return self
