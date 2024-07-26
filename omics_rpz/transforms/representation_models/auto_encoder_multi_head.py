"""Class implementing the multi-head autoencoder representation model.

This module implements the methods for class AutoEncoderMultiHead which override
methods from the parent class AutoEncoder.
"""

from typing import Optional

import mlflow
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from loguru import logger
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing_extensions import Self

from omics_rpz.constants import ESSENTIALITY_LABEL, PATHWAY_ACTIVATION_LABEL
from omics_rpz.data import OmicsDataset

try:
    from omics_rpz.losses import CoxLoss
except ImportError:
    pass
from omics_rpz.transforms.representation_models.auto_encoder import AutoEncoder
from omics_rpz.utils import initialize_early_stopping, update_early_stopping


class AutoEncoderMultiHead(AutoEncoder):
    """Representation model: Multi-Head autoencoder.

    This representation model is composed of a regular autoencoder, plus
    multiple supervised learning tasks. Each supervised learning task uses
    the latent codes (autoencoder bottleneck) as inputs, and the outputs are
    defined via task.multi_heads.

    Parameters:
        multi_heads (list[str] = None): Labels used by each head's auxiliary task
        betas (list[float] = None): Weights for the loss function terms (rec. + heads)
            You can also pass an int when you want to the fix a weight of 1 for the
            reconstruction loss and beta for all the other heads.
        aux_losses (list[str] = None): Strings denoting loss functions for each head
        aux_hidden (list[list[int]]): Number of units in each layer of each head
        aux_dropout (list[float] = None): Dropout probabilities in each head
        ce_quantile (float = 0.3): Quantile to define least frequent classes for CE loss
        Other parameters: see docstring from the parent (AutoEncoder) class

    Attributes:
        heads_nn (list[torch.nn.Sequential]): Neural network models for each head
        in_features (int): number of features from the training dataset
        last_layer_units (int): number of units in the last layer of each aux head
        train_loss(list): list storing the train loss on each epoch
        eval_loss(list): list storing the eval loss on each epoch
    """

    def __init__(
        self,
        repr_dim: int,
        hidden_n_layers: int = 2,
        hidden_n_units_first: int = 512,
        hidden_decrease_rate: float = 0.5,
        dropout: float = None,
        activation: Optional[torch.nn.Module] = torch.nn.ReLU(),
        bias: bool = True,
        num_epochs: Optional[int] = 10,
        batch_size: Optional[int] = 16,
        learning_rate: Optional[float] = 1.0e-3,
        early_stopping_use: bool = False,
        max_num_epochs: int = 300,
        early_stopping_split: float = 0.2,
        early_stopping_patience: int = 50,
        early_stopping_delta: float = 0.001,
        device: Optional[str] = "cpu",
        random_state: int = 42,
        multi_heads: list[str] = None,
        betas: list[float] = None,
        aux_losses: list[str] = None,  # BCE, MSE, CE, COX
        aux_hidden: list[list[int]] = None,
        aux_dropout: list[float] = None,
        ce_quantile: float = 0.3,
    ):  # pylint: disable=duplicate-code
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

        self.multi_heads = multi_heads
        self.betas = betas
        self.aux_losses = aux_losses
        self.aux_hidden = aux_hidden
        self.aux_dropout = aux_dropout
        self.ce_quantile = ce_quantile
        self.heads_nn = []  # initialized in _init_models method
        self.in_features = None  # initialized in fit method
        self.last_layer_units = []  # initialized in fit method

    def _init_models(self):
        """Builds the neural networks for each head.

        This method also calls _init_models() from the parent class AutoEncoder,
        which builds the encoder/decoder neural networks.

        Raises:
            ValueError if the configuration of the auxiliary heads is invalid
        """
        super()._init_models()

        # Initialize beta by default when only one additional head
        if isinstance(self.betas, (float, int)):
            self.betas = np.append([1], [self.betas] * len(self.multi_heads))

        # in self.betas[], first index is for reconstruction, others for heads
        if len(self.aux_losses) != len(self.betas) - 1:
            raise ValueError(
                f"Len losses {len(self.aux_losses)} Len betas {len(self.betas)}"
            )

        if len(self.aux_losses) != len(self.aux_hidden):
            raise ValueError(
                f"Len losses {len(self.aux_losses)} Len hidden {len(self.aux_hidden)}"
            )

        if self.aux_dropout is not None and len(self.aux_losses) != len(
            self.aux_dropout
        ):
            raise ValueError(
                f"Len losses {len(self.aux_losses)} Len dropout {len(self.aux_dropout)}"
            )

        for i, loss_str in enumerate(self.aux_losses):
            aux_head_layers = []

            # first layer, starting from autoencoder bottleneck
            aux_head_layers.append(
                torch.nn.Linear(self.repr_dim, self.aux_hidden[i][0], bias=self.bias)
            )
            aux_head_layers.append(self.activation)
            if self.aux_dropout is not None:
                aux_head_layers.append(torch.nn.Dropout(self.aux_dropout[i]))

            # middle layers
            for j in range(len(self.aux_hidden[i]) - 1):
                aux_head_layers.append(
                    torch.nn.Linear(
                        self.aux_hidden[i][j], self.aux_hidden[i][j + 1], bias=self.bias
                    )
                )
                aux_head_layers.append(self.activation)
                if self.aux_dropout is not None:
                    aux_head_layers.append(torch.nn.Dropout(self.aux_dropout[i]))

            # Last layer's output is one hidden unit for regression and binary
            # classification, or the number of classes for multi-class classification
            aux_head_layers.append(
                torch.nn.Linear(
                    self.aux_hidden[i][-1], self.last_layer_units[i], bias=self.bias
                )
            )

            # Last layer's activation function also depends on the auxiliary task
            if loss_str in {'BCE', 'COX'}:
                aux_head_layers.append(torch.nn.Sigmoid())

            self.heads_nn.append(torch.nn.Sequential(*aux_head_layers))

        for head in self.heads_nn:
            head.to(self.device)

    def forward(self, x: torch.Tensor) -> dict:
        """Computes the reconstruction and the outputs of each auxiliary head.

        Args:
            x (torch.Tensor): input tensor

        Returns:
            dict containing output tensor (x_dec) and outputs of each head (pred_heads)
        """
        x_enc = self.encoder(x)
        x_dec = self.decoder(x_enc)
        pred_heads = []
        for head in self.heads_nn:
            pred_heads.append(head(x_enc))

        return {'x_dec': x_dec, 'pred_heads': torch.cat(pred_heads, dim=1)}

    def fit(
        self,
        X,
        metrics_suffix: str = None,
        finetuning_rpz: bool = False,
        df_labels: pd.DataFrame = None,
        **kwargs,
    ) -> Self:
        """Fit the model according to the given training data and auxiliary labels.

        Args:
            X (pd.DataFrame): training data
            metrics_suffix (str): repeat/split information to be logged with the loss
            finetuning_rpz (bool): if true, don't re-initialise the model
            df_labels (pd.DataFrame): contains all df columns except RNA-Seq data,
                therefore including the auxiliary labels for each head
            **kwargs: variable number of keyword arguments, to be compatible with
                the method from parent class.

        Raises:
            ValueError: when df_labels is not a valid pandas DataFrame

        Returns:
            self
        """
        del kwargs
        if not isinstance(df_labels, pd.DataFrame):
            raise ValueError('df_labels must be a valid pandas Dataframe')

        aux_label_columns = []
        for head_str in self.multi_heads:
            if head_str in (ESSENTIALITY_LABEL, PATHWAY_ACTIVATION_LABEL):
                # multiple-column targets, all starting with same prefix head_str
                aux_label_columns.extend(
                    df_labels.filter(like=head_str).columns.tolist()
                )
            else:
                # one-column targets (general case)
                aux_label_columns.append(head_str)
        df_aux_labels = df_labels[aux_label_columns]
        n_heads = len(self.multi_heads)

        # do not preprocess labels again if retraining for early stopping
        if metrics_suffix is None or 'Retrain' not in metrics_suffix:
            self.preprocess_aux_labels(df_aux_labels, n_heads)

        # avoid printing "None" in MLflow
        metrics_suffix = metrics_suffix if metrics_suffix else ""

        # reset if early_stopping
        self.train_loss, self.eval_loss = [], []

        if self.early_stopping_use:
            X_full = X.copy()
            df_aux_labels_full = df_aux_labels.copy()
            X, X_val, df_aux_labels, df_aux_labels_val = train_test_split(
                X,
                df_aux_labels,
                test_size=self.early_stopping_split,
                random_state=self.random_state,
            )
            dataset_val = OmicsDataset(
                pd.concat((X_val, df_aux_labels_val), axis=1).values
            )
            dataloader_val = DataLoader(
                dataset_val, batch_size=self.batch_size, shuffle=True
            )

        dataset = OmicsDataset(pd.concat((X, df_aux_labels), axis=1).values)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        sample = iter(dataloader).next()

        n_aux_labels = df_aux_labels.shape[1]
        self.in_features = sample.shape[1] - n_aux_labels

        if not finetuning_rpz:
            self._init_models()

        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

        early_stopping_best, early_stopping_patience_count = 0, 0
        for epoch in tqdm(range(self.num_epochs), total=self.num_epochs):
            self.train_epoch(dataloader, n_aux_labels, metrics_suffix, epoch, optimizer)

            if self.early_stopping_use:
                self.evaluate_epoch(dataloader_val, n_aux_labels, metrics_suffix, epoch)

                # Disable "similar lines in 2 files" warning, since these two functions
                # were created exactly to reduce duplicated code
                # pylint: disable=R0801
                early_stopping_best = initialize_early_stopping(
                    self.eval_loss, early_stopping_best
                )

                (
                    early_stopping_best,
                    early_stopping_patience_count,
                ) = update_early_stopping(
                    self.eval_loss,
                    early_stopping_best,
                    self.early_stopping_delta,
                    early_stopping_patience_count,
                )

                if early_stopping_patience_count > self.early_stopping_patience:
                    logger.info(
                        f"MH-AE train finished by early stopping at epoch {epoch + 1}"
                    )
                    self.early_stopping_epoch = epoch + 1
                    break

        else:  # it means I finished all my epochs
            logger.info(f"MH-AE train finished with the max epoch number: {epoch + 1}")
            self.early_stopping_epoch = self.num_epochs

        if self.early_stopping_use:
            self.early_stopping_use = False
            self.num_epochs = self.early_stopping_epoch - self.early_stopping_patience
            self.fit(
                X_full,
                metrics_suffix=metrics_suffix + " Retrain",
                finetuning_rpz=finetuning_rpz,
                df_labels=df_aux_labels_full,
            )

        return self

    def train_epoch(
        self,
        dataloader: torch.utils.data.DataLoader,
        n_aux_labels: int,
        metrics_suffix: str,
        epoch: int,
        optimizer: torch.optim.Adam,
    ):
        """Train the model for one epoch.

        Args:
            dataloader (torch.utils.data.DataLoader): training dataloader object
            n_aux_labels (int): number of auxiliary labels
            metrics_suffix (str): repeat/split information to be logged with the loss
            epoch (int): the current epoch index
            optimizer (torch.optim.Adam): optimizer object
        """
        self.train()

        # Dict to store train losses (heads, reconstruction, and total) for this epoch
        train_losses = self.init_losses_dict()

        for batch in dataloader:
            loss = self.compute_batch_loss(batch, n_aux_labels, train_losses)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        self.train_loss.append(np.mean(train_losses['total']))

        self.log_losses(metrics_suffix, train_losses, epoch)

    def evaluate_epoch(
        self,
        dataloader: torch.utils.data.DataLoader,
        n_aux_labels: int,
        metrics_suffix: str,
        epoch: int,
    ):
        """Evaluate the model performance on the validation set for one epoch.

        Update self.eval_loss with the mean loss across batches

        Args:
            dataloader (torch.utils.data.DataLoader): eval dataloader object
            n_aux_labels (int): number of auxiliary labels
            metrics_suffix (str): repeat/split information to be logged with the loss
            epoch (int): the current epoch index
        """
        self.eval()

        # Dict to store eval losses (heads, reconstruction, and total) for this epoch
        eval_losses = self.init_losses_dict()

        for batch in dataloader:
            self.compute_batch_loss(batch, n_aux_labels, eval_losses)

        self.eval_loss.append(np.mean(eval_losses['total']))

        self.log_losses(metrics_suffix + " Eval", eval_losses, epoch)

    def compute_batch_loss(self, batch: torch.tensor, n_aux_labels: int, losses: dict):
        """Compute total loss for a single batch.

        Args:
            batch (torch.tensor): batch of data
            n_aux_labels (int): number of auxiliary labels
            losses (dict): dictionary containing all loss terms

        Returns:
            torch.tensor: computed total loss for this batch
        """
        x = batch[:, :-n_aux_labels].to(self.device)
        y_aux = batch[:, -n_aux_labels:].to(self.device).reshape(-1, n_aux_labels)

        fwd_return = self.forward(x)
        x_hat = fwd_return['x_dec']
        y_aux_hat = fwd_return['pred_heads']

        loss = self.compute_aux_heads_losses(losses, y_aux_hat, y_aux)

        reconstruction_loss = self.betas[0] * self.criterion(x_hat, x)
        losses['reconstruction'].append(reconstruction_loss.detach().cpu().numpy())

        loss += reconstruction_loss
        losses['total'].append(loss.detach().cpu().numpy())

        return loss

    def fit_transform(
        self,
        X: pd.DataFrame,
        metrics_suffix: str,
        df_labels: pd.DataFrame = None,
        **kwargs,
    ) -> np.ndarray:
        """Fit the model, then encode the data using the fitted model.

        Args:
            X (pd.DataFrame): training data
            metrics_suffix (str): repeat/split information to be logged with the loss
            df_labels (pd.DataFrame): contains all df columns except RNA-Seq data,
                therefore including the auxiliary labels for each head
            **kwargs: variable number of keyword arguments, to be compatible with
                the method from parent class.

        Returns:
            np.ndarray: encoded data
        """
        _ = self.fit(X, metrics_suffix, df_labels=df_labels, **kwargs)
        return self.transform(X)

    def preprocess_aux_labels(self, df_aux_labels: pd.DataFrame, n_heads: int):
        """Auxiliary labels arrive without preprocessing, so preprocess them here.

        Also infer how many units are needed on the last layer of each aux head.

        Args:
            df_aux_labels (pd.DataFrame): df with the auxiliary labels for each head
            n_heads (int): number of auxiliary heads
        """
        target_indices = 0
        for i in range(n_heads):
            if self.aux_losses[i] == 'MSE':
                if self.multi_heads[i] in (
                    ESSENTIALITY_LABEL,
                    PATHWAY_ACTIVATION_LABEL,
                ):
                    # multiple-column targets
                    n_targets = df_aux_labels.filter(like=self.multi_heads[i]).shape[1]
                    self.last_layer_units.append(n_targets)
                    target_indices += n_targets
                else:
                    # one-column targets (general case)
                    df_aux_labels.iloc[
                        :, target_indices
                    ] = MinMaxScaler().fit_transform(
                        df_aux_labels.iloc[:, target_indices].to_numpy().reshape(-1, 1)
                    )
                    self.last_layer_units.append(1)
                    target_indices += 1
            elif self.aux_losses[i] == 'BCE':
                self.last_layer_units.append(1)
                target_indices += 1
            elif self.aux_losses[i] == 'COX':
                self.last_layer_units.append(1)
                target_indices += 1
            elif self.aux_losses[i] == 'CE':
                # Group together low-frequency classes.
                # For example, cancer_type goes from 33 to 24 classes using q=0.3 below
                class_freq = (
                    df_aux_labels.iloc[:, target_indices].value_counts()
                ) / df_aux_labels.shape[0]
                less_freq_classes = class_freq[
                    class_freq <= class_freq.quantile(q=self.ce_quantile)
                ]
                df_aux_labels.iloc[
                    df_aux_labels.iloc[:, target_indices].isin(
                        less_freq_classes.index.tolist()
                    ),
                    target_indices,
                ] = "other"
                self.last_layer_units.append(
                    len(df_aux_labels.iloc[:, target_indices].unique())
                )

                df_aux_labels.iloc[:, target_indices] = OrdinalEncoder().fit_transform(
                    df_aux_labels.iloc[:, target_indices].to_numpy().reshape(-1, 1)
                )
                target_indices += 1

    def compute_aux_heads_losses(
        self, train_losses: dict, y_aux_hat: torch.Tensor, y_aux: torch.Tensor
    ) -> torch.Tensor:
        """Compute the losses of all auxiliary heads.

        This function supports single-column targets with single-column predictions
        (e.g. sex, age, OS), single-column targets with multiple-column predictions
        (e.g. cancer_type), and multiple-column targets with multiple-column predictions
        (e.g. essentiality, pathway activations).

        Args:
            train_losses (dict): dictionary containing the different losses
                (reconstruction, each head, total)
            y_aux_hat (torch.Tensor): predictions of all auxiliary heads for this batch
            y_aux (torch.Tensor): labels used to train all auxiliary heads in this batch

        Raises:
            ValueError: when a string in self.aux_losses is not recognized

        Returns:
            loss (torch.Tensor): sum of the losses for all auxiliary heads in this batch
        """
        loss_mappings = {
            'BCE': F.binary_cross_entropy,
            'MSE': F.mse_loss,
            'CE': F.cross_entropy,
            'COX': CoxLoss().forward,
        }

        loss = 0

        target_col = 0
        for i, loss_str in enumerate(self.aux_losses):
            if loss_str not in loss_mappings:
                raise ValueError(f'{loss_str} not in know losses')

            f_head_loss = loss_mappings[loss_str]

            # Compute the column indices corresponding to this head in y_aux_hat,
            # by summing the number of units in each head to know which indices
            # to use. For instance, if we have 3 heads with 1, 3, and 1 units:
            #  - pred_cols will be [0] for head i=0;
            #  - pred_cols will be [1, 2, 3] for head i=1;
            #  - pred_cols will be [3] for head i=2.
            pred_first_col = sum(self.last_layer_units[:i])
            pred_last_col = sum(self.last_layer_units[: i + 1])
            pred_cols = list(range(pred_first_col, pred_last_col))

            # select prediction (one or multiple columns)
            # reshape is needed for batches with just one example
            prediction = y_aux_hat[:, pred_cols].reshape(-1, len(pred_cols))

            # select target
            # exception: multiple-column targets, multiple-column predictions
            if self.multi_heads[i] in (ESSENTIALITY_LABEL, PATHWAY_ACTIVATION_LABEL):
                target = y_aux[:, target_col : target_col + len(pred_cols)].reshape(
                    -1, len(pred_cols)
                )
                target_col += len(pred_cols)
            # general case: one-column targets
            # F.cross_entropy needs the targets to be of type long
            elif loss_str == 'CE':
                target = y_aux[:, target_col].reshape(-1).long()
                target_col += 1
            else:
                target = y_aux[:, target_col].reshape(-1, 1)
                target_col += 1

            # finally, compute the loss corresponging to this head
            head_loss = self.betas[i + 1] * f_head_loss(prediction, target)

            loss += head_loss

            train_losses[i].append(head_loss.detach().cpu().numpy())

        return loss

    def log_losses(self, metrics_suffix: str, losses: dict, epoch: int):
        """Log all losses values (each head, reconstruction total) with MLFlow.

        Args:
            metrics_suffix (str): repeat/split information to be logged with the loss
            losses (dict): dictionary containing the different losses
                (reconstruction, each head, total)
            epoch (int): index of the current epoch (0 to self.num_epochs - 1)
        """
        for l_str in ('total', 'reconstruction'):
            mlflow.log_metric(
                f"Loss {l_str} {metrics_suffix}",
                np.mean(losses[l_str]),
                step=epoch,
            )
        for i, _ in enumerate(self.aux_losses):
            mlflow.log_metric(
                f"Loss head{i} {metrics_suffix}",
                np.mean(losses[i]),
                step=epoch,
            )

    def init_losses_dict(self):
        """Create an empty dictionary to store all loss terms for an epoch.

        Returns:
            Dict: the empty dictionary with the proper format
        """
        loss_dict = {}
        loss_dict['total'] = []
        loss_dict['reconstruction'] = []
        for i, _ in enumerate(self.aux_losses):
            loss_dict[i] = []

        return loss_dict
