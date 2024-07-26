"""Class for the MLP and encoder + MLP prediction models."""

import copy
from typing import Any, Optional, Union

import numpy as np
import pandas as pd
import torch
from loguru import logger
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

from omics_rpz.utils import compute_metric, convert_to_array, update_early_stopping


class DatasetFromArrays(Dataset):
    """Torch dataset created from numpy arrays
    Args:
        data (np array): feature matrix (n_samples x n_features)

        labels (np array): labels (n_samples x n_labels)
    """

    def __init__(self, data: np.ndarray, labels: np.ndarray):
        assert data.shape[0] == labels.shape[0]
        if labels.ndim == 1:
            labels = labels[:, None]
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return torch.from_numpy(self.data[idx].astype(np.float32)), torch.from_numpy(
            self.labels[idx].astype(np.float32)
        )


class MLP(torch.nn.Sequential):
    """
    MLP Module

    Parameters
    ----------
    in_features: int
    out_features: int
    hidden: Optional[List[int]] = None
    dropout: Optional[List[float]] = None,
    activation: Optional[torch.nn.Module] = torch.nn.Sigmoid
    bias: bool = True
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden: Optional[List[int]] = None,
        dropout: Optional[List[float]] = None,
        activation: Optional[torch.nn.Module] = torch.nn.Sigmoid(),
        bias: bool = True,
    ):

        if dropout is not None:
            if hidden is not None:
                assert len(hidden) == len(
                    dropout
                ), "hidden and dropout must have the same length"
            else:
                raise ValueError(
                    "hidden must have a value and have the same length as dropout if dropout is given."
                )

        d_model = in_features
        layers = []

        if hidden is not None:
            for i, h in enumerate(hidden):
                seq = [torch.nn.Linear(d_model, h, bias=bias)]
                d_model = h

                if activation is not None:
                    seq.append(activation)

                if dropout is not None:
                    seq.append(torch.nn.Dropout(dropout[i]))

                layers.append(torch.nn.Sequential(*seq))

        layers.append(torch.nn.Linear(d_model, out_features))

        super(MLP, self).__init__(*layers)


class MLPWithEncoder(torch.nn.Module):
    """Model resulting from taking an encoder and attaching a MLP to it. The MLP can
    take extra variables as inputs on top of the encoder's outputs.

    Parameters
    ----------
    encoder: torch model, the encoder whose outputs will be fed as input to the MLP

    in_features: integer, dimension of the encoder's output

    out_features: integer, dimension of the MLP's output

    dropout (float): probability to randomly set to zero some units in the hidden layers

    end_dimensions_to_not_transform: integer, number of variables the MLP will take as
        input on top of the encoder's output

    random_state: int, by default set to 0, random seed for the model.
    """

    def __init__(
        self,
        encoder: torch.nn.Module,
        in_features: int,
        out_features: int,
        hidden: list[int],
        activation: Optional[torch.nn.Module] = torch.nn.Sigmoid(),
        dropout: float = None,
        # the downstream MLP takes the encoder's output and extra inputs if necessary
        # (cf gene fingerprints representations in the gene essentiality task)
        end_dimensions_to_not_transform: int = 0,
        final_activation: Optional[torch.nn.Module] = None,
        random_state: int = 0,
    ) -> None:
        super().__init__()
        self.random_state = random_state
        torch.manual_seed(self.random_state)
        if dropout is None:
            dropout = 0
        self.final_activation = final_activation
        self.encoder = encoder
        self.end_dimensions_to_not_transform = end_dimensions_to_not_transform
        self.mlp = MLP(
            in_features=in_features + end_dimensions_to_not_transform,
            out_features=out_features,
            hidden=hidden,
            dropout=[dropout] * len(hidden),  # in classic_algos, dropout is list[float]
            activation=activation,
        )
        if self.final_activation:
            self.mlp.append(self.final_activation)

    def forward(self, x):
        """Forward function of the MLP created from an encoder
        Args:
            x (torch.tensor): input
        Returns:
            torch.tensor: output
        """
        if self.encoder:
            if self.end_dimensions_to_not_transform != 0:
                # input x is split between x_to_be_transformed_by_ae that feeds the
                # encoder and x_extra we then concatenate the encoder's output
                # (x_in_enc) and the extra variables (x_extra) if
                # end_dimensions_to_not_transform > 0 to feed the downstream MLP.
                x_to_be_transformed_by_ae = x[
                    :, : -self.end_dimensions_to_not_transform
                ]
                x_in_enc = self.encoder(x_to_be_transformed_by_ae)
                x_extra = x[:, -self.end_dimensions_to_not_transform :]
                x = torch.cat((x_in_enc, x_extra), dim=1)
            else:
                x = self.encoder(x)
        return self.mlp(x)


class MLPPrediction:
    """Prediction model: simple MLP or encoder+MLP MLPWithEncoder with training
    parameters and functions, prediction function, dataset and dataloader creation
    function (DatasetFromArrays)

    Parameters
    ----------

    loss_constructor (str): loss function used in the training of the model,
    to be modified manually

    auto_encoder: torch model, the auto-encoder whose encoder part will be extracted to
    instantiate the model MLPWithEncoder

    dropout (float): probability to randomly set to zero some units in the hidden layers

    end_dimensions_to_not_transform: integer, number of variables the MLP will take as
    input on top of the encoder's output

    early_stopping_use: bool, Whether to split the data in train val during
             training of the mlp and perform early stopping with the val set
    max_num_epochs: int, Maximum number of iterations in the case of early
             stopping. Otherwise num_epochs is used.
    early_stopping_split: float, the train/val split proportion for the early
             stopping val to use
    early_stopping_patience: int, if the model doesn't improve for n_patience by
             more than early_stopping_delta, stop the training
    early_stopping_delta: float, if the model doesn't improve for n_patience by
             more than early_stopping_delta, stop the training
    metric: Any, the metric to use to stop training with early stopping (eg C-index)

    random_state: int, by default set to 0, random seed for the model.

    Raises
    ------
    AssertionError
        if early stopping is used and patience is larger than max_num_epochs.

    Attributes
    ----------

    model: torch model, the model of type MLPWithEncoder
    early_stopping_epoch: int, the epoch found by early stopping.
    """

    def __init__(
        self,
        loss_fn: Optional[torch.nn.Module] = torch.nn.MSELoss(),
        mlp_hidden: list[int] = None,
        activation: Optional[torch.nn.Module] = torch.nn.Sigmoid(),
        final_activation: Optional[torch.nn.Module] = None,
        dropout: float = 0,
        batch_size: int = 32,
        training_drop_last: bool = False,
        num_epochs: int = 50,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        device: str = "cuda",
        auto_encoder=None,
        end_dimensions_to_not_transform: int = 0,
        early_stopping_use: bool = False,
        max_num_epochs: int = 300,
        early_stopping_split: float = 0.2,
        early_stopping_patience: int = 50,
        early_stopping_delta: float = 0.001,
        metric: Any = None,
        random_state: int = 0,
    ):
        if early_stopping_use and early_stopping_patience >= max_num_epochs:
            raise AssertionError

        # super().__init__()
        if mlp_hidden is None:
            mlp_hidden = [256, 128]
        self.loss_fn = loss_fn
        self.mlp_hidden = mlp_hidden
        self.activation = activation
        self.final_activation = final_activation
        self.dropout = dropout
        self.batch_size = batch_size
        self.training_drop_last = training_drop_last
        self.num_epochs = max_num_epochs if early_stopping_use else num_epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.device = device
        self.auto_encoder = auto_encoder
        self.end_dimensions_to_not_transform = end_dimensions_to_not_transform
        self.early_stopping_use = early_stopping_use
        self.early_stopping_split = early_stopping_split
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_delta = early_stopping_delta
        self.early_stopping_epoch = None
        self.metric = metric
        self.random_state = random_state
        self.optimizer = None
        self.model = None

    def _build_dataloader(self, X, y, training: bool):
        dataset = DatasetFromArrays(data=X, labels=y)
        shuffle = training
        drop_last = training and self.training_drop_last
        return DataLoader(
            dataset=dataset,
            shuffle=shuffle,
            num_workers=0,
            batch_size=self.batch_size,
            pin_memory=False,
            drop_last=drop_last,
        )

    def run_epoch(self, X: np.ndarray, y: np.ndarray, training: bool):
        """Training / inference loop for 1 epoch
        Args:
            X (np.array): features
            y (np.array): labels
            training (bool): training / inference mode
        Returns:
            list: predictions for each sample of that epoch
            list: losses for each batch of that epoch
        """
        if training:
            self.model.train()
        else:
            self.model.eval()

        epoch_predictions = []
        epoch_losses = []
        with torch.inference_mode(mode=not training):
            dataloader = self._build_dataloader(X, y, training=training)
            for batch in dataloader:
                features, labels = batch
                features = features.to(self.device)
                labels = labels.to(self.device)
                predictions = self.model(features)
                epoch_predictions.append(predictions.detach().cpu().numpy())
                loss = self.loss_fn(predictions, labels)
                if training:
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                epoch_losses.append(loss.detach().cpu().numpy())

        concatenated_predictions = np.concatenate(epoch_predictions, axis=0)
        epoch_losses_array = np.array(epoch_losses)
        return concatenated_predictions, epoch_losses_array

    def fit(
        self, X: Union[np.ndarray, pd.DataFrame], y: Union[np.ndarray, pd.DataFrame]
    ):
        """Training loop for the MLPWithEncoder model
        Args:
            X (np.array or pd.DataFrame): features
            y (np.array or pd.DataFrame): labels
        Returns:
            self: with trained model
        """
        if self.early_stopping_use:
            X_full, y_full = X.copy(), y.copy()  # to be able to re-train with full data
            X, _x_val, y, _y_val = train_test_split(
                X,
                y,
                test_size=self.early_stopping_split,
                random_state=self.random_state,
                stratify=y >= 0,
            )
            _x_val, _y_val = convert_to_array(_x_val), convert_to_array(_y_val)

        X, y = convert_to_array(X), convert_to_array(y)

        in_features = X.shape[1]
        out_features = y.shape[1] if y.ndim > 1 else 1

        encoder = None
        if self.auto_encoder:
            # we concatenate the pre-trained encoder with the MLP for a common training
            in_features = self.auto_encoder.repr_dim
            if self.early_stopping_use:
                # then save the pre-trained encoder to re-train it with the right
                # weights after early stopping is finished
                encoder = copy.deepcopy(self.auto_encoder.encoder)
            else:
                encoder = self.auto_encoder.encoder

        self.model = MLPWithEncoder(
            encoder=encoder,
            in_features=in_features,
            out_features=out_features,
            hidden=self.mlp_hidden,
            activation=self.activation,
            dropout=self.dropout,
            end_dimensions_to_not_transform=self.end_dimensions_to_not_transform,
            final_activation=self.final_activation,
            random_state=self.random_state,
        )
        self.model.to(self.device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        val_loss, val_metric = [], []
        early_stopping_best, early_stopping_patience_count = 0, 0
        # Training epoch
        for epoch in range(self.num_epochs):
            _, losses = self.run_epoch(X, y, training=True)

            if (epoch + 1) % 10 == 0:
                logger.info(
                    f"[MLP prediction] epoch={epoch+1},training loss={np.mean(losses)}"
                )
            if self.early_stopping_use:
                _val_preds, _val_losses = self.run_epoch(_x_val, _y_val, training=False)
                val_loss.append(np.mean(_val_losses))
                val_metric = compute_metric(self.metric, val_metric, _y_val, _val_preds)

                (
                    early_stopping_best,
                    early_stopping_patience_count,
                ) = update_early_stopping(
                    val_metric,
                    early_stopping_best,
                    self.early_stopping_delta,
                    early_stopping_patience_count,
                    use_metric=True,
                )

                if early_stopping_patience_count > self.early_stopping_patience:
                    logger.info(
                        f"MLP training finished by early stopping at epoch {epoch + 1}"
                    )
                    self.early_stopping_epoch = (
                        epoch + 1  # to return it if someone wants to use this info
                    )
                    break

        else:  # it means I finished all my epochs
            logger.info(f"MLP training finished with the max epoch number: {epoch + 1}")
            self.early_stopping_epoch = (
                self.num_epochs  # to return it if someone wants to use this info
            )

        if self.early_stopping_use:
            self.early_stopping_use = False
            self.num_epochs = self.early_stopping_epoch - self.early_stopping_patience
            self.fit(X_full, y_full)

        return self

    def predict(
        self, X: Union[np.ndarray, pd.DataFrame], y: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Inference function
        Args:
            X (np.array or pd.DataFrame): features
            y (np.array): labels
        Returns:
            output (list): model(X)
        """
        if isinstance(X, pd.DataFrame):
            X = X.values

        return_losses = True
        if y is None:
            return_losses = False
            y = np.nan * np.ones(X.shape[0])
        predictions, losses = self.run_epoch(X, y, training=False)
        if return_losses:
            return predictions, losses
        return predictions
