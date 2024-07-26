"""AutoEncoder implementation from torch."""

import copy
from typing import Optional

import mlflow
import numpy as np
import pandas as pd
import torch
from loguru import logger
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from omics_rpz.data import OmicsDataset
from omics_rpz.utils import initialize_early_stopping, update_early_stopping


class AutoEncoder(torch.nn.Module):
    """Representation model: Autoencoder.

    Parameters:
         repr_dim: int, Size of the representation dimension.
         hidden_n_layers: int = 2, number of hidden layers
         hidden_n_units_first: int = 512, number of units of the first hidden layer
         hidden_decrease_rate: float = 0.5, decrease rate of the number of units / layer
         dropout: float, Probability p corresponding to all hidden layers. During
            training, randomly zeroes (with p probability) some units of the hidden
            layers.
         activation: Optional[torch.nn.Module] = torch.nn.ReLU, Activation function
             for the hidden layers.
         bias: Optional[bool] = True, If set to False, the layers will not learn an
             additive bias.
         num_epochs: Optional[int] = 10, Maximum number of iterations. The solver
             iterates until convergence (determined by "tol") or this number of
             iterations. For stochastic solvers ("sgd", "adam"), note that this
             determines the number of epochs (how many times each data point will be
             used), not the number of gradient steps.
         batch_size: Optional[int] = 16, Size of minibatches for stochastic optimizers.
             If the solver is "lbfgs", the classifier will not use minibatch.
         learning_rate: Optional[float] = 1.0e-3, Learning rate controling the step-size
             in updating the weights. Only used when solver="sgd" or "adam".
         early_stopping_use: bool, Whether to split the data in train val during
             training of the autoencoder and perform early stopping with the val set
         max_num_epochs: int, Maximum number of iterations in the case of early
             stopping. Otherwise num_epochs is used.
         early_stopping_split: float, the train/val split proportion for the early
             stopping val to use
         early_stopping_patience: int, if the model doesn't improve for n_patience by
             more than early_stopping_delta, stop the training
         early_stopping_delta: float, if the model doesn't improve for n_patience by
             more than early_stopping_delta, stop the training
         device: Optional[str] = "cpu", Whether to use cpu or CUDA to run computations.
         criterion: torch.nn, Loss to use for the autoencoder reconstruction task
             training, defaulting to "torch.nn.MSELoss()". The reconstruction loss will
             be computed by doing the average of this loss (criterion) on the different
             batches.
         optimizer: torch.optim, by default Adam, optimizer to use in gradient descent.
         random_state: int, by default set to 42, random seed for the model.

    Raises:
    AssertionError
        if early stopping is used and patience is larger than max_num_epochs.

    Attributes:
        hidden: List[int], number of units of each intermediate hidden layer.
        in_features: int, Number of input features from the training dataset.
        train_loss: List[float], list of mean train loss computed on the different data
            batches using the criterion.
        eval_loss: List[float], list of mean eval loss computed on the different data
            batches using the criterion.
        encoder: torch.nn.Sequential, Encoder model.
        decoder: torch.nn.Sequential, Decoder model.
        early_stopping_epoch: int, the epoch found by early stopping.
        encoder_early_stopping: torch.nn.Sequential, Encoder model pre-trained used for
            finetuning_rpz after finding best epoch with early stopping.
        decoder_early_stopping: torch.nn.Sequential, Decoder model pre-trained used for
            finetuning_rpz after finding best epoch with early stopping.
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
        criterion=torch.nn.MSELoss(),
        optimizer=torch.optim.Adam,
    ):
        if early_stopping_use and early_stopping_patience >= max_num_epochs:
            raise AssertionError

        super().__init__()

        self.random_state = random_state
        torch.manual_seed(self.random_state)
        self.repr_dim = repr_dim
        self.hidden_n_layers = hidden_n_layers
        self.hidden_n_units_first = hidden_n_units_first
        self.hidden_decrease_rate = hidden_decrease_rate
        self.hidden = self.convert_hidden_config()
        self.dropout = dropout
        self.activation = activation
        self.bias = bias
        self.num_epochs = max_num_epochs if early_stopping_use else num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.early_stopping_use = early_stopping_use
        self.early_stopping_split = early_stopping_split
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_delta = early_stopping_delta
        self.early_stopping_epoch = None
        self.device = device
        self.criterion = criterion
        self.optimizer = optimizer

        # Attributes
        self.in_features = 0
        self.train_loss = []
        self.eval_loss = []
        self.encoder, self.encoder_early_stopping = None, None
        self.decoder, self.decoder_early_stopping = None, None

    def _init_models(self):
        """Init method to build the encoder/decoder neural networks.

        This method is separate from the init because it depends on the number of
        features in the training data X (the shape of the first encoder layer and last
        layer of decoder). This is thus called at each new fit call.

        Raises:
            ValueError if the configuration of the model is incorrect.
        """

        encoder_output_sizes = self.hidden + [self.repr_dim]
        # [::-1] Invert the array of hidden for the decoder
        decoder_output_sizes = self.hidden[::-1] if len(self.hidden) > 0 else []
        decoder_output_sizes.append(self.in_features)

        in_features_layer = self.in_features

        encoder_layers = []
        for i, size_of_layer_i in enumerate(encoder_output_sizes):
            layer_args = [
                torch.nn.Linear(
                    in_features=in_features_layer,
                    out_features=size_of_layer_i,
                    bias=self.bias,
                )
            ]
            in_features_layer = size_of_layer_i

            if (self.activation is not None) and (i + 1 != len(encoder_output_sizes)):
                layer_args.append(self.activation)

            if (self.dropout is not None) and (i + 1 != len(encoder_output_sizes)):
                layer_args.append(torch.nn.Dropout(self.dropout))

            encoder_layers.append(torch.nn.Sequential(*layer_args))

        decoder_layers = []
        for i, size_of_layer_i in enumerate(decoder_output_sizes):
            layer_args = [
                torch.nn.Linear(
                    in_features=in_features_layer,
                    out_features=size_of_layer_i,
                    bias=self.bias,
                )
            ]
            in_features_layer = size_of_layer_i

            if (self.activation is not None) and (i + 1 != len(decoder_output_sizes)):
                layer_args.append(self.activation)

            if (self.dropout is not None) and (i + 1 != len(decoder_output_sizes)):
                layer_args.append(torch.nn.Dropout(self.dropout))

            decoder_layers.append(torch.nn.Sequential(*layer_args))

        self.encoder = torch.nn.Sequential(*encoder_layers)
        self.decoder = torch.nn.Sequential(*decoder_layers)

        self.encoder.to(self.device)
        self.decoder.to(self.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Computes the encoding and decoding of the input x.

        Args:
            x (torch.Tensor): input tensor.

        Returns:
            tensor (x_dec) reconstructed by the network.
        """
        return self.decoder(self.encoder(x))

    def fit(
        self,
        X: pd.DataFrame,
        metrics_suffix: str = None,
        finetuning_rpz=False,
        **kwargs,
    ):
        """Fit the model according to the given training data X.

        Args:
            X (pd.DataFrame): training data.
            metrics_suffix (str): repeat/split information to be logged with the loss.
            finetuning_rpz (bool): if true, don't re-initialise the model
            **kwargs: variable number of keyword arguments, to be compatible with
                method from parent class.

        Returns:
            self with fitted decoder/encoder.
        """
        del kwargs

        # avoid printing "None" in MLflow
        metrics_suffix = metrics_suffix if metrics_suffix else ""

        # reset if finetuning_rpz or early_stopping
        self.train_loss, self.eval_loss = [], []

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

        sample = iter(dataloader).next()
        self.in_features = sample.shape[1]

        if finetuning_rpz and self.early_stopping_use:
            # save the pre-trained AE to re-train it with the right weights after
            # early stopping is finished
            self.encoder_early_stopping, self.decoder_early_stopping = (
                copy.deepcopy(self.encoder),
                copy.deepcopy(self.decoder),
            )
        elif finetuning_rpz:
            # either early stopping was already performed:
            if self.encoder_early_stopping is not None:
                # Early stopping was used just before, so we take the saved pre-train AE
                # for the finetuning_rpz training on the train+val with the epoch found
                # by early stopping on train.
                self.encoder, self.decoder = (
                    self.encoder_early_stopping,
                    self.decoder_early_stopping,
                )
            # or early stopping was not meant to happen in the experiment at all:
            else:
                # we just keep the pre-trained AE for the finetuning_rpz AE
                pass
        else:
            self._init_models()

        params = list(self.encoder.parameters()) + list(self.decoder.parameters())
        optimizer = self.optimizer(params, lr=self.learning_rate)

        early_stopping_best, early_stopping_patience_count = 0, 0
        for epoch in tqdm(range(self.num_epochs), total=self.num_epochs):
            # self.train() in the epoch loop because self.evaluate() calls self.eval()
            self.train()
            train_losses = []
            for data_batch in dataloader:
                data_batch = data_batch.to(self.device)

                data_batch_reconstructed = self.forward(data_batch)
                loss = self.criterion(data_batch_reconstructed, data_batch)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss = loss.detach().cpu().numpy()

                train_losses.append(train_loss)

            self.train_loss.append(np.mean(train_losses))
            mlflow.log_metric(
                f"Train loss {metrics_suffix}", np.mean(train_losses), step=epoch
            )

            if self.early_stopping_use:
                self.evaluate(_x_val)
                mlflow.log_metric(
                    f"Eval loss {metrics_suffix}", self.eval_loss[-1], step=epoch
                )

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

    def evaluate(self, X: pd.DataFrame):
        """Function to evaluate the model performance on X.

        Update the self.eval_loss attribute by computing the reconstruction of X and
            then evaluating the reconstructed X against the actual X using the
            self.criterion. This is done for each data batch and the mean eval loss
            across batched is then add to the eval_loss (which are a list of mean loss
            by epoch).

        Args:
            X (pd.DataFrame):Training data in input.

        Returns:
            self, with eval_loss updated.
        """
        dataset = OmicsDataset(X.values)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
        )

        self.eval()

        eval_losses = []
        for data_batch in dataloader:
            data_batch = data_batch.to(self.device)

            data_batch_reconstructed = self.forward(data_batch)
            loss = self.criterion(data_batch_reconstructed, data_batch)

            eval_losses.append(loss.detach().cpu().numpy())

        self.eval_loss.append(np.mean(eval_losses))

        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Encode the data using the fitted model.

        Args:
            X (pd.DataFrame): training data.

        Returns:
            np.ndarray: encoded data.
        """
        dataset = OmicsDataset(X.values)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
        )

        features = []
        for data_batch in tqdm(dataloader, total=len(dataloader)):
            # https://stackoverflow.com/questions/59560043/what-is-the-difference-between-model-todevice-and-model-model-todevice
            data_batch = data_batch.to(self.device)
            data_batch_reconstructed = self.encoder(data_batch)
            features.append(data_batch_reconstructed.detach())

        features = torch.cat(features, dim=0).cpu().numpy()

        return features

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

    def metagenes(self, X: pd.DataFrame) -> np.ndarray:
        """Fit model on X and decode identity matrix.

        Args:
            X (pd.DataFrame): Dataframe in input.

        Returns:
            np.ndarray: input associated to an identity matrix representation
        """
        # Should we fit the model here?
        self.fit(X)
        identity_matrix = torch.eye(self.repr_dim)
        decoded_expression = (
            self.decoder(identity_matrix).detach().clone().cpu().numpy()
        )
        return decoded_expression

    def convert_hidden_config(self):
        """Convert from the new 'hidden' config (3 params) to the old 'hidden' config.

        Convert from the 3 hidden layer configs (n_layers, n_units_first, and
        decrease_rate) to the traditional hidden list containing the number of nodes
        in each hidden layer.

        Returns:
            List[int]: number of units in each of the encoder's hidden layers
        """
        hidden = []

        if self.hidden_n_layers == 0:
            return hidden

        hidden.append(self.hidden_n_units_first)
        for i in range(1, self.hidden_n_layers):
            hidden.append(int(hidden[i - 1] * self.hidden_decrease_rate))

        return hidden
