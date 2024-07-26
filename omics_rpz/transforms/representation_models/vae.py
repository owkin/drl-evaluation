"""Class for the VAE and beta-VAE representation models."""

import copy

import mlflow
import numpy as np
import pandas as pd
import torch
from loguru import logger
from sklearn.model_selection import train_test_split

from omics_rpz.data import OmicsDataset
from omics_rpz.transforms.representation_models.auto_encoder import AutoEncoder
from omics_rpz.utils import draw_umap, initialize_early_stopping, update_early_stopping


class VariationalEncoder(torch.nn.Module):
    """Encoder part of the VAE.

    The encoder receives an input and generates a
    representation of the input as a probability distibution typically a Gaussian
    distribution N(mu(x), sigma(x)) that it samples from. The encoder calculates as well
    a KL divergence between this distribution and a prior that is then added to the
    reconstruction loss of the auto-encoder for training.

    Parameters
    ----------
        in_features (int): size of input features from the training dataset.
        repr_dim (int): size of the representation = output.
        hidden (list[int]): sizes of hidden layers.
        bias (bool): if False, the layers will not learn an additive bias.
        device (str): whether to use cpu or CUDA to run computations.
        random_state (int): random seed for the model.
        monte_carlo (bool): to choose between analytic or MC calculation of the KL
            divergence between N(mu, sigma) and N(0, Id) where mu and sigma are output
            by the encoder.
        use_mean (bool): to choose if the representation fed to downstream tasks is the
            posterior mean mu or the latent variable z.
    """

    def __init__(
        self,
        in_features: int,
        repr_dim: int,
        hidden: list[int] = None,
        bias: bool = True,
        device: str = "cpu",
        random_state: int = 5,
        monte_carlo: bool = False,
        dropout: float = 0.0,
        use_mean: bool = True,
    ):
        super().__init__()
        torch.manual_seed(random_state)
        if hidden is None:
            hidden = []

        self.device = device
        self.repr_dim = repr_dim
        self.monte_carlo = monte_carlo
        self.dropout = dropout
        self.use_mean = use_mean

        encoder_hidden = hidden
        encoder_layers = []
        for hidden_layer in encoder_hidden:
            encoder_layers.append(torch.nn.Linear(in_features, hidden_layer, bias=bias))
            encoder_layers.append(torch.nn.Dropout(p=self.dropout))
            encoder_layers.append(torch.nn.ReLU())
            in_features = hidden_layer
        self.enco = torch.nn.Sequential(*encoder_layers)
        self.linear2 = torch.nn.Linear(in_features, self.repr_dim)
        self.linear3 = torch.nn.Linear(in_features, self.repr_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward function of the encoder part of the VAE, provides representations for
        downstream tasks.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Compressed representation.
        """
        x = x.to(self.device)

        # Should we use z (self.forward2(x)[0]) or mu(x) (self.forward2(x)[1]) as a
        # representation of x for downstream tasks ?
        if self.use_mean:
            return self.forward2(x)[1]

        return self.forward2(x)[0]

    def forward2(self, x: torch.Tensor) -> torch.Tensor:
        """Output the latent variable z and parameters (mu(x), sigma(x)) of the
        distribution from which it is drawn.

        The forward method of the encoder has to output only one tensor
        (because of the fine-tuning implementation)
        but z, mu and sigma are needed for the KL divergence calculation in training
        hence the 2 forward methods.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor (encoder input).

        Returns
        -------
        Union[torch.Tensor, torch.Tensor, torch.Tensor]
            Tuple containing:
                torch.Tensor
                    Latent variable z sampled from N(mu(x), sigma(x)).
                torch.Tensor
                    Average mu(x) of the distribution from which z is drawn.
                torch.Tensor
                    Std dev sigma(x) of the distribution from which z is drawn.
        """
        x = x.to(self.device)
        x_encoded = self.enco(x)
        mu = self.linear2(x_encoded)
        log_var = self.linear3(x_encoded)
        sigma = torch.exp(log_var / 2)
        q = torch.distributions.Normal(mu, sigma)
        z = q.rsample()
        return z, mu, sigma

    def kl_div(self, z, mu, sigma):
        """KL divergence function.

        The function calculates the KL divergence between the distribution
        N(mu(x), sigma(x) and a prior.
        Calculation is analytic or uses a MC approximation according to the
        attribute of the encoder: self.monte_carlo

        Parameters
        ----------
        z : torch.Tensor
            Latent representation tensor.
        mu: torch.Tensor
            Mean of the distribution from which z is drawn.
        sigma: torch.Tensor
            Variance of the distribution from which z is drawn.
        """
        # define prior p(z)
        p_z = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(sigma))
        # define posterior q(z|x)
        q = torch.distributions.Normal(mu, sigma)
        if self.monte_carlo:
            # Monte Carlo KL calc (uses z)
            log_qzx = q.log_prob(z)
            log_pz = p_z.log_prob(z)
            kl_div = log_qzx - log_pz
            # sum over last dimension to go from single dim distribution to multi-dim
            kl_div = kl_div.sum(-1).mean()
        else:
            # analytic KL calc (doesn't use z)
            kl_div = torch.distributions.kl_divergence(q, p_z).sum(dim=1).mean()
        return kl_div


class Decoder(torch.nn.Module):
    """Decoder part of the VAE.

    The decoder takes as input the representation output by the encoder and tries to
    reconstruct the auto-encoder's input.

    Parameters
    ----------
        in_features (int): size of input and output of the VAE.
        repr_dim (int): size of the representation = input of the decoder.
        hidden (list[int]): sizes of hidden layers.
        bias (bool): if False, the layers will not learn an additive bias.
        device (str): whether to use cpu or CUDA to run computations.
        random_state (int): Random seed for the model.
    """

    def __init__(
        self,
        in_features: int,
        repr_dim: int,
        hidden: list[int] = None,
        bias: bool = True,
        device: str = "cpu",
        random_state: int = 5,
        dropout: float = 0.0,
    ):
        super().__init__()
        torch.manual_seed(random_state)
        if hidden is None:
            hidden = []

        self.repr_dim = repr_dim
        self.dropout = dropout
        decoder_hidden = hidden[::-1] if len(hidden) > 0 else []
        decoder_layers = []
        dim = self.repr_dim
        for hidden_layer in decoder_hidden:
            decoder_layers.append(torch.nn.Linear(dim, hidden_layer, bias=bias))
            decoder_layers.append(torch.nn.Dropout(p=self.dropout))
            decoder_layers.append(torch.nn.ReLU())
            dim = hidden_layer
        decoder_layers.append(torch.nn.Linear(dim, in_features, bias=bias))
        self.deco = torch.nn.Sequential(*decoder_layers)

        self.device = device
        # more flexible loss function allowed (likelihood with any p(x|z) defined in
        # the Gaussian likelihood function)
        # scale param used for the Gaussian likelihood below
        self.log_scale = torch.nn.Parameter(torch.Tensor([0.0]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward function of the decoder part of the VAE.

        Parameters
        ----------
        x : torch.tensor
            Compressed representation (decoder input).

        Returns
        -------
        torch.tensor
            Reconstructed data (decoder output).
        """
        return self.deco(x)

    def gaussian_likelihood(
        self, x_hat: torch.Tensor, logscale: torch.Tensor, x: torch.Tensor
    ) -> float:
        """Gaussian likelihood function.

        Parameters
        ----------
        x_hat : torch.tensor
            Output of the VAE
        logscale : torch.tensor
            Trainable parameter of the VAE, output of the VAE is N(x_hat, logscale)
            in fact
        x : torch.tensor
            Input of the VAE

        Returns
        -------
        float
            Likelihood of x on distribution N(x_hat, logscale)
        """
        scale = torch.exp(logscale)
        mean = x_hat
        # define p(output|z) = N(x_hat, scale)
        dist = torch.distributions.Normal(mean, scale)
        # measure probability of seeing x under p(output|z)
        log_pxz = dist.log_prob(x)
        # sum across channels and pixels for log probability of image
        log_pxz = log_pxz.sum(dim=1).mean()
        return log_pxz


class VariationalAutoencoder(torch.nn.Module):
    """Implementation of the variational autoencoder (VAE) representation model.

    Parameters
    ----------
        repr_dim (int): size of the representation dimension.
        hidden_n_layers (int): number of hidden layers
        hidden_n_units_first (int): number of units of the first hidden layer
        hidden_decrease_rate (float): decrease rate of the number of units / layer
        bias (bool): if False, the layers will not learn an additive bias.
        num_epochs (int): maximum number of iterations during training.
        batch_size (int): size of minibatches for the training algorithm.
        learning_rate (float): controls the step-size in updating the weights.
        device (str): whether to use cpu or CUDA to run computations.
        random_state (int): random seed for the model.
        split_data (bool): whether to split the data in train test during training.
        beta_max (float): maximum value of the coefficient beta that multiplies the
            KL-divergence term in beta-VAE. Param of the KL annealing cycle.
        beta_num_cycles (int): Param of the KL annealing cycle.
        beta_ramping_iters (float): Param of the KL annealing cycle.
        weight_decay (float): parameter of the Adam optimization algorithm.
        monte_carlo (bool): to choose between analytic or MC calculation of the KL
            divergence between N(mu, sigma) and N(0, Id) where mu and sigma are output
            by the encoder.
        dropout (float): dropout rate for the encoder and the decoder.
        draw_plots (bool): to choose to draw umap plots of the inputs and the
            representations.
        use_mean (bool): to choose if the representation fed to downstream tasks is the
            posterior mean mu or the latent variable z.
        early_stopping_use (bool): to choose early stopping (not implemented).

    Attributes
    ----------
        encoder (VariationalEncoder): the encoder neural network.
        decoder (Decoder): the decoder neural network.
        in_features (int): number of input features from the training dataset.
    """

    def __init__(
        self,
        repr_dim: int,
        hidden_n_layers: int = 2,
        hidden_n_units_first: int = 512,
        hidden_decrease_rate: float = 0.5,
        bias: bool = True,
        num_epochs: int = 10,
        batch_size: int = 16,
        learning_rate: float = 1.0e-3,
        device: str = "cpu",
        random_state: int = 5,
        beta_max: float = 1.0,
        beta_num_cycles: int = 1,
        beta_ramping_iters: float = 0.0,
        epochs_ref_annealing_cycle: int = 200,
        weight_decay: float = 1.0e-5,
        monte_carlo: bool = False,
        dropout: float = 0.0,
        draw_plots: bool = False,
        use_mean: bool = True,
        early_stopping_use: bool = False,
        max_num_epochs: int = 300,
        early_stopping_split: float = 0.2,
        early_stopping_patience: int = 50,
        early_stopping_delta: float = 0.001,
    ):
        if early_stopping_use and early_stopping_patience >= max_num_epochs:
            raise AssertionError
        super().__init__()
        self.hidden_n_layers = hidden_n_layers
        self.hidden_n_units_first = hidden_n_units_first
        self.hidden_decrease_rate = hidden_decrease_rate
        self.hidden = AutoEncoder.convert_hidden_config(self)
        self.repr_dim = repr_dim
        self.bias = bias
        self.num_epochs = max_num_epochs if early_stopping_use else num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.device = device
        self.random_state = random_state
        torch.manual_seed(self.random_state)
        self.beta = 1.0
        self.beta_max = beta_max
        self.beta_num_cycles = beta_num_cycles
        self.beta_ramping_iters = beta_ramping_iters
        self.epochs_ref_annealing_cycle = epochs_ref_annealing_cycle
        self.weight_decay = weight_decay
        self.monte_carlo = monte_carlo
        self.encoder, self.encoder_early_stopping = None, None
        self.decoder, self.decoder_early_stopping = None, None
        self.in_features = 0
        if dropout is None:
            dropout = 0.0
        self.dropout = dropout
        self.draw_plots = draw_plots
        self.use_mean = use_mean
        self.early_stopping_use = early_stopping_use
        self.early_stopping_split = early_stopping_split
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_delta = early_stopping_delta
        self.early_stopping_epoch = None
        self.eval_loss = []

    def _init_models(self):
        self.encoder = VariationalEncoder(
            in_features=self.in_features,
            repr_dim=self.repr_dim,
            hidden=self.hidden,
            bias=self.bias,
            device=self.device,
            random_state=self.random_state,
            monte_carlo=self.monte_carlo,
            dropout=self.dropout,
            use_mean=self.use_mean,
        )
        self.decoder = Decoder(
            in_features=self.in_features,
            repr_dim=self.repr_dim,
            hidden=self.hidden,
            bias=self.bias,
            device=self.device,
            random_state=self.random_state,
            dropout=self.dropout,
        )
        self.encoder.to(self.device)
        self.decoder.to(self.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the encoding and decoding of the input tensor x.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor (encoder input).
        Returns
        -------
        torch.Tensor
            Reconstruction of the input tensor (decoder output).
        """
        x = x.to(self.device)
        z = self.encoder.forward2(x)[0]
        x_hat = self.decoder(z)
        return x_hat

    def loss_fn(
        self, x_hat: torch.Tensor, z, mu, sigma, x: torch.Tensor
    ) -> torch.Tensor:
        """Loss function for the VAE.

        Includes beta * KL divergence like in the beta-VAE
        The reconstruction loss is the Gaussian likelihood of x under the distribution
        N(x_hat, exp(logscale)) where logscale is learnt.

        Parameters
        ----------
        x_hat : torch.Tensor
            Reconstruction of the input tensor (decoder output).
        z : torch.Tensor
            Latent representation tensor (encoder output, decoder input)
        mu: torch.Tensor
            Mean of the distribution from which z is drawn (encoder output).
        sigma: torch.Tensor
            Variance of the distribution from which z is drawn (encoder output).
        x : torch.Tensor
            Input tensor (encoder input).

        Returns
        -------
        Union[torch.Tensor, torch.Tensor, torch.Tensor]
            Tuple containing:
                torch.Tensor
                    Sum of the reconstruction loss and the beta-weighted KL-divergence
                    between the approximate posterior q(z|x) and the prior p(z).
                torch.Tensor
                    Reconstruction loss.
                torch.Tensor
                    Beta-weighted KL-divergence
        """
        rec_loss = -self.decoder.gaussian_likelihood(x_hat, self.decoder.log_scale, x)
        kl_d = self.encoder.kl_div(z, mu, sigma)
        loss = rec_loss + self.beta * kl_d
        return loss, rec_loss, kl_d

    def beta_annealing(self, num_iter, total_num_iter):
        """Implementation of the KL cyclical annealing technique.

        See the following article https://arxiv.org/abs/1903.10145 for a description of
        the algorithm.
        The function is flexible enough to accomodate many different beta profiles.

        Parameters
        ----------
        num_iter: int
            Current training iteration.
        total_num_iter: int
            Total number of training iterations.

        Returns
        -------
        beta: float
            Recommended beta (weighting factor of the KL divergence in the loss).
        """

        cycled_iter = (num_iter % (int(total_num_iter / self.beta_num_cycles))) / (
            total_num_iter / self.beta_num_cycles
        )
        if cycled_iter < self.beta_ramping_iters:
            beta = cycled_iter / self.beta_ramping_iters * self.beta_max
        else:
            beta = self.beta_max
        return beta

    # pylint: disable=too-many-statements
    # pylint: disable=too-many-branches
    # pylint: disable=duplicate-code
    def fit(
        self,
        X: pd.DataFrame,
        metrics_suffix: str = None,
        finetuning_rpz=False,
        **kwargs,
    ):
        """Fit the VAE according to the given training data X.

        Parameters
        ----------
        X : pd.DataFrame
            Training data.
        metrics_suffix : str
            Repeat/split information to be logged with the loss.
        finetuning_rpz: bool
            If true, don't re-initialise the model.
        **kwargs
            Variable number of keyword arguments, to be compatible with
            other representation models.

        Returns
        -------
            self with fitted decoder/encoder.
        """
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

        self.in_features = X.shape[1]

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
        optimizer = torch.optim.Adam(
            params, lr=self.learning_rate, weight_decay=self.weight_decay
        )

        # iteration (i.e. batch) counting for the annealing cycle uses a fixed
        # self.epochs_ref_annealing_cycle in order to avoid pbls with early stopping
        # i.e. always get the same annealing cycle
        num_iter = 0
        total_num_iter = len(dataloader) * self.epochs_ref_annealing_cycle

        self.eval_loss = []
        early_stopping_best, early_stopping_patience_count = 0, 0

        for epoch in range(self.num_epochs):
            self.train()
            self.encoder.train()
            self.decoder.train()
            epoch_loss, epoch_recloss, epoch_kl = 0, 0, 0
            for x in dataloader:
                self.beta = self.beta_annealing(num_iter, total_num_iter)
                num_iter += 1
                x = x.to(self.device)
                z, mu, sigma = self.encoder.forward2(x)
                x_hat = self.decoder(z)
                loss, rec_loss, kl_d = self.loss_fn(x_hat, z, mu, sigma, x)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss, epoch_recloss, epoch_kl = (
                    epoch_loss + loss,
                    epoch_recloss + rec_loss,
                    epoch_kl + kl_d,
                )
            if epoch % 10 == 0:
                logger.info(
                    f"Epoch {epoch + 1}/{self.num_epochs}"
                    f" Train loss {epoch_loss/len(dataloader)}"
                    f" Rec loss {epoch_recloss/len(dataloader)}"
                    f" KL div {epoch_kl/len(dataloader)}"
                    f" Beta {self.beta}"
                    f" Decoder log_scale {self.decoder.log_scale.item()}"
                )

            mlflow.log_metric(
                f"Train loss {metrics_suffix}", epoch_loss / len(dataloader), step=epoch
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
                        f"VAE training finished by early stopping at epoch {epoch + 1}"
                    )
                    self.early_stopping_epoch = (
                        epoch + 1  # to return it if someone wants to use this info
                    )
                    break

        else:  # it means I finished all my epochs
            logger.info(f"VAE training finished with the max epoch number: {epoch + 1}")
            self.early_stopping_epoch = (
                self.num_epochs  # to return it if someone wants to use this info
            )

        if self.early_stopping_use:
            self.early_stopping_use = False
            self.num_epochs = self.early_stopping_epoch - self.early_stopping_patience
            self.fit(
                X_full,
                metrics_suffix=metrics_suffix + "retrain",
            )

        if self.draw_plots:
            self.draw_rpzplots(dataset, dataloader)

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
        self.encoder.eval()
        self.decoder.eval()
        eval_losses = []
        for data_batch in dataloader:
            x = data_batch.to(self.device)
            z, mu, sigma = self.encoder.forward2(x)
            x_hat = self.decoder(z)
            loss = self.loss_fn(x_hat, z, mu, sigma, x)[0]
            eval_losses.append(loss.detach().cpu().numpy())
        self.eval_loss.append(np.mean(eval_losses))
        # logger.info(f"Eval loss {self.eval_loss}")
        return self

    def draw_rpzplots(self, dataset, dataloader):
        """Method that draws umap plots on the VAE's input and rep.

        Parameters
        ----------
        dataset : torch.dataset
            Training dataset.
        dataloader : torch.dataloader
            Training dataloader.

        Returns
        -------
        umap plots
        """
        self.eval()
        self.encoder.eval()
        self.decoder.eval()
        draw_umap(pd.DataFrame(dataset.X), filename="X_before_rpz")
        features = []
        for x in dataloader:
            x = x.to(self.device)
            f_x = self.encoder(x)
            features.append(f_x.detach().clone())
        features = torch.cat(features, dim=0).cpu().numpy()
        draw_umap(pd.DataFrame(features), filename="X_after_rpz")

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Inference method of the encoder part of the VAE.

        Calc of the representation of the input. Assign the variable draw_plots below
        to True if you want to draw plots for X, z and X reconstructed.

        Parameters
        ----------
        X : pd.DataFrame
            Training data.

        Returns
        -------
        np.ndarray
            Encoded data.
        """

        dataset = OmicsDataset(X.values)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
        )

        self.eval()
        self.encoder.eval()

        features = []
        for x in dataloader:
            x = x.to(self.device)
            f_x = self.encoder(x)
            features.append(f_x.detach().clone())

        if self.device == "cpu":
            features = torch.cat(features, dim=0).numpy()
        else:
            features = torch.cat(features, dim=0).cpu().numpy()

        return features

    def fit_transform(
        self, X: pd.DataFrame, metrics_suffix: str, **kwargs
    ) -> np.ndarray:
        """Training of the VAE on a dataset X + inference of the rep of X.

        Parameters
        ----------
        X : pd.DataFrame)
            Training data.
        metrics_suffix :str
            Repeat/split information to be logged with the loss
        **kwargs
            Variable number of keyword arguments, to be compatible with
            other representation models.

        Returns
        -------
        np.ndarray
            Encoded data.
        """
        del kwargs
        _ = self.fit(X, metrics_suffix)
        return self.transform(X)

    def metagenes(self, X: pd.DataFrame) -> np.ndarray:
        """Calculate the metagenes of the VAE.

        A metagene is supposed to be the input associated to a rep vector equal to
        [0, ..., 0, 1, 0, ..., 0]. The metagene is calculated using the decoder.

        Parameters
        ----------
        X : pd.DataFrame
            Training data.

        Returns
        -------
        np.ndarray
            Input associated to an identity matrix representation.
        """
        self.train()
        self.encoder.train()
        self.decoder.train()
        self.fit(X)
        self.eval()
        self.encoder.eval()
        id_mat = torch.eye(self.encoder.repr_dim).to(self.device)
        if self.device == "cpu":
            return self.decoder(id_mat).detach().clone().numpy()
        return self.decoder(id_mat).detach().clone().cpu().numpy()
