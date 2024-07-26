"""Class implementing the Gnn GNN architecture."""

from typing import Optional

import mlflow
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from loguru import logger
from torch.nn import Linear, MSELoss
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import (
    ASAPooling,
    GATConv,
    GCNConv,
    GraphConv,
    SAGEConv,
    SAGPooling,
    Sequential,
    TopKPooling,
    avg_pool_x,
    max_pool_x,
)
from torch_geometric.seed import seed_everything
from typing_extensions import Self

from omics_rpz.data import load_graph
from omics_rpz.losses import CoxLoss
from omics_rpz.transforms.representation_models.auto_encoder import AutoEncoder

MSG_LAYER = {
    'GCN': GCNConv,
    'SAGE': SAGEConv,
    'GAT': GATConv,
    'GraphConv': GraphConv,
}
POOLING_LAYER = {
    'average': avg_pool_x,
    'max': max_pool_x,
    'SAG': SAGPooling,
    'TOPK': TopKPooling,
    'ASA': ASAPooling,
}


class Gnn(torch.nn.Module):
    """Implementation of the supervised Graph Neural Network architecture inpspired by
    multiple litterature sources, mostly the DeepMOCCA paper.

    Parameters:
        hidden_channels: List[str], Dimensions of the embeddings for different
            GNN convolutinal layers. This parameter aldo sets the number of conv layers.
        out_dim: int, Dimension of the output we are trying to predict in a supervised
            manner.
        sigmoid: bool = True, Add sigmoid after prediction MLP when predicting OS.
        device: Optional[str] = "cpu", Whether to use cpu or CUDA to run computations.
        learning_rate: float = 1e-3, Learning rate.
        num_epochs: int = 50, Number of epochs.
        dropout: float = 0.5, Dropout rate used in the linear layer after the conv
            layers to get the right embedding dimension.
        repr_dim: int = 128, Size of the representation dimension.
        batch_size: int = 16, Batch size used when loading the dataloader.
        message_passing: str = "", Message passing layer chosen for the graph
            convolution. Must be a key of the MSG_LAYER dictionnary.
        pooling: str = "", Pooling method chosen to regroup the node embeddings. Must be
            a key of the POOLING_LAYER dictionnary.
        aggregation_SAGEConv="mean", Aggregation method used for the SAGEConv layer.
            Possible options are "mean", "max", or "lstm". Defaluts to "mean".
        optimizer: torch.optim, by default Adam, optimizer to use in gradient descent.
        criterion=CoxLoss(), Loss criterion for the supervised training.
        unsupervised: bool = True, Allows to decide if the model will be unsupervised
            or not.
        graph_loading: Optional[dict[str,str]] = None
            Dictionary containing the parameters passed on to the
            `omics_rpz.data.load_graph()` function :
                - string_threshold (float): Score threshold of the edges we want to keep
                    in the graph. This score relates to the top quantile we want to
                    keep.
                - clustering (str):  Method that we use to perform the clustering on the
                    graph. Defaults to 'louvain'.
                - cluster_resolution (int, optional): Parameter to be passed on if the
                    clustering method is Louvain. This parameter affects the resulting
                    number of clusters. Larger resolution leads to smaller and more
                    numerous clusters. Defaults to 2000.
                - n_clusters (int, optional): Parameter to decide number of clusters we
                    want, if the clustering method allows it. Defaults to 200.
                - permute_gene_names (bool, optional): Allows to permute gene names to
                    have a random graph. Defaults to False.
                - clustering_seed: int = 42, Random seed for clustering.
                - pathway_source: str = 'KEGG', Database used for pathway clustering.
        early_stopping_use=False, If we want to use early stopping implementation.
        decoder_params: Optional[dict[str,str]] = None
            Dictionary containing the auto-encoder parameters passed on to the
            `omics_rpz.transforms.representation_models.AutoEncoder` class.
                - autoencoder_hidden_n_layers: int = 2, Hidden layers for the decoder.
                - autoencoder_hidden_n_units_first: int = 128, Size of the first layer
                    of thevdecoder.
                - autoencoder_hidden_decrease_rate: float =0.5, Decrease rate of the
                    decoder layers sizes.
                - autoencoder_dropout: float = 0.2, Decoder dropout rate.


    Attributes:
        edge_index: torch.Tensor, Will be the edges of the prior graph considered.
        self.clusters: torch.Tensor, Cluster assignment coming from the graph loading.
        self.n_clusters: int, Number of clusters.
        self.lin_emb: torch.nn.Linear, Linear layer to adapt the size of embedding.
        self.lin_pred1: torch.nn.Linear, First MLP layer for the prediction part of
            the model.
        self.lin_pred2: torch.nn.Linear, Second MLP layer for the prediction part of the
            model.
        self.long_clusters: torch.Tensor, Cluster assignments adapted to the graphs in
            the same batch.
        self.labels, Labels to predict.
        self.reduced_mashups, Gene Mashup embeddings.
        self.train_loss: List = [], List of mean train loss computed on the different
            data batches using the criterion.
        self.eval_loss: List = [], List of mean eval loss computed on the different data
            batches using the criterion.
    """

    def __init__(
        self,
        hidden_channels: list[int],
        out_dim: int,
        sigmoid: bool = True,
        # add edge index and clusters (and remove str thresh and clustering and genes)
        device: Optional[str] = "cpu",
        learning_rate: float = 1e-3,
        num_epochs: int = 50,
        dropout: float = 0.5,
        repr_dim: int = 128,
        batch_size: int = 16,
        message_passing: str = "",
        pooling: str = "",
        aggregation_SAGEConv="mean",
        optimizer=torch.optim.Adam,
        criterion=CoxLoss(),
        unsupervised: bool = True,
        graph_loading: Optional[dict] = None,
        central_node: bool = False,
        early_stopping_use: bool = False,
        global_seed: int = 42,
        decoder_params: Optional[dict] = None,
        **kwargs,
    ):
        super().__init__()

        self.hidden_channels = hidden_channels
        self.out_dim = out_dim
        self.device = device
        self.sigmoid = sigmoid
        self.criterion = criterion
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.dropout = dropout
        self.repr_dim = repr_dim
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.message_passing = MSG_LAYER.get(message_passing, SAGEConv)
        self.pooling = POOLING_LAYER.get(pooling, max_pool_x)
        self.aggr = aggregation_SAGEConv
        self.unsupervised = unsupervised
        self.graph_loading = graph_loading if graph_loading is not None else {}
        self.central_node = central_node
        self.early_stopping_use = early_stopping_use
        self.global_seed = global_seed
        self.decoder_params = decoder_params if decoder_params is not None else {}

        # Attributes

        self.edge_index = None
        self.clusters = None
        self.n_clusters = None
        self.lin_emb = Linear(1, 1)
        self.lin_pred1 = Linear(1, 1)
        self.lin_pred2 = Linear(1, 1)
        self.long_clusters = None
        self.labels = None
        self.reduced_mashups = None
        self.train_loss = []
        self.eval_loss = []
        self.decoder = None
        self.input_size = None
        self.genes = None

        del kwargs

    def _init_models(self):
        seed_everything(self.global_seed)
        message_passing_layers = []
        in_features_dim = 1
        for i, hidden_channels in enumerate(self.hidden_channels):
            message_passing_layers.append(
                (
                    self.message_passing(
                        in_features_dim,
                        hidden_channels,
                        aggr=self.aggr,
                    ),
                    'x, edge_index -> x',
                )
            )
            in_features_dim = hidden_channels

            # if i < (len(self.hidden_channels) - 1):
            #     message_passing_layers.append(ReLU(inplace=True))

        self.message_passing = Sequential('x, edge_index', message_passing_layers).to(
            self.device
        )

        self.lin_emb = Linear(
            self.n_clusters * self.hidden_channels[-1], self.repr_dim
        ).to(self.device)

        self.lin_pred1 = Linear(self.repr_dim, 16).to(self.device)
        self.lin_pred2 = Linear(16, self.out_dim).to(self.device)

        long_clusters = self.clusters
        for i in range(1, self.batch_size):
            offset = i * self.n_clusters
            long_clusters = torch.concat(
                (long_clusters, torch.add(self.clusters, offset))
            )
        self.long_clusters = long_clusters

        if self.unsupervised:
            autoencoder = AutoEncoder(
                repr_dim=self.repr_dim,
                device=self.device,
                early_stopping_use=False,
                **self.decoder_params,
            )
            autoencoder.in_features = self.input_size
            autoencoder._init_models()
            self.decoder = autoencoder.decoder
            self.criterion = MSELoss()

    def embed_nodes(self, x, edge_index, batch):
        """Generate embedding by performing graph convolutions.

        Args:
            x (torch.Tensor): Node features. Usually RNAseq expression.
            edge_index (torch.Tensor): Edge list of the PPI we are using.
            batch (torch.Tensor):  Batch vector which assigns each node to a specific
                batch number.

        Returns:
            torch.Tensor: Embedding tensor for the inputs from this batch.
        """
        batch_size = batch.max() + 1
        # 1. Obtain node embeddings
        x = self.message_passing(x, edge_index)

        # 2. Pooling
        if batch_size == self.batch_size:
            long_clusters = self.long_clusters
        else:
            long_clusters = self.long_clusters[: batch_size * self.clusters.size(0)]
        x = self.pooling(long_clusters, x, batch, self.n_clusters)[0]

        # Dimensionality reduction
        x = torch.reshape(x, (batch_size, self.n_clusters * self.hidden_channels[-1]))
        x = F.dropout(x, p=self.dropout, training=self.training).to(self.device)
        x = self.lin_emb(x)
        return x

    def pred(self, x):
        """Prediction part of the model, making it work in a supervised fashion. Takes
        as input the embedding.

        Args:
            x (torch.Tensor): Patient level embedding.

        Returns:
            array-like: Prediction for input batch.
        """
        # 3. Apply final classifier
        x = x.relu()
        x = self.lin_pred1(x)
        x = x.relu()
        x = self.lin_pred2(x)

        if self.sigmoid:
            x = torch.sigmoid(x)
        return x

    def forward(self, x, edge_index, batch):
        """Forward pass, sequentially running the embedding and the prediction part.

        Args:
            x (torch.Tensor): Node features. Usually RNAseq expression.
            edge_index (torch.Tensor): Edge list of the PPI we are using.
            batch (torch.Tensor):  Batch vector which assigns each node to a specific
                batch number.

        Returns:
            array-like: Predictions in input batch.
        """

        x_emb = self.embed_nodes(x, edge_index, batch)
        if self.unsupervised:
            y_hat = self.decoder(x_emb)
            return y_hat
        y_hat = self.pred(x_emb)
        return y_hat

    def run_epoch(self, dataloader: DataLoader, training: bool):
        """Run training epoch.

        Args:
            dataloader (DataLoader): dataloader.
            training (bool):  training / inference mode.

        Returns:
            Union[np.array, np.array]: Predictions+loss.
        """

        if training:
            self.train()
        else:
            self.eval()

        epoch_predictions = []
        epoch_losses = []
        epoch_labels = []
        with torch.inference_mode(mode=not training):
            for data in dataloader:
                data = data.to(self.device)
                node_features, edge_index, batch, labels = (
                    data.x.reshape(-1, 1),
                    data.edge_index,
                    data.batch,
                    data.y.clone().detach(),
                )
                predictions = self.forward(node_features, edge_index, batch)

                epoch_predictions.extend(predictions.tolist())
                epoch_labels.extend(labels.tolist())
                if training:
                    self.optimizer.zero_grad()
                    if self.unsupervised:
                        loss = self.criterion(predictions.reshape(-1, 1), node_features)
                    else:
                        loss = self.criterion(predictions, labels.view(-1, 1))
                    loss.backward()
                    self.optimizer.step()
                    epoch_losses.append(loss.detach().cpu().numpy())
        # c_index = compute_cindex(np.array(epoch_labels), np.array(epoch_predictions))
        # logger.info(f"Epoch c-index : {c_index}.")
        concatenated_predictions = np.concatenate(epoch_predictions, axis=0)
        epoch_losses_array = np.array(epoch_losses)
        return concatenated_predictions, epoch_losses_array

    def fit(
        self,
        X,
        metrics_suffix: str = None,
        df_labels: pd.DataFrame = None,
        edge_index: torch.Tensor = None,
        clusters: torch.Tensor = None,
        reduced_mashups=None,
        finetuning_rpz=False,
        **kwargs,
    ) -> Self:
        """Fit the model according to the given training data and auxiliary labels.

        Args:
            X (pd.DataFrame): training data.
            metrics_suffix (str): repeat/split information to be logged with the loss.
            df_labels (pd.DataFrame): contains all df columns except RNA-Seq data,
                therefore including the auxiliary labels for each head.
            edge_index (torch.Tensor): edges of the graph we are using.
            clusters (torch.Tensor): Clusters to use for the pooling.
            reduced_mashups (torch.Tensor): Gene positional embeddings using mashup.
            finetuning_rpz (bool): Boolean value if we are in finetuning_rpz mode. If
                True we do not call _init_models().
            **kwargs: variable number of keyword arguments, to be compatible with
                the method from parent class.


        Raises:
            ValueError: when df_labels is not a valid pandas DataFrame

        Returns:
            self
        """
        del kwargs
        edge_index, clusters, genes = load_graph(
            genes=X.columns.values.tolist(),
            n_clusters=self.repr_dim,
            **self.graph_loading,
        )
        logger.info("Graph info")
        logger.info(f"Number of genes in graph: {len(genes)}")
        logger.info(f"Number of edges in graph: {edge_index.shape[1]}")
        logger.info(f"Number of clusters: {len(clusters.unique())}")

        self.edge_index = edge_index.to(self.device)
        self.clusters = clusters.to(self.device)
        self.n_clusters = len(clusters.unique())
        self.genes = genes
        if self.unsupervised:
            self.labels = np.zeros(X.shape[0])
        else:
            self.labels = df_labels.OS.values
        if reduced_mashups is not None:
            self.reduced_mashups = torch.Tensor(reduced_mashups.values).to(self.device)
        else:
            self.reduced_mashups = None

        X = X.loc[:, self.genes]
        self.input_size = X.shape[1]
        if not finetuning_rpz:
            self._init_models()

        datasets = [
            Data(
                x=torch.Tensor(X.values[i]),
                edge_index=self.edge_index,
                y=torch.Tensor([self.labels[i]]).long(),
                device=self.device,
                pos=self.reduced_mashups,
            )
            for i in range(X.shape[0])
        ]
        dataloader = DataLoader(datasets, batch_size=self.batch_size, shuffle=True)

        self.train()

        if not finetuning_rpz:
            self.optimizer = self.optimizer(self.parameters(), lr=self.learning_rate)
        else:
            self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

        for epoch in range(self.num_epochs):
            _, epoch_losses = self.run_epoch(dataloader, training=True)
            self.train_loss.append(np.mean(epoch_losses))
            mlflow.log_metric(f"Loss {metrics_suffix}", self.train_loss[-1], step=epoch)

        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Encode the data using the fitted model.

        Args:
            X (pd.DataFrame): training data.

        Returns:
            np.ndarray: encoded data.
        """
        X = X.loc[:, self.genes]
        datasets = [
            Data(
                x=torch.Tensor(X.values[i]),
                edge_index=self.edge_index,
                device=self.device,
                pos=self.reduced_mashups,
            )
            for i in range(X.shape[0])
        ]
        dataloader = DataLoader(datasets, batch_size=self.batch_size, shuffle=False)

        self.eval()

        features = []
        for data in dataloader:
            # https://stackoverflow.com/questions/59560043/what-is-the-difference-between-model-todevice-and-model-model-todevice
            data = data.to(self.device)
            data_batch_reconstructed = self.embed_nodes(
                data.x.reshape(-1, 1), data.edge_index, data.batch
            )
            features.append(data_batch_reconstructed.detach())

        features = torch.cat(features, dim=0).cpu().numpy()

        return features

    def fit_transform(
        self,
        X: pd.DataFrame,
        df_labels: pd.DataFrame,
        metrics_suffix: str,
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
        _ = self.fit(X, metrics_suffix, df_labels, **kwargs)
        return self.transform(X)
