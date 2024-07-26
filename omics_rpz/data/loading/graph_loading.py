"""Create the graph data used for GNNs."""
import random

import networkx as nx
import networkx.algorithms.community as nx_comm
import torch
from torch_geometric.utils import from_networkx

from omics_rpz.data.loading.msigdb_pathways import load_pathways
from omics_rpz.data.loading.string_embeddings import (
    gene_intersection_with_string,
    load_string,
)


def load_graph(
    genes: list,
    string_threshold: float = 0.95,
    clustering: str = 'louvain',
    cluster_resolution: int = 2000,
    n_clusters: int = 200,
    permute_gene_names: bool = False,
    clustering_seed: int = 42,
    pathway_source: str = 'KEGG',
):
    """This function loads the graph data used for GNNs models downstream. The graph is
    loaded from the STRNG database.

    Args:
        genes (List): List of genes that we want to keep in the graph. This list can
        come from a previous filtering step.
        string_threshold (float): Score threshold of the edges we want to keep in the
        graph. This score relates to the top quantile we want to keep.
        clustering (str):  Method that we use to perform the clustering on the graph.
        Defaults to 'louvain'.
        cluster_resolution (int, optional): Parameter to be passed on if the clustering
        method is Louvain. This parameter affects the resulting number of clusters.
        Larger resolution leads to smaller and more numerous clusters. Defaults to 2000.
        n_clusters (int, optional): Parameter to decide number of clusters we want, if
        the clustering method allows it. Defaults to 200.
        permute_gene_names (bool, optional): Allows to permute gene names to have a
        random graph. Defaults to False.
        clustering_seed: int = 42, Random seed for clustering.
        pathway_source: str = 'KEGG', Database used for pathway clustering.


    Raises:
        NotImplementedError: Error thrown if the clustering method is not implemented.

    Returns:
        Union[Tensor, Tensor, List]: tensor of the edges that form the graph + cluster
        assignments + list of genes that are in the graph.
    """
    string_full, genes_aliases = load_string()
    thr = string_full['combined_score'].quantile(string_threshold)
    string_full = string_full.query(f"combined_score > {thr}")

    genes, string = gene_intersection_with_string(
        rnaseq_variables=genes,
        string_db=string_full,
        genes_aliases=genes_aliases,
    )

    graph_nx = nx.from_pandas_edgelist(
        string,
        source='protein1',
        target='protein2',
    ).to_undirected()

    largest_cc = max(nx.connected_components(graph_nx), key=len)
    graph_nx = graph_nx.subgraph(largest_cc).copy()

    genes = list(graph_nx.nodes)
    genes_to_int = dict(zip(genes, range(len(genes))))

    if clustering == 'louvain':
        partition = nx_comm.louvain_communities(
            graph_nx, resolution=cluster_resolution, seed=clustering_seed
        )
        clusters = torch.zeros(graph_nx.number_of_nodes())
        for ind, cluster in enumerate(partition):
            for node in cluster:
                clusters[genes_to_int[node]] = ind
        clusters = clusters.long()

    elif clustering == 'fluid':
        partition = nx_comm.asyn_fluidc(graph_nx, k=n_clusters, seed=clustering_seed)
        clusters = torch.zeros(graph_nx.number_of_nodes())
        for ind, cluster in enumerate(partition):
            for node in cluster:
                clusters[genes_to_int[node]] = ind
        clusters = clusters.long()

    elif clustering == 'pathways':
        pathways = load_pathways(genes, pathway_source)
        clusters = torch.zeros(graph_nx.number_of_nodes())
        for ind, cluster in enumerate(pathways):
            for node in cluster:
                clusters[genes_to_int[node]] = ind
        cluster_number_change = dict(
            zip(clusters.unique().tolist(), range(len(clusters.unique())))
        )
        clusters_new = []
        for c in clusters.tolist():
            clusters_new.append(cluster_number_change[c])
        clusters = torch.tensor(clusters_new).long()

    else:
        raise NotImplementedError

    edge_index = from_networkx(graph_nx).edge_index

    if permute_gene_names:
        random.shuffle(genes)

    return edge_index, clusters, genes
