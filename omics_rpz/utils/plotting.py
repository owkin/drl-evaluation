"""Module with plotting functions."""

import pathlib

import matplotlib.pyplot as plt
import seaborn as sns
import umap.umap_ as umap


def draw_umap(
    pd_x,
    n_neighbors=15,
    min_dist=0.1,
    n_components=2,
    metric="euclidean",
    seed=42,
    filename="umap_plot",
):
    """Plot a umap projecion of a dataset.

    Parameters
    ----------
    pd_x: pd.DataFrame
        Dataframe being drawn
    n_neighbors : int, optional
        Parameter of UMAP, by default 15
    min_dist : float, optional
        Parameter of UMAP, by default 0.1
    n_components : int, optional
        The number of dimensions of the UMAP either 2 or 3, by default 2
    metric : str, optional
        The metric to use for the umap computation, by default "euclidean"
    seed : int, optional
        The random state given to umap, by default 42
    filename : str, optional
        Name of the file in the current script's directory where the plot will be saved.
    """
    print(f"Computing UMAP, NN {n_neighbors}, min d {min_dist}, ncomp {n_components}")
    u = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=n_components,
        metric=metric,
        random_state=seed,
    ).fit_transform(pd_x)
    colors = sns.color_palette("hls", 6)

    if n_components == 2:
        fig = plt.figure()
        axes = fig.add_subplot()
        axes.scatter(
            u[:, 0],
            u[:, 1],
            color=colors[0],
            s=1,
            label="X",
        )
        plt.xlabel("Umap dimension 1")
        plt.ylabel("Umap dimension 2")

    if n_components == 3:
        fig = plt.figure()
        axes = fig.add_subplot(projection="3d")
        axes.scatter(
            u[:, 0],
            u[:, 1],
            u[:, 2],
            color=colors[0],
            s=1,
            label="X",
        )
    plt.legend()
    out = pathlib.Path(__file__).parents[2].joinpath("outputs/" + filename)
    plt.savefig(out, bbox_inches="tight")
