"""Module implementing preprocessing steps and basic feature selection for RNA-seq
data."""
from typing import Union

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.preprocessing import FunctionTransformer, MinMaxScaler, StandardScaler

from omics_rpz.constants import PATH_PROTEIN_CODING_GENES

from .scalers import MedianRatioScaler

SCALERS = {
    "min_max": MinMaxScaler(),
    "mean_std": StandardScaler(with_std=True),
    "mean": StandardScaler(with_std=False),
    "median_of_ratios": MedianRatioScaler(),
    "identity": FunctionTransformer(func=None),
}


class RNASeqPreprocessor:
    """Preprocesses normalized RNAseq data.
    Transformations:
        1. Select genes in given gene list or protein-coding genes, if specified.
        2. Apply log(x+1) if log_scaling == 'pre_gf', or normalize data with provided
        scaler if pre_normalize == True.
        3. If gene_filtering != None, compute ranks of genes according to gene_filtering
        method.
        4. Select top max_genes genes based on their ranks.
        3. If log_scaling == True, apply log(x+1).
        4. Data is centered with the chosen scaling method.

    Parameters
    ----------
    scaling_method: str = "min_max"
        Scaling method to apply after the log transformation (min_max, mean_std, mean).
    max_genes: int = -1
        Number of genes with highest variance to keep. Keep all genes if max_genes <= 0.
    log_scaling: Union[str, bool] = True
        Either True, False or 'pre_gf'. If True, apply a log transformation to the data
        (x -> log(x+1)), if 'pre_gf', applies it before selecting genes with the
        gene_filtering method.
    gene_list : Union[str, list[str]] = None
        Path to CSV file containing a list of genes to be considered, or list of genes.
    select_protein_coding : bool = False
        If true, filter on genes coding for proteins.
    gene_filtering: Union[str, list[str]] = 'variance'
        Selecting the method in which the list of top genes is generated
    pre_normalize: bool = False
        If True, normalizes the data before selecting genes with the gene_filtering
        method.
    cohort_path : str = 'None'
        The path of the GTEX cohort to be considered together with the TCGA one
    Attributes
    ----------
    scaler:
        Scaler class selected according to the scaling_method string
    columns_to_keep: pd.Index
        Names of genes which are kept.
    gene_list_valid: list[str]
        Used when gene_list is not None; contains only genes that appear in the given df

    Raises
    ------
    AssertionError
        An AssertionError if the scaler method is unknown.
    AssertionError
        An AssertionError if the log scaling parameter is unknown.
    NotImplementedError
        A NotImplementedError if no healthy reference tissue is given for a
        distance-based gene selection method.
    """

    def __init__(
        self,
        scaling_method: str = "min_max",
        max_genes=-1,
        log_scaling: Union[str, bool] = True,
        pre_normalize=False,
        gene_list: Union[str, list[str]] = "",
        select_protein_coding: bool = False,
        gene_filtering: str = 'variance',
    ):
        self.scaling_method = scaling_method
        self.max_genes = max_genes
        self.select_protein_coding = select_protein_coding
        if gene_list:
            if isinstance(gene_list, str):
                # Path to file containing the list of genes
                gene_list = pd.read_csv(gene_list).iloc[:, 0].tolist()
            self.gene_list = gene_list
            if select_protein_coding:
                protein_coding_table = pd.read_parquet(PATH_PROTEIN_CODING_GENES)
                self.gene_list = np.intersect1d(
                    self.gene_list, protein_coding_table['symbol'].values
                )
        else:
            self.gene_list = None
            if select_protein_coding:
                protein_coding_table = pd.read_parquet(PATH_PROTEIN_CODING_GENES)
                self.gene_list = protein_coding_table['symbol'].values

        try:
            self.scaler = SCALERS[self.scaling_method]
        except KeyError as exc:
            raise AssertionError(
                f"Scaling method must be {SCALERS.keys()}, got '{self.scaling_method}'"
            ) from exc

        if gene_filtering and isinstance(gene_filtering, str):
            self.gene_filtering = [gene_filtering]
        else:
            self.gene_filtering = gene_filtering
        if log_scaling not in [True, False, 'pre_gf']:
            raise AssertionError(
                "log scaling must be either 'True' 'False' or 'pre_gf', got"
                f" {log_scaling}."
            )
        self.log_scaling = log_scaling
        self.columns_to_keep = None
        self.pre_normalize = pre_normalize

    def fit(self, X: pd.DataFrame):
        """Compute gene list and fit the scaler used for later transformation.

        Parameters
        ----------
        X : pd.DataFrame
            RNA-seq data (untransformed)

        Returns
        -------
        self

        Raises
        ------
        NotImplementedError
            Raises error for invalid gene selection methods
        """
        # If the processor has already been used in pretraining or DA for instance skip.
        if self.columns_to_keep is None:
            # Do gene filtering on pre specific gene list.
            if self.gene_list is not None:
                X = X.loc[:, X.columns.intersection(self.gene_list)]
                # Just logging.
                if self.select_protein_coding:
                    logger.info(
                        f"Selecting {X.shape[1]} genes based on"
                        + (" protein-coding" if self.select_protein_coding else "")
                        + " genes in gene list."
                    )

            # Do specific filtering like variance or wasserstein.
            if self.gene_filtering and self.max_genes > 0:
                if self.log_scaling == 'pre_gf':
                    X = X.apply(np.log1p)

                gene_ranks = self.rank_genes(X)
                logger.info(
                    f"Selecting {self.max_genes} genes based on"
                    f" {self.gene_filtering} filtering"
                    + (", pre-log-scaled" if self.log_scaling == 'pre_gf' else "")
                    + (", pre-normalized" if self.pre_normalize else "")
                    + "."
                )
                # Save columns to keep for future fit_transform.
                # The sort values is going to "revert" the second argsort in the
                # rank_genes function. You will have the index of the highest variant
                # columns: [index_of_most_variant_genes ,.._second_most_variant, etc]
                # Sorting the columns to ensure the order is consistent.
                self.columns_to_keep = np.sort(
                    gene_ranks.sort_values()[: self.max_genes].index
                )
            else:
                # No gene filtering so you take all columns of X.
                # Save columns to keep for future fit_transform.
                # Sorting the columns to ensure the order is consistent.
                self.columns_to_keep = np.sort(X.columns)

        if not set(self.columns_to_keep) <= set(X.columns):
            logger.warning('X does not have all the columns to keep')
            self.columns_to_keep = X.columns.intersection(self.columns_to_keep)
        X = X[self.columns_to_keep]

        # log transform
        if self.log_scaling is True:
            X = X.apply(np.log1p)

        # train scaler
        self.scaler.fit(X)

        return self

    def rank_genes(self, X: pd.DataFrame) -> pd.DataFrame:
        """Rank genes according to each method contained in self.gene_filtering, then
        take the minimum rank of each gene across methods.

        Parameters
        ----------
        X : pd.DataFrame
            RNA-seq data (Filtered on gene list, possibly log-scaled)

        Returns
        -------
        pd.Series


        Raises
        ------
        NotImplementedError
            Raises error for invalid gene selection methods
        """
        gene_ranks = {}
        for gene_filtering in self.gene_filtering:
            if gene_filtering == 'variance':
                variances = np.var(X, axis=0)
                # Ranks, highest variance first
                # sorts the elements of the "variances" array in ascending order and
                # returns the indices that would sort the array. Then redo an argsort
                # to have the indices of the largest variances ranked.
                ranks = (-variances).argsort().argsort()
                # [rank_of_gene1,rank_of_gene2]
                # This is needed to combine multiple filtering together.
            # Add ranks to results dict
            gene_ranks[gene_filtering] = ranks

        # Perform union of methods : take min ranks across methods
        min_gene_ranks = pd.concat(
            [ranks for _, ranks in gene_ranks.items()], axis=1
        ).apply(min, axis=1)

        return min_gene_ranks

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Perform gene selection and data scaling with the chosen method.

        Parameters
        ----------
        X : pd.DataFrame
            RNA-seq data (untransformed)

        Returns
        -------
        pd.DataFrame
            Transformed RNA-seq data after gene selection and scaling
        """
        X = X.loc[:, self.columns_to_keep]

        if self.log_scaling:
            X = X.apply(np.log1p)  # type: ignore

        # Keep X a pd DataFrame after scaling, not a np ndarray
        X.loc[:, X.columns] = self.scaler.transform(X)

        return X

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Compute gene list to be kept, fit scaler, and apply data transformation.

        Parameters
        ----------
        X : pd.DataFrame
            RNA-seq data (untransformed)

        Returns
        -------
        pd.DataFrame
            Transformed data after gene selection and scaling
        """
        self.fit(X)
        return self.transform(X)
