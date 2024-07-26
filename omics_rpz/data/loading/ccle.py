"""Loading functions for CCLE data."""
import os
import re
import time
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from omics_rpz.constants import OMICS_RPZ_PATH, PATH_CCLE, PATH_PROTEIN_CODING_GENES


@lru_cache
def load_rnaseq_ccle(normalization_rnaseq: str = "TPM") -> pd.DataFrame:
    """Loading function for CCLE RNASeq data.

    Args:
        normalization_rnaseq (str, optional): RNAeq normalization, changes loading path.
        Defaults to "TPM".

    Returns:
        pd.DataFrame: RNAseq features dataframe
    """
    if normalization_rnaseq not in ["RAW", "TPM"]:
        raise ValueError(
            "Normalization specified doesn't exist, check your spelling or avaibility."
            f" Received {normalization_rnaseq}, should be `RAW` or `TPM`."
        )
    if normalization_rnaseq == "RAW":
        ccle_expr = load_rnaseq_ccle_raw()
    if normalization_rnaseq == "TPM":
        ccle_expr = load_rnaseq_ccle_tpm()
    return ccle_expr


@lru_cache
def load_rnaseq_ccle_tpm() -> pd.DataFrame:
    """Load CCLE RNASeq data with TPM normalization.

    It is important to note that RNAseq TPM data from DepMap is already log-scaled with
    log2(X+1), therefore here we convert the TPM data with exp2(x) - 1 so that the TPM
    data has the same pre-processing as in other datasets (TCGA, PharmacoDB).

    Returns
    -------
    pd.DataFrame
        RNAseq features dataframe
    """
    path_processed = Path(PATH_CCLE) / "CCLE_expression_tpm.parquet"
    if path_processed.exists():
        ccle_expr = pd.read_parquet(path_processed)
    else:
        # load TPM logscaled data
        path = Path(PATH_CCLE) / "OmicsExpressionProteinCodingGenesTPMLogp1.csv"
        ccle_expr = pd.read_csv(path)
        ccle_expr.columns.values[0] = "DepMap_ID"
        # convert gene columns to float32
        ccle_expr = (
            ccle_expr.set_index("DepMap_ID")
            .apply(lambda x: np.exp2(x) - 1)
            .astype("float32")
        )
        # Clean column names to have gene symbol only
        ccle_expr.rename(
            columns=lambda x: re.sub(r"[\(\[ ].*?[\)\]]", "", x), inplace=True
        )
        ccle_expr.to_parquet(path_processed, engine="pyarrow", compression="brotli")

    # Check number of cell lines
    assert ccle_expr.shape[0] == 1408

    return ccle_expr


@lru_cache
def load_rnaseq_ccle_raw() -> pd.DataFrame:
    """Loading function for raw CCLE RNASeq data.

    Returns:
        pd.DataFrame: RNAseq features dataframe
    """
    path = Path(PATH_CCLE) / "CCLE_expression_raw_counts_to_use.parquet"
    if os.path.exists(path):
        ccle_expr = pd.read_parquet(path)
    else:
        # Load full raw expression matrix
        path = Path(PATH_CCLE) / "OmicsExpressionGenesExpectedCountProfile.csv"
        ccle_expr = pd.read_csv(path, index_col=0)

        # Load CCLE data omics profiles identifiers
        path = Path(PATH_CCLE) / "OmicsDefaultModelProfiles.csv"
        profiles = pd.read_csv(path)
        profiles = profiles[profiles["ProfileType"] == "rna"]
        profiles = profiles.set_index("ProfileID")

        # Load protein coding genes
        protein_coding_genes = pd.read_parquet(PATH_PROTEIN_CODING_GENES)
        protein_coding_genes = protein_coding_genes.set_index("ensembl_gene_id")

        # The expression matrix is indexed on the profile identifier
        # 1. we only select profiles that are mapped to a cell line ID
        # 2. We convert the profile ID to the Cell line ID in the expression matrix
        ccle_expr = ccle_expr.loc[list(profiles.index)]
        ccle_expr["DepMap_ID"] = list(profiles["ModelID"].values)
        ccle_expr = ccle_expr.set_index("DepMap_ID")

        # Convert gene names in ccle_exprression matrix from Symbol to Ensembl ID
        genes = ccle_expr.columns
        genes_ens = []
        for gene in genes:
            # Convert all gene symbols to Ensembl ID
            if '(' in gene:
                genes_ens.append(gene.split(" ")[1][1:-1])
            else:
                genes_ens.append(gene)

        # Intersection with protein coding genes (Ensembl ID)
        common_genes_ens = list(set(protein_coding_genes.index) & set(genes_ens))
        ccle_expr.columns = genes_ens
        ccle_expr = ccle_expr[common_genes_ens]

        # Convert back to Symbol and sort column names
        genes_cols = list(protein_coding_genes.loc[ccle_expr.columns]["symbol"].values)
        ccle_expr.columns = genes_cols
        genes_cols.sort()

        ccle_expr = ccle_expr[genes_cols]
        # Save
        path = Path(PATH_CCLE) / "CCLE_expression_raw_counts.parquet"
        ccle_expr.to_parquet(path)
        # Note: We keep saving this version in case we need it for different analysis
        # but we don't use it by default.

        # Intersect with TPM latest data
        ccle_expr_tpm = pd.read_parquet(Path(PATH_CCLE) / "CCLE_expression_tpm.parquet")

        depmap_ids_intersect = list(set(ccle_expr_tpm.index) & set(ccle_expr.index))
        genes_cols_intersect = list(set(ccle_expr_tpm.columns) & set(ccle_expr.columns))

        ccle_raw_to_save = ccle_expr.loc[depmap_ids_intersect, genes_cols_intersect]

        # Save filtered version
        ccle_raw_to_save.to_parquet(
            Path(PATH_CCLE) / "CCLE_expression_raw_counts_to_use.parquet"
        )
        ccle_expr = ccle_raw_to_save

    # Check number of cell lines
    assert ccle_expr.shape[0] == 1408

    return ccle_expr


@lru_cache
def load_labels_ccle() -> pd.DataFrame:
    """Loading function for DeepDEP essentiality scores.

    Returns:
        pd.DataFrame: Labels for Essentiality Prediction in form of a matrix
            Size (N_CELL_LINES, N_GENES)
    """

    labels_path = Path(PATH_CCLE) / "CRISPR_gene_dependency.parquet"

    gene_dependencies = pd.read_parquet(labels_path)
    # .set_index("DepMap_ID")

    # Clean column names to have gene symbol only
    gene_dependencies.columns = [
        re.sub(r"[\(\[ ].*?[\)\]]", "", i) for i in gene_dependencies.columns
    ]

    # Cell line ACH-001740 has missing dependencies
    gene_dependencies = gene_dependencies.dropna(axis=0)

    return gene_dependencies


def load_cancer_type_ccle() -> pd.Series:
    """Load cancer type labels for CCLE Data.

    Returns:
        pd.Series: Series of cancer type of each cell line sample
    """

    path = Path(PATH_CCLE) / "sample_info.csv"
    ccle_metadata = pd.read_csv(path)
    ccle_metadata = ccle_metadata.set_index("DepMap_ID")

    path = Path(PATH_CCLE) / "public_22q2_new_annotations.csv"
    annotations = pd.read_csv(path)
    annotations = annotations.set_index("DepMap_ID")

    # Select the subset of cell lines from the 22Q4 release
    annotations = annotations.loc[list(ccle_metadata.index)]

    labels = annotations.primary_disease

    return labels


@lru_cache
def load_fingerprints() -> pd.DataFrame:
    """Loading function for ground truth essentiality scores.

    Returns:
        pd.DataFrame: Fingerprints signatures
            Size (N_GENES, FINGERPRINTS_DIM)
    """
    fingerprints_path = Path(PATH_CCLE) / "fingerprint_T.csv"
    fingerprints_df = pd.read_csv(fingerprints_path, index_col=0)
    fingerprints_df.index.rename('DepOI', inplace=True)
    return fingerprints_df


@lru_cache
def load_depoi() -> pd.DataFrame:
    """Loading function for the genes that will be predicted in the gene essentiality
    task.

    Returns:
        pd.DataFrame: List of genes in study and annotations
    """
    depoi_path = Path(PATH_CCLE) / "DepIoGenes_to_use.csv"
    depoi = pd.read_csv(depoi_path, index_col=0)
    return depoi


@lru_cache
def load_ccle(
    normalization_rnaseq: str = "TPM", task: str = 'essentiality'
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Function for loading both labels and expression + aligning datasets.

    Args:

        normalization_rnaseq (str, optional): RNAeq normalization, changes loading path.
        Defaults to "TPM".
        task (str): Task for which we are loading ccle. Depending on the task, we will
        load different labels. If the task is 'alignment', the label will be cancer
        type.


    Returns:
        tuple[pd.DataFrame, pd.DataFrame]:
            - CCLE RNAseq dataset
            - CCLE Labels as a matrix
    """
    logger.info("Start loading CCLE data")
    t_1 = time.time()
    ccle_expr = load_rnaseq_ccle(normalization_rnaseq)
    if task == 'alignment':
        ccle_labels = load_cancer_type_ccle()
    else:
        ccle_labels = load_labels_ccle()

    ccle_labels = ccle_labels.loc[ccle_labels.index.isin(ccle_expr.index)]
    ccle_expr = ccle_expr.loc[ccle_expr.index.isin(ccle_labels.index)]
    t_2 = time.time()
    logger.info(f"Loading CCLE data took {t_2-t_1:.2f} seconds")
    return (ccle_expr, ccle_labels)


@lru_cache
def load_sample_info() -> pd.DataFrame:
    """Load metadata for all of DepMap's cancer cell lines.

    Returns:
        pd.DataFrame: DataFrame containing the metadata
    """
    path_sample_info = Path(PATH_CCLE) / "sample_info_22q4.csv"
    df_sample_info = pd.read_csv(path_sample_info, index_col=0)
    df_sample_info = df_sample_info.set_index("DepMap_ID")

    return df_sample_info


# Maxsize defaults to 128. Reduced if nbr_of_depoi could vary in an HP eventually?
@lru_cache(maxsize=32)
def load_data_for_essentiality_pred(
    normalization_rnaseq: str = "TPM", nbr_of_depoi: int = 59
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Function for loading CCLE RNASeq, gene fingerprints for selected genes,
    corresponding essentiality scores and metadata from the cell lines.

    Args:
        normalization_rnaseq (str, optional): RNAeq normalization, changes loading path.
        Defaults to "TPM".
        nbr_of_depoi (int, optional): Number of DepOi to select. Select the most variant
            first.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
            - CCLE RNAseq dataset
            - Genes Fingerprints
            - CCLE Labels as a matrix
            - Metadata from the cell lines
    """
    ccle_rnaseq, ccle_labels = load_ccle(normalization_rnaseq)
    depoi = load_depoi()
    fin = load_fingerprints()

    # load metadata and align to RNA-seq info
    sample_info = load_sample_info()
    sample_info = sample_info.loc[sample_info.index.isin(ccle_rnaseq.index)]
    # Select subset of DepOI for prediction and filter all datasets accordingly.

    new_depoi = (
        depoi.loc[depoi.index.isin(list(ccle_labels.columns))]
        .sort_values(by="Std in original dep score across 278 CCLs", ascending=False)
        .iloc[:nbr_of_depoi]
        .index.values
    )

    fingerprints_depoi = fin.loc[new_depoi]
    dependencies_depoi = ccle_labels[new_depoi]

    # add prefix to avoid name conflicts with gene expression
    dependencies_depoi = dependencies_depoi.add_prefix('essentiality_')
    fingerprints_depoi.index = "essentiality_" + fingerprints_depoi.index.astype(str)
    return ccle_rnaseq, fingerprints_depoi, dependencies_depoi, sample_info


def load_tcga_ccle_mapping() -> dict[str, str]:
    """Load the mapping between TCGA and CCLE cancer types.

    Returns:
        dict: dictionnary with label mappings between TCGA and CCLE,
        with TCGA labels as keys and CCLE labels as values.
    """

    mapping_path = Path(OMICS_RPZ_PATH) / "CCLE_TCGA_label_mapping.csv"
    mapping = pd.read_csv(mapping_path)
    tcga_to_ccle = dict(zip(mapping["Study Abbreviation"], mapping["CCLE lineages"]))

    return tcga_to_ccle
