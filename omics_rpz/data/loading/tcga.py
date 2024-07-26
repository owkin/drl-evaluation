"""Loading functions for TCGA data."""
from functools import lru_cache
from pathlib import Path
from typing import Union

import pandas as pd
from loguru import logger

from omics_rpz.constants import ALL_TCGA_COHORTS, TCGA_PATHS
from omics_rpz.utils import encode, handle_nan_values


def load_clinical(
    cohort: str,
) -> pd.DataFrame:
    """Load clinical data for a given TCGA cohort.

    Args:
        cohort (str): Name of the cohort

    Returns:
        pd.DataFrame: Clinical data
    """

    # Load TCGA-CDR dataset
    cdr_path = Path(TCGA_PATHS["CDR"])
    # df_cdr = pd.read_excel(cdr_path, index_col=0)
    df_cdr = pd.read_parquet(cdr_path)

    # Rename columns
    to_rename = {
        "bcr_patient_barcode": "patient_id",
        "type": "cancer_type",
        "age_at_initial_pathologic_diagnosis": "age",
        "gender": "gender",
        "race": "race",
        "ajcc_pathologic_tumor_stage": "stage",
        "histological_grade": "grade",
        "OS": "OS_event",
        "OS.time": "OS_time",
        "PFI": "PFS_event",
        "PFI.time": "PFS_time",
    }
    df_cdr = df_cdr.rename(columns=to_rename)

    # Replace invalid values by NaN
    df_cdr = df_cdr.replace("[Not Available]", None)

    # Compute OS and PFS
    df_cdr.loc[~(df_cdr.OS_time.isna()), "OS"] = df_cdr.loc[
        ~(df_cdr.OS_time.isna()), :
    ].apply(lambda row: row.OS_time if row.OS_event == 1 else -row.OS_time, axis=1)
    df_cdr.loc[~(df_cdr.PFS_time.isna()), "PFS"] = df_cdr.loc[
        ~(df_cdr.PFS_time.isna()), :
    ].apply(lambda row: row.PFS_time if row.PFS_event == 1 else -row.PFS_time, axis=1)

    # Only keep relevant columns
    to_keep = [
        "patient_id",
        "cancer_type",
        # "age",
        "gender",
        "race",
        "stage",
        # "grade",
        "OS",
        "PFS",
    ]
    df_cdr = df_cdr[to_keep]

    # Load clinical data
    path_to_clinical = TCGA_PATHS["CLINICAL"]
    path = Path(path_to_clinical) / f"TCGA-{cohort}_clinical_v2.tsv.gz"
    df_clin = pd.read_csv(path, sep="\t")

    # Only keep relevant columns
    to_keep = [
        "patient",
        "sample",
        "barcode",
        "age_at_diagnosis",
        "high_grade",
        # "Purety_CCF",
        "center",
        "TSS",
    ]
    to_keep = (
        to_keep
        + [col for col in df_clin.columns if "decoupler" in col]
        + [col for col in df_clin.columns if "cell_fraction" in col]
    )
    df_clin = df_clin[to_keep]
    df_clin['TSS'] = df_clin['TSS'].astype('str')
    # some centers like '30' can be int or str, harmonize

    # Rename columns
    to_rename = {
        "patient": "patient_id",
        "sample": "sample_id",
        "barcode": "barcode",
        "age_at_diagnosis": "age",
        "high_grade": "grade",
        # "Purety_CCF": "purity",
        "center": "center",
        "TSS": "TSS",
    }
    df_clin = df_clin.rename(columns=to_rename)

    # Merge dataframes
    df_cdr_clin = pd.merge(
        df_cdr, df_clin, how="inner", left_on="patient_id", right_on="patient_id"
    )

    # Handle NaN values
    df_cdr_clin = handle_nan_values(df_cdr_clin)

    # Encode categorical variables
    categorical_vars = ["gender", "race", "treatment", "stage", "grade"]
    df_cdr_clin, _ = encode(df_cdr_clin, categorical_vars)

    to_keep = [
        "barcode",
        "sample_id",
        "patient_id",
        "cancer_type",
        "age",
        "gender",
        "race",
        "stage",
        "grade",
        # "purity",
        "OS",
        "PFS",
        "center",
        "TSS",
    ]
    to_keep = (
        to_keep
        + [col for col in df_clin.columns if "decoupler" in col]
        + [col for col in df_clin.columns if "cell_fraction" in col]
    )
    df_cdr_clin = df_cdr_clin[to_keep]

    return df_cdr_clin


def load_rnaseq(
    cohort: str,
    normalization: str = "NORM",  # raw, tpm, norm, rpkm (without batch correction),
    # or fpkm (with batch correction)
    batch_corrected: bool = False,
    filter_tumor: bool = False,
    unique_patients: bool = False,
) -> Union[pd.DataFrame, list]:
    """Load RNASeq data for a TCGA cohort.

    Args:
        cohort (str): Name of the TCGA cohort.
        normalization (str, optional): Normalization type from: RAW, RPKM, TPM, NORM,
        FPKM-UQ. Defaults to "NORM".
        batch_corrected (bool, optional): Whether to use combat-seq batch corrected data
        filter_tumor (bool, optional): If true, returns only tumoral samples.
        unique_patients (bool, optional): If true, returns unique patients instead of
        unique barcodes.

    Returns:
        Union[pd.DataFrame, list]: Dataframe with RNAseq data and list of RNAseq vars
    """

    norm_dict = {
        "RAW": "raw",
        "RPKM": "rpkm",
        "TPM": "tpm",
        "NORM": "norm",
        "FPKM": "fpkm",
    }
    normalization = norm_dict[normalization]
    if batch_corrected:
        rnaseq_path = (
            Path(TCGA_PATHS["BATCH_CORRECTED"])
            / cohort
            / f"Counts_{normalization}_float32.parquet"
        )
        metadata_path = (
            Path(TCGA_PATHS["BATCH_CORRECTED"])
            / cohort
            / f"{cohort}_filtered_metadata.tsv"
        )
    else:
        rnaseq_path = (
            Path(TCGA_PATHS["RNASEQ_PARQUET"])
            / cohort
            / "Data"
            / f"Counts_{normalization}_float32.parquet"
        )
        metadata_path = Path(TCGA_PATHS["RNASEQ"]) / cohort / "Data" / "metadata.tsv.gz"

    # Load data
    df_rna = pd.read_parquet(rnaseq_path)
    df_rna = df_rna.set_index("Hugo")

    # Load metadata
    df_metadata = pd.read_csv(metadata_path, sep="\t", index_col="external_id")
    df_metadata["tcga.cgc_file_last_modified_date"] = pd.to_datetime(
        df_metadata["tcga.cgc_file_last_modified_date"], utc=True
    )
    df_metadata['patient_id'] = df_metadata['tcga.tcga_barcode'].map(lambda x: x[:12])

    def get_sample_type(sample):
        if sample < 10:
            return "TUMOR"
        if sample < 20:
            return "NORMAL"
        return "CONTROL"

    df_metadata['sample_type'] = df_metadata["tcga.tcga_barcode"].apply(
        lambda x: get_sample_type(int(x[13:15]))
    )

    if filter_tumor:
        logger.info(
            (
                f"Selecting {(df_metadata['sample_type'] == 'TUMOR').sum()} tumor"
                f" samples out of {df_metadata.shape[0]}."
            ),
        )
        df_metadata = df_metadata.query("sample_type == 'TUMOR'")

    if unique_patients:
        logger.info(
            f"Selecting {df_metadata.drop_duplicates('patient_id').shape[0]} unique"
            " patients."
        )

    df_metadata = df_metadata.sort_values(
        by=["tcga.cgc_file_last_modified_date"],
    ).drop_duplicates(
        "patient_id" if unique_patients else "tcga.tcga_barcode",
        keep="last",
    )

    # Map index
    map_dict = dict(zip(df_metadata.index, df_metadata['tcga.tcga_barcode']))
    # Filter RNA samples on metadata samples
    df_rna = df_rna[map_dict.keys()]
    df_rna = df_rna.T
    df_rna.index = df_rna.index.map(map_dict)

    rnaseq_variables = list(df_rna.columns.values)

    del df_metadata

    # Convert to proper format
    df_rna = df_rna.reset_index().rename(columns={"index": "barcode"})

    # df_rna["project_id"] = df_rna["barcode"].apply(lambda x: x[:4])
    # df_rna["center_id"] = df_rna["barcode"].apply(lambda x: x[:7])
    # df_rna["patient_id"] = df_rna["barcode"].apply(lambda x: x[:12])
    # df_rna["sample_id"] = df_rna["barcode"].apply(lambda x: x[:15])
    df_rna["sample_type"] = df_rna["barcode"].apply(
        lambda x: get_sample_type(int(x[13:15]))
    )

    to_keep = list(
        [
            "barcode",
            "sample_type",
        ]
        + rnaseq_variables
    )
    df_rna = df_rna[to_keep]

    return df_rna, rnaseq_variables


@lru_cache
def load_tcga(
    cohort: str,
    normalization_rnaseq: str = "TPM",
    rnaseq_only: bool = False,
    batch_corrected: bool = False,
    filter_tumor: bool = False,
    unique_patients: bool = False,
) -> Union[pd.DataFrame, list]:
    """Load RNAseq data and metadata for a given TCGA cohort.

    Args:
        cohort (str): Name of the TCGA cohort.
        normalization_rnaseq (str, optional): Normalization type
            From: RAW, RPKM, TPM, NORM. Defaults to TPM
        rnaseq_only (bool, optional): To load only RNAseq and not clinical.
            Defaults to False.
        batch_corrected (bool, optional): whether to use combat-seq batch corrected data
        filter_tumor (bool, optional): If true, returns only tumoral samples.
        unique_patients (bool, optional): If true, returns unique patients instead of
        unique barcodes.

    Returns:
        Union[pd.DataFrame, list]:
            Dataframe with clinical + RNAseq data and list of RNAseq vars
    """
    if rnaseq_only:
        df_rna, rnaseq_variables = load_rnaseq(
            cohort=cohort,
            normalization=normalization_rnaseq,
            batch_corrected=batch_corrected,
            filter_tumor=filter_tumor,
            unique_patients=unique_patients,
        )

        return df_rna, rnaseq_variables
    df_clin = load_clinical(
        cohort=cohort,
    )
    df_rna, rnaseq_variables = load_rnaseq(
        cohort=cohort,
        normalization=normalization_rnaseq,
        batch_corrected=batch_corrected,
        filter_tumor=filter_tumor,
        unique_patients=unique_patients,
    )

    # "Inner" keeps only the barcodes that exist in both dataframes (no NaN afterwards)
    tcga_df = pd.merge(
        df_clin, df_rna, how="inner", left_on="barcode", right_on="barcode"
    )

    return tcga_df, rnaseq_variables


def load_multiple_cohorts_tcga(
    cohorts: Union[list[str], str],
    normalization_rnaseq: str = "TPM",
    filter_tumor: bool = False,
    unique_patients: bool = False,
) -> tuple[pd.DataFrame, list[str]]:
    """Load data for multiple TCGA cohorts.

    Args:
    cohorts : list[str] : Sequence of strings representing the name of each cohort, or
        string equal to "ALL_TCGA_COHORTS" in the pancancer case
    normalization_rnaseq : str : Type of normalization for the RNASeq data
        From [RAW, RPKM, TPM, NORM, COMBAT_RAW, COMBAT_FPKM, COMBAT_TPM, COMBAT_NORM].
        COMBAT prefix indicates that the data has been batch corrected with COMBAT-SEQ.
        Defaults to TPM
    filter_tumor (bool, optional): If true, returns only tumoral samples.
    unique_patients (bool, optional): If true, returns unique patients instead of
        unique barcodes.
    Returns
        tuple[pd.DataFrame, list]
            Dataframe with concatenated cohorts with RNAseq and clinical data
            and list of RNAseq vars
    """
    batch_corrected = False
    if normalization_rnaseq.split('_')[0] == "COMBAT":
        batch_corrected = True
        normalization_rnaseq = normalization_rnaseq.split('_')[-1]
    if cohorts == "ALL_TCGA_COHORTS":
        return load_all_cohorts_tcga(
            normalization_rnaseq, batch_corrected, filter_tumor, unique_patients
        )
    return load_cohorts_tcga(
        cohorts, normalization_rnaseq, batch_corrected, filter_tumor, unique_patients
    )


@lru_cache
def load_all_cohorts_tcga(
    normalization_rnaseq: str = "TPM",
    batch_corrected: bool = False,
    filter_tumor: bool = False,
    unique_patients: bool = False,
) -> tuple[pd.DataFrame, list[str]]:
    """Load data for all TCGA cohorts.

    This function can be cached, because it doesn't have a list as argument (arguments
    need to be hashable for the cache to work).
    """
    cohorts = ALL_TCGA_COHORTS
    return load_cohorts_tcga(
        cohorts, normalization_rnaseq, batch_corrected, filter_tumor, unique_patients
    )


def load_cohorts_tcga(
    cohorts: list[str],
    normalization_rnaseq: str = "TPM",
    batch_corrected: bool = False,
    filter_tumor: bool = False,
    unique_patients: bool = False,
) -> tuple[pd.DataFrame, list[str]]:
    """Load data for multiple TCGA cohorts.

    Args:
    cohorts : list[str] : Sequence of strings representing the name of each cohort
    normalization_rnaseq : str : Type of normalization for the RNASeq data
        From [RAW, RPKM, TPM, NORM]. Defaults to TPM
    batch_corrected: bool : whether to used combat-seq batch correction
    filter_tumor (bool, optional): If true, returns only tumoral samples.
    unique_patients (bool, optional): If true, returns unique patients instead of
        unique barcodes.

    Returns
        tuple[pd.DataFrame, list]
            Dataframe with concatenated cohorts with RNAseq and clinical data
            and list of RNAseq vars
    """
    dataframes, rnaseq_variables = [], []
    # Load rnaseq & clinical for selected cohorts
    # clinical is useless in case of non-MH AE pretraining
    # but it simplifies the code to load it in all cases
    for c in cohorts:
        logger.info(f"Cohort: {c}")
        tcga_df, cols = load_tcga(
            cohort=c,
            normalization_rnaseq=normalization_rnaseq,
            rnaseq_only=False,
            batch_corrected=batch_corrected,
            filter_tumor=filter_tumor,
            unique_patients=unique_patients,
        )
        dataframes.append(tcga_df)
        rnaseq_variables.append(cols)

    clin_columns = list(
        set.union(*[set(df.columns) for df in dataframes])
        - set.union(*[set(x) for x in rnaseq_variables])
    )

    shared_rnaseq_variables = sorted(
        list(set.intersection(*[set(x) for x in rnaseq_variables]))
    )

    for tcga_df in dataframes:
        tcga_df.drop(
            tcga_df.columns.difference(clin_columns + shared_rnaseq_variables),
            axis=1,
            inplace=True,
        )

    cohorts_df = pd.concat(dataframes, axis=0)

    cohorts_df = cohorts_df[clin_columns + shared_rnaseq_variables].set_index(
        "barcode", drop=False
    )

    return cohorts_df, shared_rnaseq_variables
