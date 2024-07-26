"""Loading functions for RECOUNT3-GTEx dataset."""

from functools import lru_cache
from pathlib import Path
from typing import Union

import pandas as pd
from loguru import logger

from omics_rpz.constants import PATH_GTEX

# fmt: off
GTEX_COHORTS = [
    'Vagina', 'Bone_Marrow', 'Pancreas', 'Heart', 'Brain', 'Liver', 'Spleen',
    'Cervix_Uteri', 'Small_Intestine', 'Esophagus', 'Kidney', 'Prostate', 'Nerve',
    'Ovary', 'Thyroid', 'Blood_Vessel', 'Skin', 'Salivary_Gland', 'Blood', 'Lung',
    'Colon', 'Adrenal_Gland', 'Breast', 'Stomach', 'Pituitary', 'Muscle', 'Uterus',
    'Adipose_Tissue', 'Bladder', 'Testis', 'Fallopian_Tube'
]
# fmt: on

TCGA_GTEX_MAPPING = {
    "ACC": "Adrenal_Gland",
    "BLCA": "Bladder",
    "BRCA": "Breast",
    "CESC": "Cervix_Uteri",
    "CHOL": "Liver",
    "COAD": "Colon",
    "DLBC": None,
    "ESCA": "Esophagus",
    "GBM": "Brain",
    "HNSC": None,
    "KICH": "Kidney",
    "KIRC": "Kidney",
    "KIRP": "Kidney",
    "LAML": "Blood",
    "LGG": "Brain",
    "LIHC": "Liver",
    "LUAD": "Lung",
    "LUSC": "Lung",
    "MESO": None,
    "OV": "Ovary",
    "PAAD": "Pancreas",
    "PCPG": "Adrenal_Gland",
    "PRAD": "Prostate",
    "READ": None,
    "SARC": "Adipose_Tissue",
    "SKCM": "Skin",
    "STAD": "Stomach",
    "TGCT": "Testis",
    "THCA": "Thyroid",
    "THYM": None,
    "UCEC": "Uterus",
    "UCS": "Uterus",
    "UVM": None,
}
TCGA_GTEX_DETAILED_MAPPING = {
    "ESCA": ["Esophagus - Mucosa"],
    "GBM": [
        "Brain - Cerebellum",
        "Brain - Caudate (basal ganglia)",
        "Brain - Cortex",
        "Brain - Nucleus accumbens (basal ganglia)",
        "Brain - Frontal Cortex (BA9)",
        "Brain - Cerebellar Hemisphere",
        "Brain - Hypothalamus",
        "Brain - Putamen (basal ganglia)",
        "Brain - Anterior cingulate cortex (BA24)",
        "Brain - Spinal cord (cervical c-1)",
    ],
    "LGG": [
        "Brain - Caudate (basal ganglia)",
        "Brain - Cortex",
        "Brain - Nucleus accumbens (basal ganglia)",
        "Brain - Frontal Cortex (BA9)",
        "Brain - Hypothalamus",
        "Brain - Putamen (basal ganglia)",
        "Brain - Hippocampus",
        "Brain - Anterior cingulate cortex (BA24)",
        "Brain - Amygdala",
        "Brain - Substantia nigra",
    ],
    "LAML": ["Whole Blood"],
    "SKCM": ["Skin - Sun Exposed (Lower leg)", "Skin - Not Sun Exposed (Suprapubic)"],
}


@lru_cache
def load_gtex(
    cohort: str,
    normalization_rnaseq: str = "TPM",  # raw, rpkm, tpm, norm
) -> Union[pd.DataFrame, list]:
    """Load RNASeq data for a GTEx cohort.

    Args:
        cohort (str): _description_
        normalization_rnaseq (str, optional): Normalization type from: RAW, RPKM, TPM
        Defaults to "TPM".

    Returns:
        Union[pd.DataFrame, list]:
            Dataframe with annotations and RNAseq data, and list of RNAseq vars
    """

    norm_dict = {"RAW": "raw", "RPKM": "rpkm", "TPM": "tpm"}
    normalization = norm_dict[normalization_rnaseq]

    detailed_tissues = None
    if cohort in TCGA_GTEX_MAPPING:
        detailed_tissues = TCGA_GTEX_DETAILED_MAPPING.get(cohort, None)
        logger.info(
            f"Loading GTEX cohort {TCGA_GTEX_MAPPING[cohort]} for TCGA cohort {cohort}"
            + (
                f" with detailed tissues {detailed_tissues}"
                if detailed_tissues
                else "."
            ),
        )
        cohort = TCGA_GTEX_MAPPING[cohort]

    # Get paths
    rnaseq_path = (
        Path(PATH_GTEX) / cohort / "Data" / f"Counts_{normalization}_float32.parquet"
    )
    metadata_path = Path(PATH_GTEX) / cohort / "Data" / "metadata_v2.tsv.gz"

    # Load RNAseq
    df_rna = pd.read_parquet(rnaseq_path).set_index("Hugo").T
    # Load metadata
    df_metadata = pd.read_csv(metadata_path, sep="\t", index_col="external_id")
    metadata_to_keep = ["study", "gtex.subjid", "gtex.sex", "gtex.age", "gtex.sampid"]
    metadata_to_keep = metadata_to_keep + [
        col for col in df_metadata.columns if "decoupler" in col
    ]
    if detailed_tissues:
        df_metadata = df_metadata.query("`gtex.smtsd` == @detailed_tissues")
    df_metadata = df_metadata[metadata_to_keep]
    df_metadata = df_metadata.rename(columns={"study": "tissue"})
    rnaseq_variables = list(df_rna.columns.values)
    # Join and delete duplicates
    gtex_df = df_rna.join(df_metadata, how="inner")
    gtex_df = gtex_df.drop_duplicates("gtex.sampid")
    return gtex_df, rnaseq_variables


def load_multiple_cohorts_gtex(
    cohorts: list = None,
    normalization_rnaseq: str = "TPM",
) -> tuple[pd.DataFrame, list[str]]:
    """Load multiple GTEx cohorts and align them. Default is loading all of the GTEx
    cohorts.

    Args:
        cohorts (list, optional): list of the names of the GTEx cohorts.
            "When is None defaults to GTEX_COHORTS."
        normalization_rnaseq (str, optional): Normalization type from: RAW, RPKM, TPM
            Defaults to "TPM".

    Returns:
        tuple[pd.DataFrame, list[str]]: Dataframe with concatenated cohorts
        with RNAseq and clinical data and list of RNAseq vars
    """
    if cohorts is None:
        cohorts = GTEX_COHORTS
    # TODO: Refactoring. Externalize this function in /data/loading/load_utils.py
    # for instance and merge it with load_multiple_cohorts_tcga
    # https://app.asana.com/0/0/1203149175727258/f
    dataframes, rnaseq_variables = [], []
    for c in cohorts:
        print(f"Loading {c} GTEx cohort | ", end="")
        gtex_df, cols = load_gtex(cohort=c, normalization_rnaseq=normalization_rnaseq)

        dataframes.append(gtex_df)
        rnaseq_variables.append(cols)
        print(f"{len(cols)} Genes")

    cohorts_df = pd.concat(dataframes, axis=0)

    clin_columns = list(
        set(cohorts_df.columns) - set.union(*[set(x) for x in rnaseq_variables])
    )
    shared_rnaseq_variables = sorted(
        list(set.intersection(*[set(x) for x in rnaseq_variables]))
    )

    cohorts_df = cohorts_df[clin_columns + shared_rnaseq_variables]

    return cohorts_df, shared_rnaseq_variables
