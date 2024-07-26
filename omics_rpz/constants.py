"""Various constants to use throughtout omics rpz."""

PREFIX = "./benchmark_data/"
TCGA_PATHS = {
    "RNASEQ": PREFIX + "RECOUNT3/tcga/",
    "RNASEQ_PARQUET": PREFIX + "RECOUNT3/tcga/",
    "CLINICAL": PREFIX + "processed_tcga_annot/Clinical/",
    "MUTATIONS": PREFIX + "processed_tcga_annot/Mutations/",
    "CDR": PREFIX + "processed_tcga_annot/Survival/survival_labels_tcga.parquet"
}
PATH_CCLE = PREFIX + "CCLE_22Q4/"
PATH_GTEX = PREFIX + "RECOUNT3/gtex"
PATH_PROTEIN_CODING_GENES = PREFIX + "gene_with_protein_product.parquet"

PATH_TO_STRING_1 = (
    PREFIX + "STRING/9606.protein.links.detailed.v11.5.txt.gz"
)
PATH_TO_STRING_2 = PREFIX + "STRING/9606.protein.info.v11.5.txt.gz"
PATH_TO_STRING = PREFIX + "STRING"
PATH_TO_MSIGDB = PREFIX + "msigdb/msigdb.v7.5.1.tsv"

OMICS_RPZ_PATH = PREFIX + "dataset_alignment"
PATH_GENE_LISTS = PREFIX + "gene_lists"

# fmt: off
ALL_TCGA_COHORTS = [
    "ACC", "CHOL", "GBM", "KIRP", "LUAD", "PAAD", "READ", "TGCT", "UCS", "BLCA",
    "COAD", "HNSC", "LAML", "LUSC", "SARC", "THCA", "UVM", "BRCA", "DLBC",
    "KICH", "LGG", "MESO", "PCPG", "SKCM", "THYM", "CESC", "ESCA", "KIRC",
    "LIHC", "OV", "PRAD", "STAD", "UCEC",
]

OS_COHORTS = [
    "BRCA", "UCEC", "KIRC", "HNSC", "LUAD", "LGG",
    "LUSC", "SKCM", "COAD", "STAD", "BLCA"
]
# fmt: on

# List of possible labels to be predicted, for which there can be a config
TASK_LABEL_LIST = ['OS', 'PFS', 'gender', 'age', 'purity', 'treatment', 'Deconvolution']
# List of possible tasks for which there can be a config
TASK_LIST = [
    "GeneEssentiality",
    "TCGAPredictionTask",
]

# column names with gene essentiality scores contain this string
ESSENTIALITY_LABEL = "essentiality"

# column names with pathway activation scores contain this string
PATHWAY_ACTIVATION_LABEL = "progeny"

REPEATED_HOLDOUT_PATH = "repeated_holdout"
