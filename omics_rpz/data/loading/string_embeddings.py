"""STRING db helper functions, from loading to embedding generation."""

import pandas as pd
from loguru import logger

from omics_rpz.constants import PATH_TO_STRING_1, PATH_TO_STRING_2
from omics_rpz.utils import Memoized


@Memoized
def load_string():
    """Load STRING network and gene aliases from path stored in omics_rpz.constants.

    Returns
    -------
    Union[pd.DataFrame, pd.DataFrame]
        Tuple containing:
            string_db: pd.DataFrame
                STRING dataframe in a edgelist format
            gene_aliases: pd.DataFrame
                Gene aliases dataframe
    """
    # Load STRING files
    logger.info("Loading STRING PPI network")
    string_db = pd.read_table(PATH_TO_STRING_1, sep=' ')
    genes_aliases = pd.read_table(
        PATH_TO_STRING_2, sep="\t", index_col='#string_protein_id'
    )

    # Rename STRING genes with their preferred name
    logger.info("Filtering STRING genes and edges and building the graph ")
    protein_pref_names_dict = genes_aliases['preferred_name'].to_dict()
    string_db['protein1'] = string_db['protein1'].map(protein_pref_names_dict)
    string_db['protein2'] = string_db['protein2'].map(protein_pref_names_dict)

    return string_db, genes_aliases


def gene_intersection_with_string(
    rnaseq_variables: list[str], string_db: pd.DataFrame, genes_aliases: pd.DataFrame
):
    """Filter gene list and dataframe to contain only intersecting genes between the 2.

    Parameters
    ----------
    rnaseq_variables : List
        List of genes to intersect with (e.g. columns of RNA-seq counts)
    string_db : pd.DataFrame
        STRING dataframe returned by load_string
    genes_aliases : pd.DataFrame
        Gene aliases dataframe returned by load_string
    Returns
    -------
    Union[List, pd.DataFrame]
        Tuple containing:
            common_genes: List
                Filtered gene list
            string_db: pd.DataFrame
                Filtered STRING DataFrame
    """
    # Finding genes in common between STRING and rnaseq_variables
    logger.info(f"Number of genes in X: {len(rnaseq_variables)}")
    string_db_genes = set(genes_aliases['preferred_name'])
    logger.info(f"Number of STRING genes: {len(string_db_genes)}")
    common_genes = set(rnaseq_variables) & string_db_genes
    logger.info(f"Number of genes in common :{len(common_genes)}")
    # Filtering out from STRING the genes that are not in common with rnaseq_variables
    string_db = string_db[string_db['protein1'].isin(common_genes)]
    string_db = string_db[string_db['protein2'].isin(common_genes)]
    return common_genes, string_db
