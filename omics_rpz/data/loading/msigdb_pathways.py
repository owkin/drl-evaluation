"""Loading functions for MSIGDB data."""
import pandas as pd

from omics_rpz.constants import PATH_TO_MSIGDB

PATHWAY_LIBRARIES = {
    'KEGG': 'CP:KEGG',
    'REACTOME': 'CP:REACTOME',
}


def load_pathways(genes: list = None, source: str = 'KEGG') -> list:
    """Function to load pathways from the MSIGDB database.

    Args:
        genes (list, optional): List of genes that you wan to keep. If None, will keep
        all genes in the patways. Defaults to None.
        source (str, optional): Name of the library of pathways used. Defaults to
        'KEGG'.

    Returns:
        list: list of pathways, each represented by a gene list.
    """
    msigb_df = pd.read_csv(PATH_TO_MSIGDB, sep='\t')
    library = PATHWAY_LIBRARIES.get(source, 'CP:KEGG')
    msigb_df = msigb_df[msigb_df.SUB_CATEGORY_CODE == library]
    pathway_list = msigb_df.MEMBERS_SYMBOLIZED.to_list()
    pathway_list = [pathway.split(',') for pathway in pathway_list]
    if genes is None:
        return pathway_list
    pathway_list_filtered = []
    for path in pathway_list:
        path = list(set(genes) & set(path))
        if len(path) > 0:
            pathway_list_filtered.append(path)
    pathway_list = pathway_list_filtered
    return pathway_list
