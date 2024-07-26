"""Import data loading functions."""

from .datasets.omics_dataset import OmicsDataset
from .loading.ccle import (
    load_ccle,
    load_data_for_essentiality_pred,
    load_tcga_ccle_mapping,
)
from .loading.gtex import load_gtex, load_multiple_cohorts_gtex
from .loading.tcga import load_multiple_cohorts_tcga, load_tcga
from .loading.utils import create_fold_iterator, test_split
from .utils.mix_centers import create_group_envs

try:
    from .loading.graph_loading import load_graph
except ModuleNotFoundError:
    pass
