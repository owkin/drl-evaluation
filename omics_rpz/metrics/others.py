"""Metric function for the biological annotation task."""
import numpy as np
import torch


def compute_metagenes_enriched(n_pathways, annotations) -> tuple[int, np.float64]:
    """Evaluates the metagenes' annotation process. In an ideal world, all metagenes are
    annotated and there is a 1 to 1 relationship between genes and annotations
    (selectivity = 1).

    Args:
        n_pathways (List): 1st output of the GeneSetEnrichment prediction model
        annotations (Dict): 2nd output of the GeneSetEnrichment prediction model

    Returns:
        tuple (int, float): Number of annotated metagenes, Selectivity of the
        metagene -> annotations function
    """
    n_c = len(annotations)
    n_f = np.count_nonzero(np.array(n_pathways))
    l_p = np.sum(n_pathways)
    l_2 = sum(annotations.values())
    assert l_p == l_2, "problem in the annotation process"
    selectivity = (n_c + n_f) / (2 * l_p)
    return n_f, selectivity


def dice_metric(inputs: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Evaluates between a groundtruth and a predicted mask.

    Args:
        inputs (torch.Tensor): predicted mask
        target (torch.Tensor): groundtruth mask

    Returns:
        torch.Tensor: Dice metric
    """
    intersection = 2.0 * (target * inputs).sum()
    union = target.sum() + inputs.sum()
    if target.sum() == 0 and inputs.sum() == 0:
        return 1.0

    return intersection / union
