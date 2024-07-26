"""Survival metrics wrapper around sklearn metrics."""

import numpy as np
from lifelines.utils import concordance_index


def compute_cindex(labels: np.array, logits: np.array) -> np.float64:
    """Compute c-index.
    Derive the censored labels (events: 1 if observed, 0 if not) from the sign of the
        labels.
    Args:
        labels (np.array): a length-n iterable of observed survival times.
        logits (np.array): a length-n iterable of predicted scores - these could be
        survival times, or hazards, etc.

    Returns:
        np.float64: C-index, a value between 0 and 1 - same interpretation as roc auc.
    """

    times, events = np.abs(labels), 1 * (labels > 0)
    try:
        return concordance_index(times, -logits, events)
    except AssertionError:
        return 0.5
