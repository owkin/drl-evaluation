"""Import Losses used in OMICS RPZ."""
from torch.nn import (
    BCEWithLogitsLoss,
    CrossEntropyLoss,
    MSELoss,
    ReLU,
    Sigmoid,
    Softmax,
)

from .mape import MAPELoss
from .cox import CoxLoss
