"""Class encapsulating a dataset with features X and label y."""
import numpy as np
import torch


class OmicsDataset:
    """Class encapsulating a dataset with features X and label y.

    Attributes:
        X: features matrix
        y: label
    """

    def __init__(self, X, y=None):
        self.X = X
        self.y = y

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, item: int):
        features = torch.from_numpy(self.X[item].astype(np.float32))
        if self.y is not None:
            label = torch.Tensor([self.y[item]]).float()
            return features, label

        return features
