"""MEAN ABSOLUTE PERCENTAGE ERROR (MAPE) implementation in pytorch."""
import torch


class MAPELoss(torch.nn.modules.loss._Loss):
    """Computes the Mean Absolute Percentage Error (MAPE) loss between the actual and
    predicted values."""

    def __init__(self):
        super().__init__()

    def forward(self, actual, pred):
        """Computes the MAPE loss between the actual and predicted values.

        Args:
        actual (torch.Tensor): Actual values.
        pred (torch.Tensor): Predicted values.

        Returns:
        torch.Tensor: The MAPE loss between actual and predicted values.
        """
        error = torch.abs((actual - pred) / (actual + 1e-4))
        mape_loss = torch.mean(error) * 100
        return mape_loss
