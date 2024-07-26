"""Dice Loss implementation in Pytorch."""

import torch


class DiceLoss(torch.nn.modules.loss._Loss):
    """Computes the Dice Loss between the groundtruth and the predicted masks."""

    def __init__(self):
        super().__init__()

    def forward(
        self, inputs: torch.Tensor, target: torch.Tensor, smooth: float = 1.0
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            inputs (torch.Tensor): predicted mask
            target (torch.Tensor): groundtruth mask
            smooth (float): smoothing parameter

        Returns:
            torch.Tensor: Dice Loss
        """
        num = target.size(0)
        inputs = inputs.reshape(num, -1)
        target = target.reshape(num, -1)
        intersection = inputs * target
        dice = (2.0 * intersection.sum(1) + smooth) / (
            inputs.sum(1) + target.sum(1) + smooth
        )
        dice = 1 - dice.sum() / num
        return dice
