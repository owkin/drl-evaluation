"""Cox Loss implementation in Pytorch."""

import torch

def cox(input: torch.Tensor, target: torch.Tensor, reduction: str = "mean"):
    """
    Cox Loss implemented in PyTorch
    There shouldn't be any zero value (because we couldn't determine censure)

    Parameters
    ----------
    input: torch.Tensor
        risk prediction from the model (higher score means higher risk and lower survival)
    target: torch.Tensor
        labels of the event occurences. Negative values are the censored values.
    reduction: str
        Specifies the reduction to apply to the output:
        ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
        ``'mean'``: the sum of the output will be divided by the number of
        elements in the output, ``'sum'``: the output will be summed. Note: :attr:`size_average`
        and :attr:`reduce` are in the process of being deprecated, and in the meantime,
        specifying either of those two args will override :attr:`reduction`. Default: ``'mean'``

    Returns
    -------
    loss: torch.Tensor
        the cox loss (scalar)
    """
    input, target = _sort_risks(input, target)

    # The tensors are reversed because the generator gives the target in
    # ascending order, and this implementation assumes a descending order.
    input = input.flip(0)
    target = target.flip(0)

    hazard_ratio = torch.exp(input)
    e = (torch.sign(target) + 1) / 2.0

    log_risk = torch.log(torch.cumsum(hazard_ratio, 0))
    uncensored_likelihood = input - log_risk
    censored_likelihood = -uncensored_likelihood * e

    if reduction != "none":
        censored_likelihood = (
            torch.mean(censored_likelihood)
            if reduction == "mean"
            else torch.sum(censored_likelihood)
        )
    return censored_likelihood


class CoxLoss(torch.nn.modules.loss._Loss):
    """
    Cox Loss implemented in PyTorch
    There shouldn't be any zero value (because we couldn't determine censure)

    Parameters
    ----------
    reduction: str
        Specifies the reduction to apply to the output:
        ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
        ``'mean'``: the sum of the output will be divided by the number of
        elements in the output, ``'sum'``: the output will be summed. Note: :attr:`size_average`
        and :attr:`reduce` are in the process of being deprecated, and in the meantime,
        specifying either of those two args will override :attr:`reduction`. Default: ``'mean'``
    """

    def __init__(self, reduction: str = "mean"):
        super(CoxLoss, self).__init__(reduction=reduction)

    def forward(self, input: torch.Tensor, target: torch.Tensor):
        """
        Parameters
        ----------
        input: torch.Tensor
            risk prediction from the model (higher score means higher risk and lower survival)
        target: torch.Tensor
            labels of the event occurences. Negative values are the censored values.
        Returns
        -------
        cox_loss: torch.Tensor
        """
        return cox(input, target, reduction=self.reduction)
