"""Miscellaneous functions useful for models and metrics."""

import numpy as np


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Implementation of the sigmoid function.

    Args:
        x (np.ndarray): input to the sigmoid function

    Returns:
        np.ndarray: output of the sigmoid function
    """
    return 1.0 / (1.0 + np.exp(-x))


def softmax(x: np.ndarray) -> np.ndarray:
    """Implementation of the softmax function.

    Args:
        x (np.ndarray): input of the softmax function

    Returns:
        np.ndarray: output of the softmax function
    """
    z = x - x.max(axis=1, keepdims=True)
    numerator = np.exp(z)
    denominator = np.sum(numerator)
    return numerator / denominator
