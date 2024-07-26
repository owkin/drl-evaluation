"""Functions useful for logging with MLflow."""

import mlflow
from omegaconf import DictConfig


def _explore_recursive(parent_name, element):
    if isinstance(element, DictConfig):
        for k, v in element.items():
            if isinstance(v, DictConfig):
                _explore_recursive(f"{parent_name}.{k}", v)
            elif len(str(v)) < 500:
                # mlflow does not accept parameters longer than 500 characters
                mlflow.log_param(f"{parent_name}.{k}", v)
    else:
        mlflow.log_param(parent_name, element)


def log_params_recursive(params):
    """Log with MLflow all parameters from the params dictionary.

    Args:
        params (Dict): Dictionary containing the parameters to be logged
    """
    for k, v in params.items():
        _explore_recursive(k, v)
