"""Wrapper for the Lifelines Cox Proportional Hazards (CoxPH) implementation."""
from typing import Union

import lifelines
import numpy as np
import pandas as pd


class CoxModel:
    """Wrapper for the Lifelines Cox Proportional Hazards (CoxPH) implementation.

    Parameters:
        penalizer (float): Regularization coefficient
        l1_ratio (float): Ratio assigned to a L1 vs L2 penalty
        step_size (float): > 0.001 to determine a starting step size in NR algorithm.
    Attributes:
        convergence_success (bool): If convergence of model succeeded or not.
    """

    def __init__(
        self, penalizer: float = 1e-5, l1_ratio: float = 0.0, step_size: float = 0.95
    ):
        self.model = lifelines.CoxPHFitter(
            penalizer=penalizer,
            l1_ratio=l1_ratio,
        )
        self.convergence_success = True
        self.step_size = step_size

    def fit(self, X: Union[np.ndarray, pd.DataFrame], y: np.ndarray) -> None:
        """
        Args:
            X (np.ndarray or pd.DataFrame): Training data.
            y (np.ndarray): Target values.
        """
        data = pd.DataFrame(X, index=y.index)
        data["T"] = np.abs(y)
        data["E"] = 1 * (y > 0)

        try:
            self.model.fit(
                data,
                "T",
                event_col="E",
                show_progress=False,
                fit_options={"step_size": self.step_size},
            )
            self.convergence_success = True
        except lifelines.exceptions.ConvergenceError:
            self.convergence_success = False
            print("Convergence Failed, returning random predictions.")

    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Predict using Cox Model with feature matrix X.

        Args:
            X (np.ndarray or pd.DataFrame): Samples

        Returns:
            np.ndarray: Predicted values
        """
        if self.convergence_success:
            return -self.model.predict_expectation(X).values
        return np.random.uniform(low=0, high=1, size=(len(X),))
