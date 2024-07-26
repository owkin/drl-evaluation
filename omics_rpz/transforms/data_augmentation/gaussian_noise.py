"Class for data augmentation using gaussian noise."

import numpy as np
import pandas as pd
from loguru import logger


class GaussianNoise:
    """Add a gaussian noise to normalized RNAseq data.

    Transformations:
        1. Duplicate data with the addition of a gaussian noise.

    Parameters
    ----------
    gaussian_std: float = 0.01
        Std of the gaussian noise.
    number_of_duplication: int = 2
        Number of times the data is duplicated and concatenated to the original data.
    variables_to_perturbate: list[str], list of variables for which
            we want to add noise. Variables in df not in this list will be
            duplicated without noise.
    """

    def __init__(
        self,
        gaussian_std: float = 0.01,
        number_of_duplication: int = 2,
        variables_to_perturbate: list[str] = None,
    ):
        self.gaussian_std = gaussian_std
        self.number_of_duplication = number_of_duplication
        self.variables_to_perturbate = variables_to_perturbate

    def fit(
        self,
    ):
        """Nothing to do here for now.

        Later maybe we adapt the noise to the data.
        """
        return self

    def transform(self, df_to_transform: pd.DataFrame) -> pd.DataFrame:
        """Return the input dataframe with the addition of gaussian noise.

        Args:
            df_to_transform: pd.DataFrame, Input df for which we want to add noise.

        Use the attributes of the class:
            gaussian_std: float, Std of the gaussian noise.
            number_of_duplication: int, Number of times the data is duplicated
                and concatenated with the original data.
            variables_to_perturbate: list[str], list of variables for which
                we want to add noise. Variables in df_to_transform not in this list
                will be duplicated without noise. If set to None select all columns.

        Returns:
            pd.DataFrame, of shape
                (df_to_transform.shape[0]*(1+number_of_duplication),
                df_to_transform.shape[1])
                with the added gaussian noise on the variables_to_perturbate.
        """
        return add_noise(
            df_to_transform,
            self.gaussian_std,
            self.number_of_duplication,
            self.variables_to_perturbate,
        )

    def transform_with_labels(
        self, df_to_transform: pd.DataFrame, y_labels: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Return the input dataframes with the addition of gaussian noise.

        Args:
            df_to_transform: pd.DataFrame, Input df for which we want to add noise.
            y_labels: pd.DataFrame, Default to None. Labels to concatenated

        Use the attributes of the class:
            gaussian_std: float, Std of the gaussian noise.
            number_of_duplication: int, Number of times the data is duplicated
                and concatenated with the original data.
            variables_to_perturbate: list[str], list of variables for which
                we want to add noise. Variables in df_to_transform not in this list
                will be duplicated without noise. If set to None select all columns.

        Returns:
            pd.DataFrame, of shape
                (df_to_transform.shape[0]*(1+number_of_duplication),
                df_to_transform.shape[1])
                with the added gaussian noise on the variables_to_perturbate.
            pd.DataFrame (or one columns pd.DataFrame), copy of size
                (df_to_transform.shape[0]*(1+number_of_duplication)
                with no noise addition.
        """
        df_to_transform_all = df_to_transform.copy()
        # Join is only left at this stage as we may want to use more X samples even if
        # they don't have corresponding y.
        # Index is used to do the join.
        df_to_transform_all = df_to_transform_all.join(y_labels.copy(), how='left')

        df_transformed_all = add_noise(
            df_to_transform_all,
            self.gaussian_std,
            self.number_of_duplication,
            self.variables_to_perturbate,
        )
        return (
            df_transformed_all.drop(columns=y_labels.columns),
            df_transformed_all[y_labels.columns],
        )

    def fit_transform(self, df_to_transform: pd.DataFrame) -> pd.DataFrame:
        """Fit the transform_method (nothing) and transform the input df."""
        self.fit()
        return self.transform(df_to_transform)


def sample_noise(
    gaussian_std: float,
    df_to_transform: pd.DataFrame,
    variables_to_perturbate: list,
    seed: int,
) -> pd.DataFrame:
    """Sample a noise matrix with the right shape and column names.

    Parameters
    ----------
    gaussian_std : float
        noise standard deviation
    df_to_transform : pd.DataFrame
        dataframe to noise
    variables_to_perturbate : list[str]
    list of variables for which
        we want to add noise. Variables in df_to_transform not in this list
        will be duplicated without noise. If set to None select all columns.
    seed : int
        seed for random number generation

    Returns
    -------
    pd.DataFrame
        noise matrix
    """
    np.random.seed(seed)

    # Parameters of the gaussian noise.
    # mean is always set to zero for now.
    mean = 0
    sigma = gaussian_std

    # Creating a noise with the same dimension as the dataset.
    noise = pd.DataFrame(
        np.random.normal(
            mean, sigma, (df_to_transform.shape[0], len(variables_to_perturbate))
        ),
        columns=variables_to_perturbate,
        index=df_to_transform.index,
    )
    return noise


def add_noise(
    df_to_transform: pd.DataFrame,
    gaussian_std: float,
    number_of_duplication: int,
    variables_to_perturbate: list[str],
) -> pd.DataFrame:
    """Return the input dataframe with the addition of gaussian noise.

    Args:
        df_to_transform: pd.DataFrame, Input df for which we want to add noise.
        gaussian_std: float, Std of the gaussian noise.
        number_of_duplication: int, Number of times the data is duplicated
            and concatenated with the original data. If set to zero, the original data
            will be noised and returned.
        variables_to_perturbate: list[str], list of variables for which
            we want to add noise. Variables in df_to_transform not in this list
            will be duplicated without noise. If set to None select all columns.

    Returns:
        final_df: pd.DataFrame, of shape:
            df_to_transform.shape[0]*(1+number_of_duplication),df_to_transform.shape[1])
            with the added gaussian noise on the variables_to_perturbate.
    """
    final_df = df_to_transform.copy()

    # Store each noised df into a dict by seed to optimize final concatenation
    # (instead of doing concat at every step).
    dfs_with_noise = {}

    if variables_to_perturbate is None:
        variables_to_perturbate = df_to_transform.columns
        try:
            df_to_transform = df_to_transform.astype(float)
        except ValueError:
            # If not only floats log the error and use only columns that are floats.
            logger.error(
                '''Need to have only floats in your input dataframe or
            pass a specific list of columns to the Noise transformer.'''
            )
            variables_to_perturbate = list(
                df_to_transform.dtypes.loc[df_to_transform.dtypes == 'float64'].index
            )

    for i in range(number_of_duplication):
        # Setting a reproducible different seed for each run.
        df_with_noise = df_to_transform.copy()
        df_with_noise.index = df_with_noise.index.astype('str') + '_noise_' + str(i)
        # This will convert the noised df index to strings while the original one may
        # still be a float for instance. But it's okay for now to potentially have two
        # types in the index and it will simply be an object.
        noise = sample_noise(
            gaussian_std=gaussian_std,
            df_to_transform=df_with_noise,
            seed=i,
            variables_to_perturbate=variables_to_perturbate,
        )
        df_with_noise.loc[:, variables_to_perturbate] += noise
        dfs_with_noise[i] = df_with_noise

    if number_of_duplication == 0:
        # In this case we just want to add noise to the original data,
        # without duplication
        noise = sample_noise(
            gaussian_std=gaussian_std,
            df_to_transform=df_to_transform,
            seed=0,
            variables_to_perturbate=variables_to_perturbate,
        )
        final_df.loc[:, variables_to_perturbate] += noise

        return final_df

    # Merge all the noised df + the original into one.
    final_df = pd.concat([final_df] + list(dfs_with_noise.values()))
    return final_df
