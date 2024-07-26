"""Utils functions that can apply to different datasets."""
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold


def test_split(
    df_all: pd.DataFrame,
    groups: np.array = None,
    stratify: np.array = None,
    split_ratio: int = 0.2,
    random_seed: int = 0,
):
    """Splits dataset in train/test using a fixed random seed for reproducibility.

    Note: reproducibility is limited as a change in the input dataset
    might change the split.

    Args:
        df_all (pd.DataFrame): input dataset to split in train/test datasets.
        groups (np.array): input groups used for the split, default to None.
        stratify (np.array): label on which to stratify - if None do not stratify
        split_ratio (float): ratio to use as a test set from df_all
        random_seed (int): random_seed used by ShuffleSplit to split train/test datasets

    Returns:
        List[pd.DataFrame]: train and test datasets.
    """
    # StratifiedGroupKFold might be used for CV in some cases.
    # Here it just allows not to have a same patient in both sets
    # and stratify on censorship
    if split_ratio == 0:
        return df_all, pd.DataFrame(columns=df_all.columns)

    # Here we actually output (1 / split_ratio) folds and we will select the first one.
    splits_gen = create_fold_iterator(
        df_all, groups, stratify, split_ratio, random_seed
    )
    # splits_gen is a generator object, we get the value from it using next
    train_idx, test_idx = next(splits_gen)

    return df_all.iloc[train_idx], df_all.iloc[test_idx]


def create_fold_iterator(
    df_all: pd.DataFrame,
    groups: np.array = None,
    stratify: np.array = None,
    split_ratio: int = 0.2,
    random_seed: int = 0,
) -> StratifiedGroupKFold:
    """Create sklearn split iterator to return indexes for train,test split.

    Use the appropriate sklearn.model_selection function depending on the params given.

    Args:
        df_all (pd.DataFrame): input dataset to split in train/test datasets.
        groups (np.array): input groups used for the split, default to None.
        stratify (np.array): label on which to stratify - if None do not stratify
        split_ratio (float): ratio to use as a test set from df_all
        random_seed (int): random_seed used by ShuffleSplit to split train/test datasets

    Returns:
        List[pd.DataFrame]: train and test datasets.
    """
    if stratify is None:
        # Create dummy vector.
        stratify = np.zeros(len(df_all))
        # Cannot do stratify here in the case we do have groups we want to use but no
        # stratify.
        # TODO: could add if no group and if no stratify do KFold

    assert len(stratify) == len(df_all), "X and y have different shapes"

    if len(np.unique(stratify)) == 2:
        # If stratify is boolean do no change anything
        pass
    else:
        # If stratify can be negative stratify based on positivity - equivalent to
        # censorship for survival prediction.
        stratify = stratify >= 0

    if groups is None or len(set(groups)) == df_all.shape[0]:
        sgkf = StratifiedKFold(
            n_splits=int(1 / split_ratio), shuffle=True, random_state=random_seed
        )
        splits_gen = sgkf.split(X=np.arange(len(df_all)), y=stratify)
    else:
        sgkf = StratifiedGroupKFold(
            n_splits=int(1 / split_ratio), shuffle=True, random_state=random_seed
        )
        splits_gen = sgkf.split(
            X=np.arange(len(df_all)),
            y=stratify,
            groups=groups,
        )

    return splits_gen
