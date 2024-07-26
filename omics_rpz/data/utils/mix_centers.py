"""Implementation of a function to mix centers into groups called 'environments' in the
TCGA datasets."""

import binpacking
import numpy as np
import pandas as pd


def mix_centers(df_center: pd.DataFrame, n_patients_min=40) -> pd.DataFrame:
    """This function takes as input a dataframe with patient_id and TSS (=center id) as
    input, and outputs a dataframe with an additional column corresponding to the
    created envrionments. Centers are assigned to environments by solving a binpacking
    problem (https://en.wikipedia.org/wiki/Bin_packing_problem)

    Args:
        df_center (pd.DataFrame): Dataframe containing patient_id and TSS (center id)
                          information
        n_patients_min (int, optional): Minimum number of patients per environement.
        Defaults to 40. Which set somewhat arbitrarely to ensure we have enough patients
        in each group - we could increase it further but the folds are balanced with
        this number.


    Returns:
        pd.DataFrame:  dataframe with the created envrionments
    """
    # df_center is a DataFrame with 'center' as column and patient_id as index
    center_counts = df_center['center'].value_counts().to_dict()

    # separate singleton centers with mixed centers
    complete = {}
    incomplete = {}
    for center, counts in center_counts.items():
        if counts >= n_patients_min:
            complete[center] = counts
        else:
            incomplete[center] = counts

    binning_incomplete = binpacking.to_constant_volume(incomplete, n_patients_min)

    binning = [
        {center: counts} for center, counts in complete.items()
    ] + binning_incomplete

    envs = {}
    for i, binn in enumerate(binning):
        for center, counts in binn.items():
            # tolerance for slightly slower binns
            if np.sum(list(binn.values())) < n_patients_min * 0.9:
                envs[center] = np.nan
            else:
                envs[center] = i

    centers = pd.DataFrame(
        {'count': pd.Series(center_counts), 'environment': pd.Series(envs)}
    )
    orphan_centers_by_descending_size = (
        centers[centers.environment.isna()]
        .sort_values(by='count', ascending=False)
        .index.to_list()
    )
    for code in orphan_centers_by_descending_size:
        # find smallest env
        smallest_env = (
            centers.groupby('environment')
            .sum()['count']
            .sort_values(ascending=True)
            .index[0]
        )
        # put the largest orphan center in the smallest env
        centers.loc[code, 'environment'] = smallest_env

    df_environments = df_center.merge(
        centers[['environment']], how='left', left_on='center', right_index=True
    )
    return df_environments


def create_group_envs(df_data: pd.DataFrame, group_covariate: str) -> np.array:
    """Create groups to stratify data for train test.

    Args:
        df_data (pd.DataFrame): Data to split for train test.
        group_covariate (str): Column name to do the stratification.

    Raises:
        ValueError: If the group_covariate is not in the df_data.

    Returns:
        np.array: groups for stratification
    """
    if group_covariate not in set(df_data.columns):
        raise ValueError(
            f"Covariate {group_covariate}                             is not in the"
            " dataframe"
        )

    # Startify on centers:
    # 1. We create first create new balanced environments (binpacking)
    # Each environment has a minimum number of patients and non overlapping
    # centers.
    # 2. We stratify the train/test split on the new environments
    df_center = pd.DataFrame(df_data.loc[:, group_covariate])
    df_center.rename(columns={group_covariate: "center"}, inplace=True)
    df_mix_center = mix_centers(df_center=df_center)
    groups = df_mix_center["environment"].values
    return groups
