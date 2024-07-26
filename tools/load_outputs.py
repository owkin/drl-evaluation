"""Utils loading result functions."""

import concurrent
import os
import pickle
from collections.abc import Callable, Generator
from pathlib import Path
from typing import Any, Optional

import google.cloud.storage as gcp_storage
import numpy as np
import pandas as pd
from hydra.utils import instantiate
from loguru import logger
from omegaconf import DictConfig
from tqdm import tqdm

from omics_rpz.constants import ALL_TCGA_COHORTS, OS_COHORTS, TCGA_PATHS
from omics_rpz.metrics import compute_bootstrapped_metrics, compute_cindex
from omics_rpz.utils import load_pickle


def flatten_df(nested_df: pd.DataFrame) -> pd.DataFrame:
    """Flattens a dataframe containing dictionary.

    Recursive function to tackle nested dictionaries as values.

    Args:
        nested_df (pd.DataFrame): Dataframe with nested values

    Returns:
        pd.DataFrame: Dataframe with expanded columns that contained dictionary
    """
    # Extract top level names of the DictConfig
    dfs = []
    # For each param, we extract the nested values
    for col in nested_df.columns:
        if isinstance(nested_df[col].values[0], DictConfig):
            flattened_col = nested_df[col].apply(pd.Series)
            flattened_col.columns = [f"{col}.{c}" for c in flattened_col.columns]
            col_df = flatten_df(flattened_col)
        else:
            col_df = nested_df[col]
        dfs.append(col_df)
    return pd.concat(dfs, axis=1)


def flatten_nested_config(config_dict: DictConfig) -> pd.DataFrame:
    """Flattens the config dictionary.

    Args:
        config_dict (DictConfig): Hydra Config Dictionary

    Returns:
        pd.DataFrame: Dataframe with all parameters as columns
    """
    flatten_header = pd.json_normalize(dict(config_dict))
    flattened_config = flatten_df(flatten_header)
    return flattened_config


def get_experiment_folders(
    path_to_logs: str,
    study_names: list,
    bucket: gcp_storage.Bucket = None,
    client: gcp_storage.Client = None,
    start_date: str = None,
    end_date: str = None,
    repeated_holdout: bool = False,
) -> list:
    """List all experiment folders within the given path to logs.

    Args:
        path_to_logs (str): Path where to search
        study_names (list, optional): List of relevant study names. Defaults to None.
        In that case all studies are loaded
        bucket (gcp_storage.Bucket, optional): GCP bucket. Defaults to None.
        client (gcp_storage.Client, optional): GCP client. Defaults to None.
        start_date (str, optional): Start date to filter on. Defaults to None.
        end_date (str, optional): End date to filter on. Defaults to None.
        repeated_holdout (bool, optional): Whether the experiments to load were launched
        using the repeated_holdout.py script. Defaults to False

    Returns:
        list: list of experiments paths
    """
    if study_names and not isinstance(study_names, list):
        raise ValueError(f"study_names is a {type(study_names)}, it should be a list.")

    if bucket is None:
        if repeated_holdout:
            # if the experiment is repeated holdout, we have to go one directory deeper
            # only necessary in local
            experiments_folders = list(Path(path_to_logs).glob("*/*"))
        else:
            experiments_folders = list(Path(path_to_logs).glob("*"))
    else:
        experiment_blobs = client.list_blobs(bucket, prefix=path_to_logs)
        experiments_folders = list(set(Path(b.name).parent for b in experiment_blobs))

    if start_date or end_date:
        experiments_folders = [
            f
            for f in experiments_folders
            if (str(f)[-19:] >= start_date if start_date else True)
            and (str(f)[-19:] <= end_date if start_date else True)
        ]

    if not study_names:
        # no study name filtering
        return experiments_folders

    to_keep = [
        xp
        for xp in experiments_folders
        if any(study_name in str(xp) for study_name in study_names)
    ]
    return to_keep


def load_results(
    result_file: str, params_file: str, bucket: gcp_storage.Bucket
) -> tuple[pd.DataFrame, Any]:
    """Loads the given file. result_file as a dataframe, params_file is unpickled.

    Args:
        result_file (str): .csv results file path
        params_file (str): .pickle param config file
        bucket (gcp_storage.Bucket): Gcp bucket. If None is passed, files are searched
            locally.

    Returns:
        tuple[pd.DataFrame, Any]: Loaded outputs
    """

    if bucket is not None:
        results = pd.read_csv(f"gs://{bucket.name}/{result_file}")
        params = bucket.blob(params_file).download_as_string()
        params = pickle.loads(params)

    else:
        results = pd.read_csv(result_file)
        with open(params_file, "rb") as file_pointer:
            params = pickle.load(file_pointer)

    return results, params


def not_corrupted(path: str, bucket: Optional[gcp_storage.Bucket]) -> bool:
    """Wether or not the file exists.

    Args:
        path (str): Path to file
        bucket (Optional[gcp_storage.Bucket]): GCP bucket. If None is passed,
        the file path will be checked locally.

    Returns:
        bool: Wether or not the file exists.
    """
    if bucket is None:
        return Path(path).exists()

    return bucket.blob(path).exists()


def _multi_thread_load_experiment(
    _method: Callable,
    paths: list[Path],
    bucket: Optional[str],
    max_workers: int = None,
) -> Generator[tuple[Optional[pd.DataFrame], Optional[str]], None, None]:
    """Multi threading _load_experiment function.

    As this function requires a lots of IO and network connection,
    multithreading it allows to fasten the process.


    Args:
        _method (Callable): Load experiment function to call
        paths (list[Path]): List of paths given as a first arg to the _method
        bucket (str, optional): GCP bucket name, if None, trying to read paths from
            local.
        max_workers (int, optional): Num workers to multithread on. If None
            automatically set by concurrent package (see doc) Defaults to None.
        max_workers (int, optional): Num workers to multithread on. If None
            automatically set by concurrent package (see doc). Defaults to None.

    Yields:
        Tuple: First value is the result data frame is it exists, else None. Second
            element is the id of the exp if the result data frame is None. Exactly one
            of the two element is None.
    """
    with tqdm(total=len(paths)) as pbar:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(_method, *(path, bucket)) for path in paths]

            for future in concurrent.futures.as_completed(futures):
                pbar.update(1)
                yield future.result()


def _load_experiment(
    experiment_folder: Path, bucket: Optional[str]
) -> tuple[Optional[pd.DataFrame], Optional[str]]:
    """Load results from the given path.

    Args:
        experiment_folder (Path): Folder where to find the experiment results
            (results.csv and params.pkl)
        bucket (str, optional): Bucket if path is from a gcp bucket.

    Returns:
        tuple[pd.DataFrame, str]d.DataFrame, Any]: First value is the result data frame
            if it exists, else None. Second element is the id of the exp if the result
            data frame is None. Exactly one of the two element is None.
    """
    result_file = str(experiment_folder / "results.csv")
    params_file = str(experiment_folder / "params.pkl")
    id_ = experiment_folder.name

    if not_corrupted(result_file, bucket=bucket) and not_corrupted(
        params_file, bucket=bucket
    ):
        # Getting performances scores
        results, params = load_results(
            result_file=result_file, params_file=params_file, bucket=bucket
        )
        # For experiments that fail the results csv is empty.
        if len(results) > 0:
            results["id"] = id_
            results['pred_file'] = experiment_folder / 'test_predictions.parquet'

            # Getting params of the experiments
            config_df = flatten_nested_config(params)
            repeated_config_df = pd.concat(
                [config_df] * len(results), axis=0
            ).reset_index()
            res_df = pd.concat([results, repeated_config_df], axis=1)
            return (res_df, None)

        return (None, id_)

    return (None, id_)


def load_experiment_output(
    path_to_logs: str,
    study_names: list = None,
    bucket_name: str = None,
    start_date: str = None,
    end_date: str = None,
    repeated_holdout: bool = False,
) -> tuple[pd.DataFrame, list]:
    """Load and concatenate all the CV results in a repository.

    Args:
        path_to_logs (str): Path to directory with saved results
        study_names (list): List of study names to load. Defaults to None, in that case
        all studies are loaded.
        bucket_name (str): Bucket name if results are to be loaded from GCP.
            If None, results will be loaded from local IO. Default to None.
        start_date (str): Filter result folders after start date. Defaults to None.
        end_date (str): Filter result folders before end date. Defaults to None.
        repeated_holdout (bool, optional): Whether the experiments to load were launched
            using the repeated_holdout script. Defaults to False

    Returns:
        pd.DataFrame: DataFrame with all performance scores
        list: List of experiment IDs with missing results files
    """

    if bucket_name is not None:
        client = gcp_storage.Client()
        bucket = client.bucket(bucket_name)

    else:
        bucket = None
        client = None

    experiments_folders = get_experiment_folders(
        path_to_logs=path_to_logs,
        study_names=study_names,
        bucket=bucket,
        client=client,
        start_date=start_date,
        end_date=end_date,
        repeated_holdout=repeated_holdout,
    )
    res_list = []
    corrupted_exps = []

    for exp in _multi_thread_load_experiment(
        _load_experiment, experiments_folders, bucket
    ):
        if exp[0] is not None:
            res_list.append(exp[0])
        else:
            corrupted_exps.append(exp[1])

    corrupted_exps_unique = np.unique(corrupted_exps)

    if len(res_list) == 0:
        logger.info(
            "No experiments found. Check the files in"
            " the path you provided are correct."
        )
        return None

    master_table = pd.concat(res_list).set_index("id")

    if not repeated_holdout:
        return master_table, corrupted_exps_unique

    return master_table, corrupted_exps_unique


def _select_latest_holdout_test_tasks(master_table: pd.DataFrame) -> pd.DataFrame:
    """Filter the given table to only select latest test tasks for each seed.

    Args:
        master_table (pd.DataFrame): Result of the load_experiment_output function.

    Returns:
        pd.DataFrame: Filtered table.
    """
    master_table["date"] = master_table.index.str[-19:]
    master_table["complete_study_name"] = master_table.index.str[:-20]
    master_table["full_id"] = (
        master_table.index + '_' + master_table["test_split_random_seed"].astype(str)
    )
    master_table = master_table.set_index("full_id")

    test_task = master_table[master_table.test_metric.notna()]
    to_keep = (
        test_task.groupby(["complete_study_name", "test_split_random_seed"])
        .agg({"date": "max"})
        .reset_index()
    )
    to_keep = (
        to_keep["complete_study_name"]
        + '_'
        + to_keep["date"]
        + '_'
        + to_keep["test_split_random_seed"].astype(str)
    )

    latest_test_task = test_task.loc[to_keep].copy()

    latest_test_task_agg = (
        latest_test_task.dropna(axis=1, how="all")
        .drop(["REPEAT", "test_metric"], axis=1)
        .drop_duplicates()
    )
    latest_test_task_agg["test_metric"] = latest_test_task.groupby(
        latest_test_task.index
    )["test_metric"].mean()
    latest_test_task_agg["test_metric_95_min"] = latest_test_task.groupby(
        latest_test_task.index
    )["test_metric"].quantile(0.025)
    latest_test_task_agg["test_metric_95_max"] = latest_test_task.groupby(
        latest_test_task.index
    )["test_metric"].quantile(0.975)

    return latest_test_task_agg


def _pull_last_dates_cv(series: pd.Series, nbr_of_trials: int = 50):
    """Custom function to return to 50 latest elements of a pd.Series (dates here)."""
    return series.sort_values(ascending=False).head(nbr_of_trials)


def _select_latest_holdout_cv_results(
    master_table: pd.DataFrame, nbr_of_trials: int = 50
) -> pd.DataFrame:
    """Filter the given table to only select latest cv tasks for each seed.

    Args:
        master_table (pd.DataFrame): Result of the load_experiment_output function.
        nbr_of_trials (int): Number of last trials to pull should match the n_trials of
            the grid or 50 if TPE.

    Returns:
        pd.DataFrame: Filtered table.
    """
    master_table["date"] = master_table.index.str[-19:]
    master_table["complete_study_name"] = master_table.index.str[:-20]

    cv_task = master_table[~master_table.test_metric.notna()]
    dates_cv = (
        cv_task.groupby(
            ["complete_study_name", "REPEAT", "SPLIT", "test_split_random_seed"]
        )
        .agg({"date": lambda series: _pull_last_dates_cv(series, nbr_of_trials)})
        .date.tolist()
    )
    # Regroup all subarrays together
    merged_dates_cv = []
    for sub_array in dates_cv:
        merged_dates_cv.extend(sub_array)
    latest_cv_task = cv_task[cv_task.date.isin(merged_dates_cv)].copy()

    return latest_cv_task, cv_task


def load_benchmark_holdout_latest_cv_tasks(
    master_table: pd.DataFrame = None,
    path_to_logs: str = None,
    study_names: list = None,
    bucket_name: str = None,
    start_date: str = None,
    end_date: str = None,
    nbr_of_trials: int = 50,
    params_to_look_at: list[str] = None,
):
    """For the repeated holdout benchmark, load the latest test tasks result for each
    seed.

    Args:
        master_table (pd.DataFrame): Result of the load_experiment_output function.
        path_to_logs (str): Path to directory with saved results
        study_names (list): List of study names to load. Defaults to None, in that case
        all studies are loaded.
        bucket_name (str): Bucket name if results are to be loaded from GCP.
            If None, results will be loaded from local IO. Default to None.
        start_date (str): Filter result folders after start date. Defaults to None.
        end_date (str): Filter result folders before end date. Defaults to None.
        nbr_of_trials (int): Number of last trials to pull should match the n_trials of
            the grid or 50 if TPE.
        params_to_look_at (list[str]): List of params to analyze

    Returns:
        _type_: _description_
    """
    if master_table is None:
        master_table, _ = load_experiment_output(
            path_to_logs=path_to_logs,
            study_names=study_names,
            bucket_name=bucket_name,
            start_date=start_date,
            end_date=end_date,
            repeated_holdout=True,
        )

    latest_cv_task, cv_task = _select_latest_holdout_cv_results(
        master_table=master_table, nbr_of_trials=nbr_of_trials
    )

    print(cv_task.shape)
    print(latest_cv_task.shape)
    print("Should 2500 for the last one")

    if params_to_look_at:
        types_param = latest_cv_task[params_to_look_at].dtypes
        params_to_look_at = []
        for param in list(types_param.index):
            if types_param[param] == 'O':
                latest_cv_task[param + '_str'] = latest_cv_task[param].astype('str')
                params_to_look_at.append(param + '_str')
            if types_param[param] == 'bool':
                latest_cv_task[param + '_int'] = latest_cv_task[param].astype('int')
                params_to_look_at.append(param + '_int')
            else:
                params_to_look_at.append(param)
        results_agg = latest_cv_task.groupby(by=params_to_look_at).agg(
            {
                'cv_train_metric': ['mean', 'std'],
                'cv_val_metric': ['mean', 'std', 'count'],
            }
        )

        results_agg = results_agg.sort_values(
            by=[tuple(['cv_val_metric', 'mean'])], ascending=False
        )
        results_agg.reset_index(inplace=True)
    return latest_cv_task


def load_benchmark_holdout_latest_test_task(
    path_to_logs: str,
    study_names: list = None,
    bucket_name: str = None,
    start_date: str = None,
    end_date: str = None,
):
    """For the repeated holdout benchmark, load the latest test tasks result for each
    seed.

    Args:
        path_to_logs (str): Path to directory with saved results
        study_names (list): List of study names to load. Defaults to None, in that case
        all studies are loaded.
        bucket_name (str): Bucket name if results are to be loaded from GCP.
            If None, results will be loaded from local IO. Default to None.
        start_date (str): Filter result folders after start date. Defaults to None.
        end_date (str): Filter result folders before end date. Defaults to None.

    Returns:
        _type_: _description_
    """
    master_table, _ = load_experiment_output(
        path_to_logs=path_to_logs,
        study_names=study_names,
        bucket_name=bucket_name,
        start_date=start_date,
        end_date=end_date,
        repeated_holdout=True,
    )

    latest_test_tasks = _select_latest_holdout_test_tasks(master_table=master_table)

    return latest_test_tasks


def filter_exps_and_load_preds(
    master_table: pd.DataFrame,
    params_to_select: dict,
    bucket: gcp_storage.Bucket,
) -> tuple[pd.DataFrame, str]:
    """Filter a set of experiments to keep only those with saved predictions, and load
    the predictions.

    Parameters
    ----------
    master_table : pd.DataFrame
        dataframe of all the runs of the relevant experiments
    params_to_select : dict
        dictionary of additional filters, mapping parameter names to lists of relevant
        parameter values to keep e.g :
        {'model_config': ['lgbm_regressor', 'hierarchical_model']}
    bucket : gcp_storage.Bucket
        Gcp bucket. If None is passed, files are searched
        locally.

    Returns
    -------
    Tuple[pd.DataFrame, str]:
        Table of prediction vectors + string describing the metric to use

    Raises
    ------
    AssertionError
        If the provided filters result in an empty list of runs
    """
    agg_rules = {column: 'first' for column in master_table.columns}
    agg_rules['test_metric'] = 'sum'
    # average the test metric of the bootstrapped predictions. The other columns will be
    # constant but they would be dropped by default, so to keep them we use the policy
    # 'first'

    params_to_select["save_predictions"] = [True]
    params_to_select["train_test_no_cv"] = [True]
    # we save preds only in train/test experiments, not CV cases

    for param in params_to_select.keys():
        # filter experiments according to specified parameters
        master_table = master_table[master_table[param].isin(params_to_select[param])]

    if len(master_table.index) == 0:
        raise AssertionError("No experiments to ensemble with the specified filters")

    to_ensemble = (
        master_table.groupby(['test_split_random_seed', 'study_name'])
        .agg(agg_rules)
        .dropna(how='all', axis=1)
    )

    preds = {}
    for index, file in to_ensemble['pred_file'].items():
        logger.info(file)
        if bucket:
            pred = pd.read_parquet(f"gs://{bucket.name}/{file}")
        else:
            pred = pd.read_parquet(file)
        pred = pred.rename(mapper={'y_pred_0': 'y_pred'}, axis=1)
        # this is required to be consistent between survival and essentiality naming

        index_true = tuple(['y_true'] + list(index))
        index_pred = tuple(['y_pred'] + list(index))
        # the intex contains the study name and index of the repeat

        preds[index_pred] = pred['y_pred']
        preds[index_true] = pred['y_true']

    preds_df = pd.DataFrame.from_dict(preds, orient='index')
    preds_df.index.set_names(['is_pred', 'holdout_idx', 'study_name'], inplace=True)
    metric = master_table["task.metric._target_"].iloc[0]
    return preds_df, metric


def compute_ensemble_metrics(
    exp_results: pd.DataFrame,
    metric_str: str,
) -> pd.DataFrame:
    """Average predictions and compute the performance metric of the resulting ensemble
    prediction.

    Parameters
    ----------
    exp_results : pd.DataFrame
        dataframe of predictions of the relevant experiments, output by
        filter_exp_and_load_preds function
    metric_str: str
        string describing the metric to use, e.g spearmanr

    Returns
    -------
    pd.DataFrame
        Performance metrics of the individual models and ensemble model.
    """
    metric_dict = {'_target_': metric_str, '_partial_': True}
    metric = instantiate(metric_dict)

    ensemble_results = exp_results.groupby(['holdout_idx', 'is_pred']).mean()
    study_names = list(exp_results.index.unique('study_name'))
    indices = ensemble_results.index.unique('holdout_idx')

    metrics = {
        (study_name, idx): []
        for idx in indices
        for study_name in study_names + ["ensemble"]
    }

    for holdout_idx in indices:
        for study_name in study_names:  # performance of each study
            y_test = exp_results.loc['y_true', holdout_idx, study_name].dropna().values
            y_pred_test = (
                exp_results.loc['y_pred', holdout_idx, study_name].dropna().values
            )
            metrics[study_name, holdout_idx] = compute_bootstrapped_metrics(
                y_true=y_test,
                y_pred=y_pred_test,
                metric=metric,
                n_bootstraps=1000,
                sampling_rate=1.0,
            )

        # performance of ensemble results
        y_test = ensemble_results.loc[holdout_idx, 'y_true'].dropna().values
        y_pred_test = ensemble_results.loc[holdout_idx, 'y_pred'].dropna().values
        metrics['ensemble', holdout_idx] = compute_bootstrapped_metrics(
            y_pred=y_pred_test, y_true=y_test, metric=metric
        )
    return pd.DataFrame(metrics)


def get_best_cfg(path_to_logs, rename_best_cfg_folder=False):
    """Get the configuration associated with the best CV result in a folder.

    Args:
        path_to_logs (str): path to directory with saved results
        rename_best_cfg_folder(bool): if True, append '_best_params' to the folder name
            of the best config

    Returns:
        DictConfig: loaded configuration from the corresponding .pkl file
    """
    res = load_experiment_output(path_to_logs)

    best_experiment = (
        res[0]
        .reset_index()
        .groupby('id')
        .agg({'cv_val_metric': "mean"})
        .idxmax()
        .values[0]
    )

    best_parameters_path = path_to_logs + "/" + best_experiment + "/" + "params.pkl"

    cfg = load_pickle(best_parameters_path)

    if rename_best_cfg_folder:
        best_folder = os.path.dirname(best_parameters_path)
        os.rename(best_folder, best_folder + '_best_params')

    return cfg


def get_test_metric(path_to_logs):
    """Get the test metric in a folder.

    Args:
        path_to_logs (str): path to directory with saved results

    Returns:
        float: test metric
    """
    res = load_experiment_output(path_to_logs)

    # Nan values (from cross-validation) are automatically excluded
    return res[0].test_metric.mean()


def load_hp_outputs(
    study_name=str,
    bucket_name: str = "omics-rpz-results",
    trial_aggregate: bool = True,
    cohorts: Optional[list] = None,
    drop_extra_columns: Optional[list] = None,
) -> pd.DataFrame:
    """Load all results from the bucket for an HP search (particularly useful for WP2),
    and output result in the format of a dataframe.

    Parameters
    ----------
    study_name : str
        Name of the study to load
    bucket_name : str, optional
        Name of the bucket where the results are, by default "omics-rpz-results"
    trial_aggregate : bool, optional
        Aggregate metrics for all runs of a same trial (for all n_repeats_cv and
        cv_splits), by default True
    cohorts : Optional[list], optional
        List of cohorts to keep, by default None
    drop_extra_columns : Optional[list], optional
        columns to eventually drop before aggregating, by default None. This can be
        useful when some parameters have changed but you know that it does not affect
        the experiment at all (for instance when a name has changed from n_epochs to
        num_epochs).

    Returns
    -------
    pd.DataFrame
        parameters and results of the HP experiment.
    """
    # Load data from bucket
    master_table, corrupted_exps_unique = load_experiment_output(
        path_to_logs="results/", study_names=[study_name], bucket_name=bucket_name
    )
    logger.info(
        f"{len(master_table)} experiments loaded from the bucket {bucket_name} for the"
        f" HP {study_name}."
    )
    logger.info(f"{len(corrupted_exps_unique)} corrupted experiments found.")

    # Aggregate if required per trial (default behavior)
    drop_extra_columns = drop_extra_columns if drop_extra_columns is not None else []
    drop_columns = [
        "REPEAT",
        "SPLIT",
        "nb_samples_train",
        "nb_samples_val",
        "index",
        "cv_train_metric",
        "cv_val_metric",
    ] + drop_extra_columns

    drop_columns = [drop_c for drop_c in drop_columns if drop_c in master_table]

    df_results = master_table.copy()
    df_results["trial_name"] = df_results.index
    if trial_aggregate:
        logger.info("Aggregate dataframe per trial.")
        df_results = (
            df_results.drop(drop_columns, axis=1).dropna(axis=1).drop_duplicates()
        )
        df_results["mean_cv_train_metric"] = master_table.groupby(master_table.index)[
            "cv_train_metric"
        ].mean()
        df_results["std_cv_train_metric"] = master_table.groupby(master_table.index)[
            "cv_train_metric"
        ].std()
        df_results["mean_cv_val_metric"] = master_table.groupby(master_table.index)[
            "cv_val_metric"
        ].mean()
        df_results["std_cv_val_metric"] = master_table.groupby(master_table.index)[
            "cv_val_metric"
        ].std()

    # Drop parameters columns with no variance
    columns_to_keep = [
        col_name for col_name in df_results if len(set(df_results[col_name])) > 1
    ]
    df_results = df_results[columns_to_keep]

    # Check for duplicate experiments
    n_initial = len(df_results)
    params_col = [col_name for col_name in df_results.columns if '.' in col_name]
    if not trial_aggregate:
        params_col += ["REPEAT", "SPLIT"]
    df_results = df_results.sort_values(["trial_name"], ascending=True)
    df_results = df_results.drop_duplicates(subset=params_col, keep="last")
    n_after = len(df_results)
    logger.info(f"{n_initial - n_after} duplicates have been removed.")

    if (
        (cohorts is not None)
        and ("task.data.dataset.cohorts" in df_results)
        and all(df_results["task.data.dataset.cohorts"].apply(len) == 1)
    ):
        df_results["task.data.dataset.cohorts"] = df_results[
            "task.data.dataset.cohorts"
        ].apply(lambda x: x[0])
        df_results = df_results.query("`task.data.dataset.cohorts` in @cohorts")

    return df_results


def get_clinical():
    """Get the clinical data from TCGA to get the cohort of every sample for the
    compute_cindex_per_cohort function."""
    clinical = pd.DataFrame()

    for cohort in ALL_TCGA_COHORTS:
        path_to_clinical = TCGA_PATHS["CLINICAL"]
        path = Path(path_to_clinical) / f"TCGA-{cohort}_clinical_v2.tsv.gz"
        df_clin = pd.read_csv(path, sep="\t", index_col=0)

        clinical = pd.concat([clinical, df_clin])

    clinical = clinical.set_index("barcode")

    return clinical


def compute_cindex_per_cohort(
    table, clinical, cohorts_of_interest=False, bucket_name='omics-rpz-results'
):
    """Compute the cindex per cohort and per seed for a given pancancer algorithm.

    Args:
        table (pd.DataFrame): the table returned by a csv file inside
            pre_computed_results_light folder.
        clinical (pd.DataFrame): the table returned get_clinical, which is used to get
            the cohort of every sample.
        cohorts_of_interest (bool, optional): whether to only return the cindex for the
            11 cohorts of interest. Defaults to False.
        bucket_name (str): which bucket to take the data from. Defaults to
            omics_rpz_results.

    Returns:
        pd.DataFrame: Dataframe with cindex per cohort (columns) and seeds (rows)
    """
    cindex_per_cohort = pd.DataFrame(
        index=range(10), columns=ALL_TCGA_COHORTS
    )  # each (row, column) is a (seed, cohort)

    for path, seed in zip(table.pred_file, table.test_split_random_seed):
        results = pd.read_parquet(f"gs://{bucket_name}/{path}")
        results["cohort"] = clinical.loc[
            set(clinical.index).intersection(set(results.index)), "project_id"
        ]

        for cohort in results["cohort"].value_counts().index:
            results_cohort = results.loc[results.cohort == cohort]
            if (
                len(results_cohort) > 100
            ):  # if not enough samples, then useless to compute cindex
                cindex = compute_cindex(
                    results_cohort["y_true"], results_cohort["y_pred_0"]
                )

            cindex_per_cohort.loc[seed, cohort.split("-")[1]] = cindex

    if cohorts_of_interest:
        cindex_per_cohort = cindex_per_cohort[OS_COHORTS]

    return cindex_per_cohort
