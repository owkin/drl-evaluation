"""Launch a given task using a given representation models and hyperparameter."""
import os
import sys
from datetime import datetime
from itertools import zip_longest
from pathlib import Path
from typing import Optional, Union

import hydra
import mlflow
import numpy as np
import pandas as pd
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from loguru import logger
from omegaconf import DictConfig, OmegaConf, open_dict
from utils import drop_ensemble_field, solve_cfg, solve_ensemble_cfg

from omics_rpz.constants import REPEATED_HOLDOUT_PATH
from omics_rpz.experiment_tracking import git_interface, storage
from omics_rpz.utils import log_params_recursive, save_pickle, seed_everything
from omics_rpz.utils.io import define_remote_prefix


@hydra.main(version_base=None, config_path="../conf/", config_name="config")
def main(cfg: DictConfig = None) -> Optional[float]:
    """Run the task and save the results in the logging folder.

    Parameters are chosen thanks to .yaml files in the ../conf folder

    Args:
        cfg (DictConfig): Configuration for the experiment as in config.yaml
    """
    seed_everything(2023)
    hydra_cfg = OmegaConf.to_container(HydraConfig.get().runtime.choices)
    hyperparam_search = HydraConfig.get().overrides.hydra[0] == "hydra.mode=MULTIRUN"

    assert not (hyperparam_search is True and cfg["train_test_no_cv"] is True), (
        "Hyperparameter Search must be performed with cross-validation."
        "In config file please change train_test_no_cv to False."
    )

    # pass global config params to the task
    with open_dict(cfg):
        cfg.task.test_split_random_seed = cfg["test_split_random_seed"]
        cfg.task.split_ratio = cfg["test_split_ratio"]

    outcome_sweep = -1

    if cfg["train_test_no_cv"]:
        # Run the experiment without cross-validation for final test
        with open_dict(cfg):
            cfg.task.train_test_no_cv = True
        run_experiment(cfg, hydra_cfg)
    else:
        # Run the experiment with cross-validation
        outcome_sweep = run_experiment(cfg, hydra_cfg)
    return outcome_sweep


# pylint: disable=too-many-statements
def run_experiment(cfg: DictConfig, hydra_cfg: DictConfig) -> Optional[float]:
    """Run one or multiple experiments and ensemble their results if multiple.

    Args:
        cfg (DictConfig): Configuration for the experiments
        hydra_cfg (DictConfig): Hydra configuration

    Returns:
        float: global metric to optimize
    """
    # Get parameters
    solve_cfg(cfg)
    task_cfg = cfg["task"]
    rpz_cfg = cfg["representation_model"]
    pred_cfg = cfg["prediction_model"]
    logging_path = cfg["logging_path"]
    mlflow_path = cfg["mlflow_path"]

    is_ensembling_experiment = (
        (cfg['task']['data']['dataset']['ensemble'] is not None)
        or (cfg['prediction_model']['ensemble'] is not None)
        or (cfg['representation_model']['ensemble'] is not None)
    )
    # if any of those is not None, the experiment is an ensembling experiment

    # Define experiment name

    exp_p = (
        cfg["study_name"],
        hydra_cfg["task"],
        hydra_cfg["representation_model"],
        hydra_cfg["prediction_model"],
        datetime.now().strftime('%Y-%m-%d_%H:%M:%S'),
    )
    experiment_name = f"{exp_p[0]}_{exp_p[1]}_{exp_p[2]}_{exp_p[3]}_{exp_p[4]}"

    if hydra_cfg["prediction_model"] == "lgbm_regressor":
        # retrieve CPU per cohort in global config
        # and update LGBM config
        cfg["prediction_model"].update({"n_jobs": cfg["cpu_per_repeat"]})

    # Create experiment folder
    experiment_folder = Path(logging_path).joinpath(experiment_name)
    experiment_folder.mkdir(parents=True, exist_ok=True)

    # Close previous log files, if any (e.g. in a repeated holdout, close log file
    # from the previous execution), then enable logging to stderr
    logger.remove()
    logger.add(sys.stderr)

    # Create log file
    path = os.path.join(experiment_folder, f"{experiment_name}.log")
    logger.add(path)

    # Log experiment info
    logger.info(f"Experiment name: {experiment_name}")
    logger.info(f"Experiment folder: {experiment_folder}")
    logger.info("\n")

    # Log params
    logger.info(cfg)
    logger.info("\n")

    # Store params
    path = os.path.join(experiment_folder, "params.pkl")
    save_pickle(path, cfg)

    # MLFlow: Init
    mlflow.set_tracking_uri(mlflow_path)
    mlflow.set_experiment(hydra_cfg["task"])

    with mlflow.start_run(run_name=experiment_name):
        # MLFlow: Log parameters
        log_params_recursive(cfg)

        # pylint: disable=broad-except
        try:
            y_preds, y_true, df_results = instantiate_and_run(
                task_cfg=task_cfg,
                rpz_cfg=rpz_cfg,
                pred_cfg=pred_cfg,
                path=path,
                ensemble=is_ensembling_experiment,
                save_predictions=cfg['save_predictions'],
            )
        except Exception as exception:
            logger.exception(exception)
            y_preds, y_true, df_results = {}, {}, pd.DataFrame()

        df_results["rpz_config"] = hydra_cfg["representation_model"]
        df_results["model_config"] = hydra_cfg["prediction_model"]
        df_results["task_config"] = hydra_cfg["task"]

        if is_ensembling_experiment:
            ensemble_metric_per_fold = compute_and_log_ensemble_metric(
                cfg, y_preds, y_true
            )
            df_results["ensemble_metric"] = ensemble_metric_per_fold * len(y_preds)
            # one ensemble metric per fold
            df_results["mean_ensemble_metric"] = np.mean(ensemble_metric_per_fold)

        # Store results
        df_results.to_csv(Path(experiment_folder) / "results.csv", index=False)

    if cfg["track"]:
        if git_interface.tag_already_exists(cfg["git-tag"]):
            logger.info("The experiment code version has already been uploaded.")
        else:
            logger.info(f"Storing the code version as the git tag: {cfg['git-tag']}")
            git_interface.store_reference(cfg["git-tag"])

        remote_prefix = define_remote_prefix(
            experiment_folder=experiment_folder,
            keep_tree_from=f"/{REPEATED_HOLDOUT_PATH}",
        )

        storage.upload_folder(
            experiment_folder, cfg["results_bucket"], remote_prefix=remote_prefix
        )

    if "cv_val_metric" in df_results:
        # If cv_val_metric in df, then it is a supervised task and
        # the metric can be returned for optuna optimisation
        return df_results["cv_val_metric"].mean()

    if len(df_results) == 0:
        return -1

    return -1


def instantiate_and_run(
    task_cfg: DictConfig,
    rpz_cfg: DictConfig,
    pred_cfg: DictConfig,
    path: str,
    ensemble: bool,
    save_predictions=False,
) -> tuple[Union[list[dict], dict], dict, pd.DataFrame]:
    """Instantiate and run one or multiple tasks.

    Parameters
    ----------
    task_cfg : DictConfig
        task config
    rpz_cfg : DictConfig
        representation model config
    pred_cfg : DictConfig
        prediction model config
    path : str
        path to save results
    ensemble : bool
        whether to do ensembling or not
    save_predictions : bool
        whether to do save prediction when pertinent :
            - task=survival_prediction_tcga_task and train_test_no_cv=True
            - task=task=gene_essentiality and train_test_no_cv=False
            - task=task=gene_essentiality and train_test_no_cv=True
        or not. (Default to False).

    Returns
    -------
    tuple[Union[list, dict], dict, pd.DataFrame]
        predictions, labels and dataframe of results. With ensembling the prediction is
        a list of dictionaries, otherwise it is a single dictionary
    """
    if not ensemble:
        # Drop 'ensemble' key because it is not used in this case
        # otherwise when instantiating, the 'ensemble' argument will not be recognized
        drop_ensemble_field(task_cfg['data']['dataset'])
        drop_ensemble_field(rpz_cfg)
        drop_ensemble_field(pred_cfg)

        # Instantiate task
        task = instantiate(
            task_cfg,
            rpz_model=rpz_cfg,
            pred_model=pred_cfg,
            logging_path=path,
            _recursive_=False,
            save_predictions=save_predictions,
        )

        # Run task
        y_pred, y_true, df_results = task.run()
        return y_pred, y_true, df_results

    data_cfg_list = solve_ensemble_cfg(task_cfg['data']['dataset'])
    rpz_cfg_list = solve_ensemble_cfg(rpz_cfg)
    pred_cfg_list = solve_ensemble_cfg(pred_cfg)

    y_preds, results_ensemble = [], []
    for i, (inner_data_cfg, inner_rpz_cfg, inner_pred_cfg) in enumerate(
        zip_longest(data_cfg_list, rpz_cfg_list, pred_cfg_list)
    ):
        # zip_longest in case one list is longer than the others, then replace
        # None with first value (e.g if we ensemble 2 pred models with the same rpz
        # model, we should repeat rpz model config twice)
        inner_data_cfg = inner_data_cfg or data_cfg_list[0]
        inner_rpz_cfg = inner_rpz_cfg or rpz_cfg_list[0]
        inner_pred_cfg = inner_pred_cfg or pred_cfg_list[0]
        task_cfg['data']['dataset'] = inner_data_cfg

        # Instantiate task
        task = instantiate(
            task_cfg,
            rpz_model=inner_rpz_cfg,
            pred_model=inner_pred_cfg,
            logging_path=path,
            _recursive_=False,
            save_predictions=save_predictions,
        )

        # Run task
        y_pred, y_true, df_results_model = task.run()
        y_preds.append(y_pred)
        df_results_model['model_index'] = i
        results_ensemble.append(df_results_model)
    df_results = pd.concat(results_ensemble, axis=0)
    return y_preds, y_true, df_results


def compute_and_log_ensemble_metric(
    cfg: DictConfig, y_preds: list[dict], y_true: dict
) -> list:
    """Compute the performance metric of an ensemble of predictions.

    Parameters
    ----------
    cfg : DictConfig
        experiment config
    y_preds : list[dict]
        predictions of individual models (list indices) per fold (dictionary keys)
    y_true : dict
        ground truth on each fold (one fold per key)

    Returns
    -------
    list
        list of ensemble metric per fold
    """
    ensemble_preds = {}
    # Need to instantiate the metric outside of the loop on tasks
    metric = instantiate(cfg.task.metric.copy())

    # Compute mean of model predictions per fold
    metrics_folds = []
    for fold_idx in y_preds[0].keys():
        ensemble_preds[fold_idx] = np.mean(
            [y_preds[i][fold_idx].flatten() for i in range(len(y_preds))], axis=0
        )
        fold_metric = metric(y_true[fold_idx], ensemble_preds[fold_idx].flatten())
        if not isinstance(fold_metric, float):
            # some metrics like spearmanr output a list with (value, p_value)
            # we keep the value only
            fold_metric = fold_metric[0]
        metrics_folds.append(fold_metric)

    mean_ensemble_metric, std_ensemble_metric = np.mean(metrics_folds), np.std(
        metrics_folds
    )
    logger.info(
        "Ensemble mean CV metric ="
        f" {mean_ensemble_metric:.3f} ({std_ensemble_metric:.3f})"
    )
    mlflow.log_metric("mean_ensemble_metric", mean_ensemble_metric)
    mlflow.log_metric("std_ensemble_metric", std_ensemble_metric)

    return metrics_folds


if __name__ == "__main__":
    main()
