"""Launch a given task using a given representation model and hyperparameter on each
TCGA cohort for TCGA prediction tasks."""
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Optional

import hydra
import pandas as pd
from benchmark import run_experiment
from hydra.core.hydra_config import HydraConfig
from load_outputs import load_experiment_output
from loguru import logger
from omegaconf import DictConfig, OmegaConf, open_dict

from omics_rpz.constants import ALL_TCGA_COHORTS


def get_obj_hash(obj):
    """Return a hash from str(obj)."""
    obj_str = str(obj)
    obj_hash = hashlib.md5(obj_str.encode()).hexdigest()
    return obj_hash


def main(cohort: str = None):
    """Wrapper around the main function as the @hydra.main decorator does not allow to
    have arguments for main. This is useful for the benchmark_each_tcga_cohort script.
    Solution found here: https://github.com/facebookresearch/hydra/issues/333.

    Args:
        cohort (str, optional): Cohort to run experiment on. Defaults to None.
    """

    @hydra.main(version_base=None, config_path="../conf/", config_name="config")
    def _main(cfg: DictConfig = None) -> Optional[float]:
        """Run the task with CV and without CV and save the results in the logging
        folder.

        Parameters are chosen thanks to .yaml files in the ../conf folder

        Args:
            cfg (DictConfig): Configuration for the experiment as in config.yaml
        """
        hydra_cfg = OmegaConf.to_container(HydraConfig.get().runtime.choices)

        if cohort:
            cfg.task.data.dataset.cohorts = [cohort]

        with open_dict(cfg):
            cfg.hash = get_obj_hash(cfg)

        cfg_2 = cfg.copy()
        with open_dict(cfg_2):
            cfg_2.task.train_test_no_cv = True

        # Run the experiment with cross-validation
        outcome_sweep = run_experiment(cfg, hydra_cfg)

        # Run the experiment without cross-validation for final test
        run_experiment(cfg_2, hydra_cfg)

        return outcome_sweep

    _main()


def filter_n_best_models(df_hp_all: pd.DataFrame) -> pd.DataFrame:
    """From the outputs of an hyperparameter search, aggregate all results per SPLIT and
    REPEAT, and keeps only the best N_BEST_MODELS_TO_KEEP models.

    Args:
        df_hp_all (pd.DataFrame): output params and scores of the hyperparameter search.

    Returns:
        pd.DataFrame: one line per run for the best N_BEST_MODELS_TO_KEEP models.
    """
    df_hp_all_cv = (
        df_hp_all.query("SPLIT == SPLIT")
        .drop(
            columns=[
                "REPEAT",
                "SPLIT",
                "nb_samples_train",
                "nb_samples_val",
                "index",
                "cv_train_metric",
                "nb_patients_val",
            ],
            errors='ignore',
        )
        .rename({"task.data.dataset.cohorts": "cohort"}, axis=1)
    )
    df_agg_cv = pd.DataFrame(index=df_hp_all_cv.index.unique())

    df_hp_all_not_cv = df_hp_all.query("SPLIT != SPLIT")[["test_metric", "hash"]]
    df_agg_not_cv = pd.DataFrame(index=df_hp_all_not_cv.index.unique())

    # iterate over each column of the original DataFrame
    for col in df_hp_all_cv.columns:
        if df_hp_all_cv[col].dtype == 'float64':
            df_agg_cv[col + '_mean'] = df_hp_all_cv.groupby(df_hp_all_cv.index)[
                col
            ].mean()
            df_agg_cv[col + '_std'] = df_hp_all_cv.groupby(df_hp_all_cv.index)[
                col
            ].std()
        else:
            df_agg_cv[col] = df_hp_all_cv.groupby(df_hp_all_cv.index)[col].apply(
                lambda x: x[0]
            )

    df_agg_cv = df_agg_cv.sort_values(["cv_val_metric_mean"], ascending=False)

    # iterate over each column of the original DataFrame
    for col in df_hp_all_not_cv.columns:
        if df_hp_all_not_cv[col].dtype == 'float64':
            df_agg_not_cv[col + '_mean'] = df_hp_all_not_cv.groupby(
                df_hp_all_not_cv.index
            )[col].mean()
            df_agg_not_cv[col + '_95_max'] = df_hp_all_not_cv.groupby(
                df_hp_all_not_cv.index
            )[col].quantile(0.975)
            df_agg_not_cv[col + '_95_min'] = df_hp_all_not_cv.groupby(
                df_hp_all_not_cv.index
            )[col].quantile(0.025)
        else:
            df_agg_not_cv[col] = df_hp_all_not_cv.groupby(df_hp_all_not_cv.index)[
                col
            ].apply(lambda x: x[0])

    df_agg_cv = df_agg_cv.reset_index().merge(df_agg_not_cv, on="hash", how="left")
    columns_min = ["id", "cohort"] + [
        col for col in df_agg_cv.columns if "metric_" in col
    ]

    df_agg_cv = df_agg_cv[
        columns_min + df_agg_cv.columns.difference(columns_min).tolist()
    ]

    return df_agg_cv


if __name__ == "__main__":
    all_df_results = []

    # It cannot use and overrided logging path in the task file.
    logging_path = OmegaConf.load('./conf/config.yaml')['logging_path']

    # For all cohorts
    for cohort in ALL_TCGA_COHORTS:
        try:
            # Run Hyperparameter Search for all models
            logger.info(f"Start hyperparameter search with cohort {cohort}")
            d_start = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
            main(cohort=cohort)
            d_end = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')

            # Load results for all models
            logger.info(f"Load outputs for cohort {cohort}")
            hp_trials_info = load_experiment_output(logging_path, d_start, d_end)[0]

            # Keep N_BEST_MODELS_TO_KEEP best results
            hp_n_best_info = filter_n_best_models(hp_trials_info)
            all_df_results.append(hp_n_best_info)

        # pylint: disable=broad-except
        except Exception as exception:
            logger.exception(exception)
            logger.info(f"Not able to perform HP search with cohort {cohort}")

    # Save results
    df_all_results = pd.concat(all_df_results, axis=0)
    filename = (
        "results_benchmark_all_cohorts_"
        + datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
        + ".csv"
    )
    df_all_results.to_csv(Path(logging_path) / filename)
