"""Launch experiment with the repeated hold-out cross validation."""
import argparse
import subprocess
import sys
from argparse import Namespace
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import ray
import yaml
from load_outputs import get_best_cfg, get_test_metric
from loguru import logger
from omegaconf import OmegaConf

from omics_rpz.constants import REPEATED_HOLDOUT_PATH
from omics_rpz.experiment_tracking import storage
from omics_rpz.utils.io import define_remote_prefix
from tools.repeated_holdout_best_saved_models import best_AE, best_LGBM, best_MLP

running_folder = Path(__file__).resolve().parent


def run_one_holdout_experiment(
    sys_args: list, parsed_args: Namespace, start_time: str, holdout_seed: int
):
    """Run one iteration of repeated holdout for a given seed with Ray.

    Args:
        sys_args (list): System Arguments passed to the script
        parsed_args (argparse.Namespace): object containing some parsed arguments
        start_time (str): Timestamp at the beginning of the script
        holdout_seed (int): the repeat id to run in the iteration

    Returns:
        float: test metric
    """
    args = sys_args.copy()

    # if use_best_AE and/or use_best_MLP, update args here for this repetition
    update_args_best_model(args, parsed_args, holdout_seed)

    study_name_str = parsed_args.study_name

    logging_path = (
        './outputs'
        f'/{REPEATED_HOLDOUT_PATH}_{study_name_str}_{start_time}_{holdout_seed}'
    )
    print(logging_path)
    args.append(f'logging_path={logging_path}')
    args.append(f'test_split_random_seed={holdout_seed}')

    # if use_no_cv_only=True, jump CV
    if not parsed_args.use_no_cv_only:
        args.append('--multirun')

        # execute benchmark.py with --multirun and train_test_no_cv=False
        # pylint: disable=subprocess-run-check
        ret = subprocess.run(["python", "tools/benchmark.py"] + args, check=False)
        check_benchmark_exit_status(ret.returncode)

        # identify the best set of hyperparameters from this grid search
        best_cfg = get_best_cfg(logging_path, rename_best_cfg_folder=True)

        # remove --multirun (last appended arg), and update args to reflect best_cfg
        del args[-1]
        update_args_to_test(args, best_cfg)

    args.append('train_test_no_cv=True')

    # execute traditional benchmark.py with train_test_no_cv=True
    ret = subprocess.run(["python", "tools/benchmark.py"] + args, check=False)
    check_benchmark_exit_status(ret.returncode)
    return get_test_metric(logging_path)


@ray.remote(max_calls=1)
def remote_run_one_holdout_experiment(
    sys_args: list, parsed_args: Namespace, start_time: str, holdout_seed: int
):
    """Remote execution of run_one_holdout_experiment.

    Args:
        sys_args (list): System Arguments passed to the script
        parsed_args (argparse.Namespace): object containing some parsed arguments
        start_time (str): Timestamp at the beginning of the script
        holdout_seed (int): the repeat id to run in the iteration

    Returns:
        float: test metric
    """
    return run_one_holdout_experiment(sys_args, parsed_args, start_time, holdout_seed)


def main():
    """Execute repeated holdout evaluation pipeline.

    Perform num_repeated_holdout runs of:
        1. Randomly split out train/val and test sets with a different seed.
        2. Perform grid search with cross-validation to find the best hyperparameters.
        3. Use the best hyperparameters to measure performance on the test set.

    The result is num_repeated_holdout performance measures.
    We report mean (SD) and 95% empirical confidence intervals.
    """

    sys_args, parsed_args = get_base_args(sys.argv[1:])

    # Handles arguments that can be specified both in cmd line or in config file
    cfg = OmegaConf.load('./conf/config.yaml')
    num_repeated_holdout = load_params_cmd_config(
        parsed_args, "num_repeated_holdout", int, cfg
    )
    random_seed_start_at = load_params_cmd_config(
        parsed_args, "random_seed_start_at", int, cfg
    )
    # load resources argument
    num_cpu_available = load_params_cmd_config(
        parsed_args, "num_cpus_for_ray", int, cfg
    )
    num_gpu_available = load_params_cmd_config(
        parsed_args, "num_gpus_for_ray", float, cfg
    )
    cpu_per_repeat = load_params_cmd_config(parsed_args, "cpu_per_repeat", int, cfg)
    gpu_per_repeat = load_params_cmd_config(parsed_args, "gpu_per_repeat", float, cfg)
    use_ray = load_params_cmd_config(parsed_args, "use_ray", bool, cfg)

    study_name_str = parsed_args.study_name
    start_time = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')

    if use_ray:
        print("Initialize cluster")
        # initiate ray cluster
        try:
            ray.init(num_cpus=num_cpu_available, num_gpus=num_gpu_available)
        except ValueError:
            print("Ray cluster already exists")

    # loop over K runs, changing logging_path and test_split_random_seed
    exps = []
    test_metrics = []

    for holdout_seed in range(
        random_seed_start_at, num_repeated_holdout + random_seed_start_at
    ):
        print("Launching repeated with seed: ", holdout_seed)
        if use_ray:
            exps.append(
                remote_run_one_holdout_experiment.options(
                    num_cpus=cpu_per_repeat, num_gpus=gpu_per_repeat
                ).remote(sys_args, parsed_args, start_time, holdout_seed)
            )
        else:
            test_metrics.append(
                run_one_holdout_experiment(
                    sys_args, parsed_args, start_time, holdout_seed
                )
            )

    if use_ray:
        # call ray.get to get results
        test_metrics = ray.get(exps)
        ray.shutdown()

    # after num_repeated_holdout runs, aggregate test set measures, compute mean/std, CI
    mean_test_metrics = np.mean(test_metrics)
    std_test_metrics = np.std(test_metrics)
    ci_95_low, ci_95_high = np.quantile(test_metrics, 0.025), np.quantile(
        test_metrics, 0.975
    )

    # log aggregated test metrics info
    logging_path = (
        f'./outputs/{REPEATED_HOLDOUT_PATH}_{study_name_str}_{start_time}_test_metrics'
    )
    folder_test_metrics = Path(logging_path)
    folder_test_metrics.mkdir()
    logger.add(logging_path + "/test_metrics.log")
    logger.info(f"Repeated holdout finished after {num_repeated_holdout} repetitions")
    logger.info(
        f"Test metric Mean (SD): {mean_test_metrics:.3f} ({std_test_metrics:.3f})"
    )
    logger.info(f"Test metric 95% CI: [{ci_95_low:.3f}, {ci_95_high:.3f}]")
    logger.info("\n")

    # store results on a .csv file
    df_test_metrics = pd.DataFrame(data=test_metrics, columns=['test_metric'])
    df_test_metrics.index.names = ['random_seed']
    df_test_metrics["sys_args"] = [' '.join(sys_args)] * len(test_metrics)
    df_test_metrics.to_csv(logging_path + "/test_metrics.csv")

    with open(running_folder / "../conf/config.yaml", "rb") as cfg:
        static_cfg = yaml.load(cfg, Loader=yaml.loader.SafeLoader)

    if static_cfg["track"]:
        remote_prefix = define_remote_prefix(
            experiment_folder=folder_test_metrics,
            keep_tree_from=f"/{REPEATED_HOLDOUT_PATH}",
        )

        storage.upload_folder(
            folder_test_metrics,
            static_cfg["results_bucket"],
            remote_prefix=remote_prefix,
        )


def load_params_cmd_config(
    parsed_args: Namespace, arg_name: str, arg_type: type, config: OmegaConf
):
    """Load parameters for config when Hydra is not in charge.

    We need to account for both command line and yaml.
    """
    # The if below handles the case when the user passes num_repeated_holdout as a
    # command line argument, and the else applies when there is no such argument and the
    # config has to be extracted from the yaml, and same for resources parameters.
    print(f"Loading {arg_name} from", end=" ")
    try:
        if (arg := getattr(parsed_args, arg_name)) is None:
            raise AttributeError
        if arg_type == bool:
            arg = arg == "True"
        arg = arg_type(arg)
        print(f"from command line: {arg}")
    except (AttributeError, TypeError) as exception:
        arg = config[arg_name]
        print(f"from config: {arg}")
        print(exception)
    return arg


def get_base_args(sys_args: list[str]) -> tuple[list[str], list[str]]:
    """Return sys and parsed args.

    Do basic checks on the args and then get the parsed args and update sys args.

    Args:
        sys_args (list[str]): system arguments

    Returns:
        tuple[list[str],list[str]]: sys_args, parsed_args.
    """
    for arg in sys_args:
        assert not ("multirun" in arg or "train_test_no_cv" in arg), (
            "Please do not use 'multirun' or 'train_test_no_cv' with this script. "
            "This script executes benchmark.py multiple times, automatically using "
            "'multirun' or 'train_test_no_cv' arguments when needed."
        )

    parsed_args = get_parsed_args(
        sys_args,
        [
            "use_best_AE",
            "use_best_MLP",
            "use_best_LGBM",
            "use_best_AE_plus_MLP",
            "use_no_cv_only",
            "train_test_no_cv",
            "study_name",
            "task",
            "task.data.dataset.cohorts",
            "num_repeated_holdout",
            "prediction_model",
            "random_seed_start_at",
            "num_cpus_for_ray",
            "num_gpus_for_ray",
            "use_ray",
            "cpu_per_repeat",
            "gpu_per_repeat",
        ],
    )

    # Drop use_best_AE, use_best_MLP, use_no_cv_only from sys_args, since they are not
    # used by the benchmark script
    sys_args = [
        s
        for s in sys_args
        if "use_best_AE" not in s
        and "use_best_MLP" not in s
        and "use_best_LGBM" not in s
        and "use_no_cv_only" not in s
        and "use_best_AE_plus_MLP" not in s
    ]

    assert parsed_args.study_name, "Please run this script defining a study_name."

    return sys_args, parsed_args


def update_args_to_test(args, best_cfg):
    """Update args from the cross-validation execution to the test-set execution.

    Args:
        args (list[str]): list of arguments, modified in-place
        best_cfg (DictConfig): best cross-validation configuration, from the .pkl file
    """
    for i, arg in enumerate(args):
        if any(substr in arg for substr in ('choice', 'range', 'interval')):
            # truncate string before '='
            arg = arg[: arg.find('=')]

            # split config levels
            if '+' in arg:
                config_levels = arg[1:].split('.')
            else:
                config_levels = arg.split('.')

            # identify best config from CV
            cfg_value = best_cfg
            for level in config_levels:
                cfg_value = cfg_value[level]

            # update this arg
            arg += '=' + str(cfg_value)
            args[i] = arg


# pylint: disable=too-many-branches
def update_args_best_model(args, parsed_args, holdout_seed):
    """Update the argument list using configurations from the best AE/MLP models.

    Args:
        args (list[str]): list of arguments, modified in-place
        parsed_args (argparse.Namespace): object containing some parsed arguments
        holdout_seed (int): seed used to index the best AE/MLP configurations
    """
    # if survival task, also need to specify the cohort
    if (task := parsed_args.task) == 'survival_prediction_tcga_task':
        task = task + "_" + getattr(parsed_args, 'task.data.dataset.cohorts')

    # task is best_AE key, conf is best_AE[task] key, and holdout_seed is the index
    if parsed_args.use_best_AE == 'True':
        for conf in (
            'repr_dim',
            'hidden_n_layers',
            'hidden_n_units_first',
            'hidden_decrease_rate',
            'dropout',
            'batch_size',
            'learning_rate',
        ):
            args.append(
                f"+representation_model.{conf}={best_AE[task][conf][holdout_seed]}"
            )

    # task is best_MLP key, conf is best_MLP[task] key, and holdout_seed is the index
    if parsed_args.use_best_MLP == 'True':
        if parsed_args.prediction_model == 'ensemble_normalizations':
            arg_prefix = "prediction_model.model_configs.mlp_prediction"
        else:
            arg_prefix = "prediction_model"
        for conf in (
            'learning_rate',
            'dropout',
            'batch_size',
            'mlp_hidden',
        ):
            args.append(f"+{arg_prefix}.{conf}={best_MLP[task][conf][holdout_seed]}")

    # in case one wants to concatenate the encoder from the best AE with the best MLP
    if parsed_args.use_best_AE_plus_MLP == 'True':
        if parsed_args.prediction_model == 'ensemble_normalizations':
            arg_prefix = "prediction_model.model_configs.mlp_prediction"
        else:
            arg_prefix = "prediction_model"
        for conf in (
            'learning_rate',
            'dropout',
            'batch_size',
        ):
            args.append(f"+{arg_prefix}.{conf}={best_MLP[task][conf][holdout_seed]}")
        args.append(
            f"+{arg_prefix}.mlp_hidden={best_MLP[task]['mlp_plus_ae'][holdout_seed]}"
        )

    # task is best_LGBM key, conf is best_LGBM[task] key, and holdout_seed is the index
    if parsed_args.use_best_LGBM == 'True':
        for conf in ('learning_rate', 'reg_alpha'):
            args.append(
                f"+prediction_model.{conf}={best_LGBM[task][conf][holdout_seed]}"
            )


def check_benchmark_exit_status(returncode):
    """Checks the exit status of the benchmark child process.

    Args:
        returncode (int): subprocess.CompletedProcess.returncode, 0 means successful.
    """
    assert (
        returncode == 0
    ), f"Error: benchmark.py exit status {returncode} was not successful!"


def get_parsed_args(args, args_to_parse):
    """Convert list of arguments to argparse format, and parse a subset of them.

    Args:
        args (list[str]): list of arguments in Hydra format
        args_to_parse (list[str]): list of arguments to parse

    Returns:
        argparse.Namespace: object containing the parsed arguments
    """
    # first needs to convert arguments from Hydra to argparse format
    new_args = []
    for arg in args:
        key, value = arg.split('=')
        new_args.extend(['--' + key, value])

    # then can use the standard argparse pipeline to parse the desired subset of args
    parser = argparse.ArgumentParser()
    for arg in args_to_parse:
        parser.add_argument(f'--{arg}')

    parsed_args, _ = parser.parse_known_args(new_args)

    return parsed_args


if __name__ == "__main__":
    main()
