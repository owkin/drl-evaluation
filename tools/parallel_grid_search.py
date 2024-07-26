"""Wrapper to launch jobs with Ray."""
from typing import Optional

import hydra
import ray
from benchmark import run_experiment
from hydra.core.hydra_config import HydraConfig
from loguru import logger
from omegaconf import DictConfig, OmegaConf, open_dict

from omics_rpz.utils import seed_everything


@ray.remote(max_calls=1)
def run_repeated_gridsearch(
    cfg: DictConfig,
    hydra_cfg: DictConfig,
    hp_search: bool = True,
    repeat_id: int = 0,
) -> Optional[float]:
    """Remote Execution of the benchmark.py main function for a specific seed of the CV.
    Load config with Hydra and modify the config to execute one repeat at repeat_id
    Meant to be run in a loop through seeds for repeated cv.

    Args:
        cfg (DictConfig): Configuration for the experiments
        hydra_cfg (DictConfig): Hydra configuration
        hp_search (bool, optional): If we are doing an HP search or not.
            Defaults to True.
        repeat_id (int, optional): Repetition ID, ie the seed we use for the CV split.
            Defaults to 0.

    Returns:
        Optional[float]: _description_
    """

    seed_everything(2023)

    assert not (hp_search is True and cfg["train_test_no_cv"] is True), (
        "Hyperparameter Search must be performed with cross-validation."
        "In config file please change train_test_no_cv to False."
    )

    # pass global config params to the task
    with open_dict(cfg):
        cfg.task.test_split_random_seed = cfg["test_split_random_seed"]
        cfg.task.split_ratio = cfg["test_split_ratio"]
        cfg.task.n_random_seed_cv_start_at = repeat_id
        cfg.task.n_repeats_cv = 1
        cfg.study_name = cfg.study_name + f"_seed_{repeat_id}"

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


@hydra.main(version_base=None, config_path="../conf/", config_name="config")
def main(cfg: DictConfig = None):
    """Executes benchmark using Ray to parallelize executions of multiple random seeds.

    Args:
        cfg (DictConfig): Configuration for the experiment as in config.yaml
    """
    hydra_cfg = OmegaConf.to_container(HydraConfig.get().runtime.choices)
    hyperparam_search = HydraConfig.get().overrides.hydra[0] == "hydra.mode=MULTIRUN"

    with open_dict(cfg):
        num_repeats = cfg.task.n_repeats_cv
        starting_seed_cv = cfg.task.n_random_seed_cv_start_at
    # Initialize ray cluster explicitly
    num_cpu_available = cfg["num_cpus_for_ray"]
    num_gpu_available = cfg["num_gpus_for_ray"]
    cpu_per_repeat = cfg["cpu_per_repeat"]
    gpu_per_repeat = cfg["gpu_per_repeat"]

    try:
        ray.init(num_cpus=num_cpu_available, num_gpus=num_gpu_available)
    except ValueError:
        logger.info("Ray cluster already exists")

    # Looping through seed following advice from
    # https://docs.ray.io/en/latest/ray-core/patterns/ray-get-loop.html
    exps = []
    for i in range(num_repeats):
        logger.info("Running :", i)
        repeat_id = starting_seed_cv + i

        exps.append(
            run_repeated_gridsearch.options(
                num_cpus=cpu_per_repeat, num_gpus=gpu_per_repeat
            ).remote(
                cfg,
                hydra_cfg=hydra_cfg,
                hp_search=hyperparam_search,
                repeat_id=repeat_id,
            )
        )

    # Calling ray.get only once
    ray.get(exps)
    ray.shutdown()


if __name__ == "__main__":
    try:
        main()
    except hydra.errors.InstantiationException as e:
        logger.error(e)
        logger.error(
            "Check that the Hydra sweeper sampler points to the correct class, as"
            " specified in config.yaml !"
        )
        raise
