"""Utils functions for config dictionary manipulation."""

import sys
from typing import Union

from loguru import logger
from omegaconf import DictConfig, ListConfig, OmegaConf, open_dict

from omics_rpz.constants import TASK_LABEL_LIST, TASK_LIST
from omics_rpz.experiment_tracking import git_interface


def solve_cfg(cfg: DictConfig):
    """Modify the experiment's model config in-place with correct parameter priority.

    From bottom to top priority :
     - default parameters
     - task-specific parameters
     - label-specific parameters
     - command-line or sweeper parameters (e.g. representation_model.device='cpu')

    Parameters
    ----------
    cfg : DictConfig
        Configuration for the experiment, will be modified in-place.
    """
    task = cfg["task"]["_target_"].split(".")[-1]
    label = cfg["task"].get('label', None)
    label = (
        "Deconvolution"
        if (
            isinstance(label, ListConfig)
            and all("cell_fraction" in sub_label for sub_label in label)
        )
        else label
    )
    solve_tracking_cfg(cfg)
    solve_model_cfg(cfg, "representation_model", task, label)
    solve_model_cfg(cfg, "prediction_model", task, label)


def solve_tracking_cfg(cfg: DictConfig) -> None:
    """Modify the experiment's tracking config in-place with correct parameters.

    Args:
        cfg (DictConfig): Configuration for the experiment, will be modified in-place.
    """
    tag = git_interface.experiment_id(cfg)
    with open_dict(cfg):
        cfg["git-tag"] = tag


def solve_model_cfg(
    cfg: DictConfig, model_type_name: str, task: str, label: str
) -> None:
    """Modify the experiment's model config in-place with correct parameter priority.

    From bottom to top priority :
     - default parameters
     - task-specific parameters
     - label-specific parameters
     - command-line or sweeper parameters (e.g. representation_model.device='cpu')

    No return but the model config in cfg will be updated in place with the right
    priorities.

    Parameters
    ----------
    cfg : DictConfig, Configuration for the experiment, will be modified in-place.
    model_type_name: str, type of the model to change, eg "representation_model"
    task: str, indicating the task of interest.
    label: str, indicating the label of interest.
    """
    # Start with default parameters.
    model = cfg[model_type_name]
    if model.get('_target_', None) == 'omics_rpz.models.EnsembleNormalizations':
        solve_model_cfg(
            cfg["prediction_model"]["model_configs"],
            model.get('model_choice'),
            task,
            label,
        )

    else:
        model_cfg = model['default']
        with open_dict(model_cfg):
            if model_cfg_task := model.get(task, None):
                # Update with task-specific parameters.
                model_cfg.update(
                    {
                        key: val
                        for key, val in model_cfg_task.items()
                        # Update only task-specific base parameters
                        # ignore label-specific configs
                        if val is not None and key not in TASK_LABEL_LIST
                    }
                )
                if label and (model_cfg_task_label := model_cfg_task.get(label, None)):
                    # Update with label-specific parameters.
                    model_cfg.update(model_cfg_task_label)

            model_cfg.update(
                # Update with command line or sweeper parameters.
                {
                    key: val
                    for key, val in model.items()
                    # Update only base parameters, ignore task-specific configs
                    if val is not None and key not in ['default'] + TASK_LIST
                }
            )
        cfg[model_type_name] = model_cfg

    if (
        model_type_name == 'representation_model'
        and model_cfg['_target_'] == 'omics_rpz.transforms.Gnn'
    ):
        try:
            # pylint: disable-next=import-outside-toplevel disable-next=unused-import
            from omics_rpz.transforms import Gnn  # noqa
        except ImportError:
            logger.error(
                "Omics_rpz was installed without GNN capabilities and user asked for"
                " GNN reprentation. Re-install with 'poetry install -E gnn'"
            )
            sys.exit(1)


def solve_ensemble_cfg(cfg: DictConfig) -> Union[DictConfig, list[DictConfig]]:
    """From a config dictionary with an ensemble field, create a list of configs that
    differ only in the ensembling parameter. E.g input a model config with ensembling
    parameter num_epochs and values [10, 100], the output will be two dict config
    objects, one with 10 epochs and the other with 100. All other parameters will be the
    same. If no ensembling parameters are specified, the function will return the
    original config.

    Parameters
    ----------
    cfg : DictConfig
        configuration of models to ensemble

    Returns
    -------
    Union[DictConfig, List[DictConfig]]
        list of configs, identical except for the ensembling parameters. If no
        ensembling parameters are specified, the original config is returned
    """
    if cfg['ensemble'] is None:
        drop_ensemble_field(cfg)
        # we need to drop the ensemble field otherwise there is an error when
        # instantiating the object
        return [cfg]
    ensemble_params = cfg['ensemble']
    configs = []
    n_ensemble = len(list(ensemble_params.values())[0])
    for i in range(n_ensemble):
        for parameter in ensemble_params.keys():
            config_i = cfg.copy()
            drop_ensemble_field(config_i)
            config_i[parameter] = ensemble_params[parameter][i]
        configs.append(config_i)
    return configs


def drop_ensemble_field(cfg: DictConfig) -> None:
    """Drops the unnecessary 'ensemble' field in a config. The config is modified in
    place.

    Parameters
    ----------
    cfg : DictConfig
        config to modify
    """
    if 'ensemble' in cfg.keys():
        OmegaConf.set_struct(cfg, False)
        cfg.pop('ensemble')
