<h1 align="center" style="font-size:75px;">Robust evaluation of DRL Methods</h1>

<p align="center">GitHub repository hosting the code used in the paper Robust Evaluation of Deep Learning-based Representation Methods for Survival and Gene Essentiality Prediction on Bulk RNA-seq Data.</p>


## Installation

Use `conda` and `poetry` to easily set up the repository :

```bash
conda create -n omics_rpz_env python=3.9
conda activate omics_rpz_env
pip install "poetry==1.4.0"
make install-all # or make install-M1 for M1 Macs
pre-commit install
```

## Download data 

The data used in our study is available for download on this bucket. 
Once downloaded, make sure to update the `PREFIX` variable in `omics_rpz/constants.py` to point to the location of downloaded folder `benchmark_data`.

## Launch an experiment

Several tasks are available on the repository. The `benchmark` script can be run to perform CV or in a train/test settings.

```bash
python ./tools/benchmark.py task=survival_prediction_tcga_task prediction_model=cox_model representation_model=pca
python ./tools/benchmark.py task=gene_essentiality prediction_model=linear_regression representation_model=pca
```

## Hyperparameter settings

Hyperparameters per model per task are stored in the config files `/conf/representation_model/` as nested dicts :

```
default:
  _target_: omics_rpz.transforms.AutoEncoder
  repr_dim: 128
  hidden: [512, 256]
  dropout: [0.5, 0.5]
  bias: True
  num_epochs: 10
  batch_size: 32
  learning_rate: 0.0005
  split_data: False
  device: cpu

TCGAPredictionTask:
  num_epochs: 50
  OS:
    repr_dim: 64
    dropout: [0, 0]
    num_epochs: 100
    batch_size: 128
```

By default, the `default` configuration is used and overriden with task-specific parameters if the task class name (given by the `_target_` field in `conf/task/` files) is an entry in the conf file. For `TCGAPredictionTask`, task-specific parameters are overriden by label-specific parameters as shown above.
Values given with command line arguments (e.g. `python ./tools/benchmark.py representation_model=auto_encoder representation_model.hidden=[1024,512]`) override both default and task-specific parameters.

## Hyperparameter Tuning with Hydra-Optuna Plugin

To perform hyperparameter-tuning with Optuna on TCGA prediction tasks, modify the `config.yaml` file with the tuning options, and add the option `--multirun` in the command line.

### Example

One example on purity prediction task:

```bash
python ./tools/benchmark.py task=purity_prediction_tcga_task prediction_model=linear_regression representation_model=pca --multirun
```

You can also specify the hyperparameter range or set in the command line. For example:

```bash
python ./tools/benchmark.py task=purity_prediction_tcga_task prediction_model=linear_regression representation_model=pca 'representation_model.repr_dim=range(16, 511)' --multirun
```

### Good to know

- Because of a hydra code choice (detailed [here](https://github.com/facebookresearch/hydra/issues/2003)), to change samplers, 2 lines must be changes in the config file: `defaults / override /hydra/sweeper/sampler` and `hydra / sweeper / sampler / _target_`. See config.yaml for more details.
- To easily filter all trials of a given study in MLflow, we may set study_name=<descriptive_string>, either in the config.yaml file or via command line. Then, in MLflow, we filter using params.study_name="descriptive_string".

More info on the hydra-optuna plugin [here](https://hydra.cc/docs/plugins/optuna_sweeper/)

## Repeated holdout pipeline

The `repeated_holdout.py` script executes `benchmark.py` as many times as necessary to finish a repeated holdout experiment.

This script performs `num_repeated_holdout` (10 is the current default value from `config.yaml`) runs of:

1. Randomly split out train/val and test sets with a different `test_split_random_seed` (from `config.yaml`).
2. Perform hyperparameter search with cross-validation to find the best hyperparameters (executing our traditional `benchmark.py`, with `--multirun` and `train_test_no_cv=False`).
3. Use the best hyperparameters to measure performance on the test set (using our traditional `benchmark.py`, with `train_test_no_cv=True`).

These runs generate `num_repeated_holdout` test-set performance metrics. We report mean (SD) and 95% empirical confidence intervals.

Logs and .csv results from each experiment are kept in folders with the following format: `outputs/repeated_holdout_{study_name}_{start_time}_{k}`, where k is the run index (from 0 to `num_repeated_holdout` - 1). Final performance metrics are stored in `outputs/repeated_holdout_{study_name}_{start_time}_test_metrics`.

Example of how to run this pipeline (here only the HP `hidden` is searched over a space of two choices):

```bash
python ./tools/repeated_holdout.py task=survival_prediction_tcga_task prediction_model=mlp_prediction representation_model=auto_encoder +"representation_model.hidden=choice([1024], [512])" study_name="survival_repeated_holdout"
```

You can use `ray` to // the run of experiments by seeds by specifying `use_ray=True`. Make sure to specify the following arguments to match your machine configuration:
- `num_cpus_for_ray`: number of CPUs that you want to give access to Ray globally (summed over all repetitions). Try to leave room for buffer jobs by specifying around 12 CPUs less than that you have on your machine.
- `num_gpus_for_ray`: number of GPUs that you want to give access to Ray globally (summed over all repetitions).
- `cpu_per_repeat`: number of CPUs per // thread
- `gpu_per_repeat`: number of GPUs per // thread

By not specifying correclty these arguments, `ray` may struggle to launch in //. Make sure what you are requesting is consistent (no more CPU than what you have, no GPU if you don't have GPU or CUDA installed, NB_REPEATS*`cpu_per_repeat` < `num_cpus_for_ray` < full number of CPUS)


## Dashboards

### Track results with MLFlow

To track your results with MLFlow, run:

```bash
mlflow ui --backend-store-uri ./outputs/mlflow/ --port 8889
```

### Ray Dashboard

Ray automatically starts a dashboard when `ray.init()` is called if the dependencies required for the dashboard are installed (which is the case in the latest lock file).
The dashboard by default is created on the port 8265.

## Adding Libraries to the environment

You can add libraries to the dependencies of this project by using `poetry`

```bash
poetry add optuna
```

# Acknowledgements

The results shown here are in whole or part based upon data generated by the TCGA Research Network: https://www.cancer.gov/tcga. We thank Oussama Tchita, Omar Darwiche Domingues and Thomas Chaigneau for their valuable contributions and comments to strengthen our pipeline and coding best practices. We thank Gilles Wainrib for initial ideas and discussions, Nicolas Loiseau for his advice and statistical expertise, Floriane Montanari, Benoît Schmauch, Gilles Wainrib and Jean-Philippe Vert for their detailed proofreading and insightful comments.
