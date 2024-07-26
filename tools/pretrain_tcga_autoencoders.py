"""Pretrain autoencoders on TCGA and finetune HPT based on reconstruction loss."""
import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import ray
import torch
from loguru import logger
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

from omics_rpz.constants import ALL_TCGA_COHORTS, OS_COHORTS
from omics_rpz.data import load_multiple_cohorts_tcga
from omics_rpz.transforms import AutoEncoder, RNASeqPreprocessor

PRETRAIN_COHORTS_OS = [x for x in ALL_TCGA_COHORTS if x not in OS_COHORTS]


def launch_fold(
    train_data: pd.DataFrame, val_data: pd.DataFrame, gene_list_path: str, params: dict
) -> float:
    """Train and evaluate an AutoEncoder for a single fold of cross-validation.

    Args:
        train_data (pd.DataFrame): Training data for the fold.
        val_data (pd.DataFrame): Validation data for the fold.
        gene_list_path (str): Path to the gene list used by the RNASeqPreprocessor.
        params (dict): Hyperparameters for the AutoEncoder.

    Returns:
        float: Validation loss for the fold.
    """

    # Preprocess dataset
    scaler = RNASeqPreprocessor(gene_list=gene_list_path)
    train_data_scaled = scaler.fit_transform(train_data)
    val_data_scaled = scaler.transform(val_data)

    logger.info("Training...")
    # Initialize and train the AutoEncoder
    autoencoder = AutoEncoder(**params)

    # Train the autoencoder and get validation loss
    val_loss = 0.0  # Replace with actual validation loss computation
    autoencoder.fit(train_data_scaled)

    # Evaluate the autoencoder on validation data and get validation loss
    logger.info("Evaluating...")
    autoencoder.evaluate(val_data_scaled)
    val_loss = autoencoder.eval_loss[0]
    return val_loss


@ray.remote(num_cpus=16, num_gpus=0.5)
def launch_cv(
    cohorts: list, random_seed: int, gene_list_path: str, params: dict
) -> float:
    """Launch cross-validation for a set of hyperparameters using Ray.

    Args:
        cohorts (list): List of cohorts to include or 'ALL_TCGA_COHORTS' string.
        random_seed (int): Random seed for reproducibility.
        gene_list_path (str): Path to the gene list used by the RNASeqPreprocessor.
        params (dict): Hyperparameters for the AutoEncoder.

    Returns:
        float: Average validation loss across all folds.
    """
    folds_losses = []
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_seed)

    # Load data
    logger.info("Loading TCGA data")
    data, _ = load_multiple_cohorts_tcga(cohorts)
    y = data.cancer_type.values

    for train_idx, val_idx in tqdm(kfold.split(X=np.zeros(len(data)), y=y)):
        train_data, val_data = data.iloc[train_idx], data.iloc[val_idx]
        loss_on_fold_i = launch_fold(train_data, val_data, gene_list_path, params)
        folds_losses.append(loss_on_fold_i)
    return np.mean(folds_losses)


def optimize_autoencoder(
    trial: optuna.Trial, cohorts: list, gene_list_path: str
) -> float:
    """Optimize the AutoEncoder model hyperparameters using Optuna.

    Args:
        trial (optuna.Trial): The Optuna trial object for hyperparameter optimization.
        cohorts (list): List of cohorts to include or 'ALL_TCGA_COHORTS' string.
        gene_list_path (str): Path to the gene list used by the RNASeqPreprocessor.

    Returns:
        float: The average validation loss achieved during cross-validation.
    """
    # Sample hyperparameters using Optuna's suggest methods
    ae_params = {}
    ae_params["repr_dim"] = trial.suggest_int('repr_dim', 16, 256)
    ae_params["hidden_n_layers"] = trial.suggest_categorical(
        'hidden_n_layers', [0, 1, 2]
    )
    ae_params["hidden_n_units_first"] = trial.suggest_int(
        'hidden_n_units_first', 256, 1024
    )
    ae_params["learning_rate"] = trial.suggest_float('learning_rate', 0.000005, 0.0005)
    ae_params["dropout"] = trial.suggest_float('dropout', 0.0, 0.2)
    ae_params["hidden_decrease_rate"] = trial.suggest_categorical(
        'hidden_decrease_rate', [0.5, 1.0]
    )

    # Other hyperparameters
    ae_params["early_stopping_use"] = True
    ae_params["max_num_epochs"] = 1000
    ae_params["early_stopping_patience"] = 20
    ae_params["early_stopping_delta"] = 0.00001
    ae_params["batch_size"] = 1024
    ae_params["device"] = 'cuda' if torch.cuda.is_available() else 'cpu'

    val_losses = []

    # Perform repeated CV
    repeated_cvs = [
        launch_cv.remote(cohorts, i, gene_list_path, ae_params) for i in range(5)
    ]

    val_losses = ray.get(repeated_cvs)
    avg_val_loss = np.mean(val_losses)

    return avg_val_loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="AutoEncoder Hyperparameter Optimization Script"
    )
    parser.add_argument(
        "--gene_list",
        type=str,
        required=True,
        help="Path to the gene list for RNASeqPreprocessor",
    )
    parser.add_argument(
        "--output_path", type=str, required=True, help="Path to save the output"
    )
    parser.add_argument(
        "--cohorts",
        type=str,
        required=False,
        default="ALL_TCGA_COHORTS",
        help="Path to the gene list for RNASeqPreprocessor",
    )
    args = parser.parse_args()

    gene_list = args.gene_list
    output_path = Path(args.output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    if (cohorts := args.cohorts) == "PRETRAIN_COHORTS_OS":
        cohorts = PRETRAIN_COHORTS_OS

    # Use Ray for //
    ray.init(num_cpus=80, num_gpus=4, ignore_reinit_error=True)

    study = optuna.create_study(
        direction='minimize', sampler=optuna.samplers.TPESampler()
    )
    logger.info("Optimizing RPZ model")
    study.optimize(
        lambda trial: optimize_autoencoder(trial, cohorts, gene_list), n_trials=50
    )

    print("Number of finished trials: ", len(study.trials))
    print("Best trial:")
    trial = study.best_trial

    print("Value: ", trial.value)
    print("Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

    with open(output_path / "best_ae_params.json", "w", encoding="utf-8") as outfile:
        json.dump(trial.params, outfile)

    # Scale the whole dataset using the gene list and RNAseqPreprocessor
    data, _ = load_multiple_cohorts_tcga(cohorts)
    scaler = RNASeqPreprocessor(gene_list=args.gene_list)
    data_scaled = scaler.fit_transform(data)

    # Initialize and train the best auto-encoder on the whole dataset
    best_autoencoder = AutoEncoder(
        repr_dim=trial.params['repr_dim'],
        hidden_n_layers=trial.params['hidden_n_layers'],
        hidden_n_units_first=trial.params['hidden_n_units_first'],
        hidden_decrease_rate=trial.params['hidden_decrease_rate'],
        dropout=trial.params['dropout'],
        early_stopping_use=True,
        max_num_epochs=1000,
        batch_size=1024,
        learning_rate=trial.params['learning_rate'],
        device='cuda' if torch.cuda.is_available() else 'cpu',
    )

    best_autoencoder.fit(data_scaled)

    # Save pickle of preprocessor
    with open(output_path / "scaler.pkl", "wb") as scaler_file:
        pickle.dump(scaler, scaler_file)

    # Save pickle of autoencoder
    with open(output_path / "ae.pkl", "wb") as model_file:
        pickle.dump(best_autoencoder, model_file)

    # Save Optuna study
    with open(output_path / "optuna_study.pkl", "wb") as study_file:
        pickle.dump(study, study_file)
