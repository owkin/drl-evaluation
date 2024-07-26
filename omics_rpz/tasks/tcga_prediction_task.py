"""Class for TCGA predictions (supervised) tasks."""
import copy
import pickle
from pathlib import Path
from typing import Optional, Union

import mlflow
import numpy as np
import pandas as pd
from hydra.utils import instantiate
from loguru import logger

from omics_rpz.data import create_fold_iterator, create_group_envs, test_split
from omics_rpz.transforms import GaussianNoise, rpz_transform_wrapper
from omics_rpz.utils import save_predictions


class TCGAPredictionTask:
    """TCGA prediction task class.

    Parameters
    ----------
    data :
        Config for the dataset.
    preprocessor :
        Config for the preprocessor.
    rpz_model :
        Config for the representation model.
    pred_model :
        Config for the prediction model for the task.
    metric :
        Config for the metric used. The metric differs depending on the task.
    logging_path : str
        Path where to save the logs.
    label : Union[str, list]
        label for the target that is predicted. Can be list for multi-label regression.
    pretraining : bool, optional
        To activate pretraining, by default False
    pretrained_ae_path : str, optional
        Path to load pretrained RPZ method and associated scaler. Will overide
        pretraining parameter above if not None.
    finetuning_rpz: bool, optional
        To activate finetuning of rpz model when pretraining is True, by default False
    noise_training : bool, optional
        Use of noise in training, by default False
    noise_pretraining : bool, optional
        Use noise in pretraining, by default False
    noise_level_training : float, optional
        Standard deviation of noise in training, by default 0.0
    noise_level_pretraining : float, optional
        Standard deviation of noise in pretraining, by default 0.0
    n_augment_training: int, optional
        Number of times the training data is duplicated and concatenated to the original
         data.
    n_augment_pretraining: int, optional
        Number of times the training data is duplicated and concatenated to the original
         data.
    normalizations_augment: list, optional
        Normalization types from: RAW, RPKM, TPM, NORM. The data normalized this way
         will be used as augmentation data for training
    train_test_no_cv : bool, optional
        When True, there is no cross-val but just a train-test, by default False
    test_split_random_seed : int
        Random seed for the split between train/validation and test sets
    n_repeats_cv : int, optional
        number of repeats of the cross-validation, by default 3
    n_random_seed_cv_start_at: int, optional
        random seed to start the repeats of the cross-validation, by default 0
    n_splits : int, optional
        splits for the cross-validation, by default 5
    bootstrapping_test : int, optional
        number of times bootstrapping is performed for metric calculation on the test
        set when there is no cross-validation, by default 1000
    finetuning : bool, optional
        If true then finetuning is used for the training of the pred and representation
        models.

    Raises
    ------
    AssertionError
        if pretraining activated and no pretrain dataset given.

    AssertionError
        if finetuning_rpz activated and no pretraining activated

    AssertionError
        if finetuning activated and the prediction model is not MLP.

    AssertionError
        if noise augmentation is activated and either of the noise parameters is zero.


    Attributes
    ----------
    preprocessor
        Preprocessing model. Defaults to None. Built from config in run method.
    rpz_model
        Representation model. Defaults to None. Built from config in run method.
    pred_model
        Prediction model. Defaults to None. Built from config in run method.
    rpz_model_pretrained
        Representation model. Defaults to None. Built in case of finetuning_rpz to not
        introduce leak in the data in cross-validation.
    metric
        Metric function built from the metric config.
    rnaseq_variables
        RNAseq variables that are kept for the prediction task.
    ensembling_normalization
        Whether the prediction model is an ensemble model on different normalization.
        One AE is fitted for all the normalization concatenated together but then one
        prediction model is fitted for each transformed representation. If not set there
        will be just one pred model fitted on all the different normalization.


    Methods
    -------
    pretrain_rpz_model
        activates pretraining of the model when the pretraining parameter is True.
    train
        to train the representation and prediction model.
    run_cv
        Runs the cross-validation.
    run_no_cv
        Runs train and test only without cross-validation.
    run
        Calls the other run functions depending on whether there is a cross-validation
        or not.
    predict_output_and_compute_eval_metric
        Compute predictions from models and then evaluation metrics.
    """

    def __init__(
        self,
        data,
        preprocessor,
        rpz_model,
        pred_model,
        metric,
        logging_path: str,
        label: Union[str, list[str]],
        save_predictions: bool = False,
        pretraining: bool = False,
        pretrained_ae_path: str = None,
        finetuning_rpz: bool = False,
        noise_training: bool = False,
        noise_pretraining: bool = False,
        noise_level_training: float = 0.0,
        noise_level_pretraining: float = 0.0,
        n_augment_training: int = 0,
        n_augment_pretraining: int = 0,
        normalizations_augment: list = None,
        train_test_no_cv: bool = False,
        split_ratio: float = 0.2,
        test_split_random_seed: int = 0,
        n_repeats_cv: int = 3,
        n_random_seed_cv_start_at: int = 0,
        n_splits: int = 5,
        bootstrapping_test: int = 1000,
        finetuning: bool = False,
    ):
        if pretraining and "pretrain_dataset" not in data:
            raise AssertionError
        if finetuning_rpz and not pretraining:
            raise AssertionError

        if finetuning and pred_model.get('_target_').split('.')[-1] != 'MLPPrediction':
            raise AssertionError
        if noise_training and noise_level_training == 0.0:
            raise AssertionError
        if noise_pretraining and noise_level_pretraining == 0.0:
            raise AssertionError

        self.data_cfg = data
        self.preprocessor_cfg = preprocessor
        self.rpz_model_cfg = rpz_model
        self.pred_model_cfg = pred_model
        self.metric_cfg = metric
        self.labels = [label] if isinstance(label, str) else label

        self.save_predictions = save_predictions
        self.logging_path = logging_path
        self.pretraining = True if pretrained_ae_path is not None else pretraining
        self.pretrained_ae_path = pretrained_ae_path
        self.finetuning_rpz = finetuning_rpz
        self.noise_training = noise_training
        self.noise_pretraining = noise_pretraining
        self.noise_level_pretraining = noise_level_pretraining
        self.noise_level_training = noise_level_training
        self.n_augment_training = n_augment_training
        self.n_augment_pretraining = n_augment_pretraining
        self.normalizations_augment = normalizations_augment
        self.train_test_no_cv = train_test_no_cv
        self.split_ratio = split_ratio
        self.test_split_random_seed = test_split_random_seed
        self.n_repeats_cv = n_repeats_cv
        self.n_random_seed_cv_start_at = n_random_seed_cv_start_at
        self.n_splits = n_splits
        self.bootstrapping = bootstrapping_test
        self.finetuning = finetuning
        self.ensembling_normalization = (
            self.pred_model_cfg['_target_'] == 'omics_rpz.models.EnsembleNormalizations'
        )
        if finetuning:
            self.rpz_model_finetuning = None

        self.preprocessor = None
        self.rpz_model = None
        self.pred_model = None
        self.prefitted_scaler = None
        self.rpz_model_pretrained = None
        self.rnaseq_variables = []
        self.augment_norm = None

        self.metric = instantiate(self.metric_cfg)

    def pretrain_rpz_model(self, df_train_val) -> None:
        """Function to handle pretraining of the rpz_model if needed.

        Intersect genes between pretraining and training datasets. Fit preprocessor with
        training dataset to choose the most variant genes based on training dataset.
        Fit_transform of preprocessor and then rpz_model on pretraining dataset.

        Args:
            df_train_val (pd.DataFrame): gene expression data for training/validation.
        """

        if self.pretrained_ae_path is not None:
            logger.info("Loading pretrained autoencoder on TCGA...")
            pretrained_output_pt = Path(self.pretrained_ae_path)
            with open(pretrained_output_pt / "scaler.pkl", "rb") as pretrained_rpz_pkl:
                preprocessor = pickle.load(pretrained_rpz_pkl)
            with open(pretrained_output_pt / "ae.pkl", "rb") as pretrained_rpz_pkl:
                pretrained_rpz_model = pickle.load(pretrained_rpz_pkl)
            self.rpz_model = pretrained_rpz_model
            self.preprocessor = preprocessor

            if self.finetuning_rpz:
                self.prefitted_scaler = copy.deepcopy(self.preprocessor)
                self.rpz_model_pretrained = copy.deepcopy(self.rpz_model)

            return self

        # Load pre-training data
        logger.info("Loading pre-training data...")

        load_pretrain_data = instantiate(self.data_cfg["pretrain_dataset"])
        df_pretrain, pretrain_rnaseq_variables = load_pretrain_data()

        # Intersect genes columns.
        self.rnaseq_variables = sorted(
            list(
                set(self.rnaseq_variables).intersection(set(pretrain_rnaseq_variables))
            )
        )

        # Log pre-training data info
        logger.info("Pre-training data info")
        logger.info(f"Number of samples: {len(df_pretrain)}")
        logger.info(f"Number of genes: {len(self.rnaseq_variables)}")
        logger.info("\n")

        # Pre-training preprocessing
        logger.info("Pre-training Preprocessing")
        X_pretrain = df_pretrain[self.rnaseq_variables]
        # We create the gene filters based on the training dataset.
        self.preprocessor = instantiate(self.preprocessor_cfg)
        self.preprocessor.fit(df_train_val[self.rnaseq_variables])
        # Now the gene list is stored in the self.preprocessor to be used for
        # the cohort.
        X_pretrain_processed = self.preprocessor.fit_transform(X_pretrain)

        # Noise Pretraining added AFTER pre-processing
        if self.noise_pretraining:
            # We add gaussian noise once all variables have been normalized.
            noiser = GaussianNoise(
                variables_to_perturbate=X_pretrain_processed.columns,
                gaussian_std=self.noise_level_pretraining,
                number_of_duplication=self.n_augment_pretraining,
            )
            X_pretrain_processed = noiser.transform(X_pretrain_processed)

        # Instantiate and fit rpz model
        logger.info("Pre-training RPZ model")
        self.rpz_model = instantiate(self.rpz_model_cfg)
        self.rpz_model.fit(X_pretrain_processed, df_labels=df_pretrain)

        if self.finetuning_rpz:
            self.rpz_model_pretrained = copy.deepcopy(self.rpz_model)

        return self

    def train(
        self,
        df_training: pd.DataFrame,
        metrics_suffix: Optional[str] = None,
    ):
        """Handles training pipeline with a train dataset. Allows the use of the
        pipeline with different inputs (if crossval or test).

        Args:
            df_training (pd.DataFrame): training set
            metrics_suffix (Optional[str], optional): Optional name for logging.
            Defaults to None.
        """
        # Create X and y for train and validation from the dataframes
        # Need to filter X_train on self.rnaseq_variables in the event of pretraining.
        X_train = df_training[self.rnaseq_variables]
        y_train = df_training[self.labels]
        # preprocessing
        logger.info("Pre-processing RNASeq data")
        if not self.pretraining:
            # Need to instantiate the preprocessor if not done in pretraining (cannot be
            # done elsewhere as in case of CV it needs a new preprocessor at every fold)
            self.preprocessor = instantiate(self.preprocessor_cfg)
            X_prep_train = self.preprocessor.fit_transform(X_train)
        elif self.pretraining and not self.finetuning_rpz:
            X_prep_train = self.preprocessor.transform(X_train)
        elif self.pretraining and self.finetuning_rpz:
            self.preprocessor = copy.deepcopy(self.prefitted_scaler)
            X_prep_train = self.preprocessor.fit_transform(X_train)

        if self.normalizations_augment is not None:
            X_prep_train, normalizations_col = self.augment_norm.concatenate_augmented(
                X_prep_train, fit_preprocessor=True
            )

        # Add noise AFTER pre-processing
        if self.noise_training:
            logger.info("Adding noise to RNAseq data")
            # We add gaussian noise once all variables have been normalized.
            noiser = GaussianNoise(
                variables_to_perturbate=X_prep_train.columns,
                gaussian_std=self.noise_level_training,
                number_of_duplication=self.n_augment_training,
            )
            X_prep_train, y_train = noiser.transform_with_labels(
                # Ensure input for labels is a dataframe
                X_prep_train,
                df_training[self.labels],
            )

        # Representation model
        if not self.pretraining:
            logger.info("Training RPZ method")
            # if there is no pretraining the representation model needs to be
            # instantiated and fitted on the training data.
            self.rpz_model = instantiate(self.rpz_model_cfg)
            self.rpz_model.fit(
                X=X_prep_train,
                metrics_suffix=metrics_suffix,
                df_labels=df_training.loc[X_prep_train.index],
            )
            # the .loc enforces that X and df_labels have the same index, useful for
            # data augmentation cases

        if self.finetuning_rpz:
            # If pretrained, re-train the rpz model on the train dataset
            logger.info("Finetuning RPZ method")

            self.rpz_model = copy.deepcopy(self.rpz_model_pretrained)

            if self.rpz_model_cfg["early_stopping_use"]:
                # then early_stopping_use has been changed to False after pretraining
                # and num_epochs to number of epochs found by early stopping pretraining
                # so reset them in this case
                self.rpz_model.early_stopping_use = True
                self.rpz_model.num_epochs = self.rpz_model_cfg["max_num_epochs"]
                self.rpz_model.early_stopping_delta = self.rpz_model_cfg[
                    "early_stopping_delta"
                ]
                self.rpz_model.early_stopping_patience = self.rpz_model_cfg[
                    "early_stopping_patience"
                ]

            self.rpz_model.fit(
                X=X_prep_train,
                metrics_suffix=metrics_suffix,
                df_labels=df_training.loc[X_prep_train.index],
                finetuning_rpz=True,
            )
            # the .loc enforces that X and df_labels have the same index, useful for
            # data augmentation cases

        # prediction model
        if self.finetuning:
            # with finetuning, the construction of the model is different as the
            # prediction model needs the representation model to be initialised.
            self.pred_model_cfg["_partial_"] = True
            pred_model_constructor = instantiate(
                self.pred_model_cfg, _recursive_=(not self.ensembling_normalization)
            )
            # For the ensemble model we do not want individual models to be instantiated
            # at this time. For all the other models we keep the default behavior
            self.pred_model = pred_model_constructor(
                auto_encoder=copy.deepcopy(self.rpz_model)
            )
            df_emb_train = X_prep_train.to_numpy()
        else:
            self.pred_model = instantiate(
                self.pred_model_cfg, _recursive_=(not self.ensembling_normalization)
            )
            # For the ensemble model we do not want individual models to be instantiated
            # at this time. For all the other models we keep the default behavior
            df_emb_train = rpz_transform_wrapper(self.rpz_model, X_prep_train)
        # Note: the type of X_emb_train is different if finetuning (numpy) or not
        # (dataframe)
        # Here add back the normalization columns saved previously.
        if self.normalizations_augment is not None:
            df_emb_train["normalization"] = normalizations_col
            y_train = df_training[self.labels]
        self.pred_model.fit(df_emb_train, y_train)

    def predict_output_and_compute_eval_metric(
        self, df_validation: pd.DataFrame
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """Returns the prediction, ground truth label and evaluation metric of the
        trained model on a dataset.

        Parameters
        ----------
        df_validation : pd.DataFrame
            Dataset fed to model for generating the metric.

        Returns
        -------
        tuple[np.ndarray, np.ndarray,float]
            Prediction, ground truth label and metric.
        """
        # Need to filter X_valid on self.rnaseq_variables in the event of pretraining.
        X_valid = df_validation[self.rnaseq_variables]
        y_valid = df_validation[self.labels]

        # Preprocessing
        X_prep_valid = self.preprocessor.transform(X_valid)
        if self.augment_norm and self.ensembling_normalization:
            X_prep_valid, normalization_col = self.augment_norm.concatenate_augmented(
                X_prep_valid,
                fit_preprocessor=False,
            )
        # RPZ model
        if self.finetuning:
            X_emb_valid = X_prep_valid.to_numpy()
        else:
            X_emb_valid = rpz_transform_wrapper(self.rpz_model, X_prep_valid)

        # self.normalizations_augment and self.ensembling_normalization
        if self.normalizations_augment is not None:
            y_valid = df_validation[self.labels]
            X_emb_valid['normalization'] = normalization_col

        # Predict
        y_preds_valid = self.pred_model.predict(X_emb_valid)

        # Metric
        return y_preds_valid, y_valid, self.metric(y_valid, y_preds_valid)

    def run_cv(self, df_train_val: pd.DataFrame) -> tuple[dict, dict, pd.DataFrame]:
        """Given one dataframe, runs a cross-validation for the task n_repeat times.

        Args:
            df_train_val (pd.DataFrame): Dataframe used for the cross-validation.

        Raises:
            ValueError: Not possible to have more splits than groups

        Returns:
            tuple[dict, dict, pd.DataFrame]
            prediction and label dictionaries with one key per fold, and dataframe
            that contains the metrics on the validation sets.
        """
        # pylint: disable=duplicate-code
        # Cross-validation
        logger.info("Cross-validation loop")
        logger.info("\n")

        r_indices, s_indices = [], []
        cv_train_metric, cv_val_metric = [], []
        nb_samples_train, nb_samples_val = [], []
        labels_val, predictions_val = {}, {}

        for random_state in range(
            self.n_random_seed_cv_start_at,
            self.n_repeats_cv + self.n_random_seed_cv_start_at,
        ):
            # Create fold iterator
            # Stratify on environment for OOD
            if self.data_cfg.group_covariate:
                groups = create_group_envs(df_train_val, self.data_cfg.group_covariate)
                if self.n_splits > len(np.unique(groups)):
                    raise ValueError("Not possible to have more splits than groups")
            # Group by patient if no domain adaptation split
            else:
                groups = df_train_val.patient_id.values
            stratify = (len(self.labels) == 1) and (self.labels[0] in {"OS", "PFS"})
            splits_gen = create_fold_iterator(
                # why did we use np.arange(len(df_train_val)) before?
                df_all=df_train_val,
                groups=groups,
                # Stratify on covariate if not null - censoring will be used
                stratify=df_train_val[self.labels].values if stratify else None,
                random_seed=random_state,
            )

            for split, (train_indices, val_indices) in enumerate(splits_gen):
                # groups argument = ensures non-overlapping group between splits
                # (so not the same patient in two splits)
                # y= (df.label > 0) stratifies the splits but this will only
                # work for binary labels and stratify censored
                # patients for survival and won't "work" for regression tasks
                # (except on the <0 versus >0)

                # Split data
                df_train = df_train_val.iloc[train_indices]
                df_valid = df_train_val.iloc[val_indices]

                # to log repeat/split information with MLflow
                metrics_suffix = f"rep{random_state} split{split}"

                self.train(df_train, metrics_suffix)
                (
                    (_, _, train_metric),
                    (y_pred_val, y_val, valid_metric),
                ) = self.predict_output_and_compute_eval_metric(
                    df_train
                ), self.predict_output_and_compute_eval_metric(
                    df_valid
                )

                # Log split info.
                logger.info(f"REPEAT {random_state} SPLIT {split}")
                logger.info(f"Train set: {len(train_indices)}")
                logger.info(f"Valid set: {len(val_indices)}")
                logger.info(f"Training metric: {train_metric:.3f}")
                logger.info(f"Validation metric: {valid_metric:.3f}")
                logger.info("\n")

                # Save split info
                r_indices.append(random_state)
                s_indices.append(split)

                cv_train_metric.append(train_metric)
                cv_val_metric.append(valid_metric)

                predictions_val[split], labels_val[split] = y_pred_val, y_val

                nb_samples_train.append(len(train_indices))
                nb_samples_val.append(len(val_indices))

        # Compute cross-validation metrics.
        mean_cv_train_metric, std_cv_train_metric = np.mean(cv_train_metric), np.std(
            cv_train_metric
        )

        mean_cv_val_metric, std_cv_val_metric = np.mean(cv_val_metric), np.std(
            cv_val_metric
        )

        # Log cross-validation metrics
        logger.info("Cross-validation metrics")
        logger.info(
            f"Training metric: {mean_cv_train_metric:.3f} ({std_cv_train_metric:.3f})"
        )
        logger.info(
            f"Validation metric: {mean_cv_val_metric:.3f} ({std_cv_val_metric:.3f})"
        )
        logger.info("\n")

        # MLFlow: Log cross-val metrics.
        mlflow.log_metric("mean_cv_train_metric", mean_cv_train_metric)
        mlflow.log_metric("std_cv_train_metric", std_cv_train_metric)

        mlflow.log_metric("mean_cv_val_metric", mean_cv_val_metric)
        mlflow.log_metric("std_cv_val_metric", std_cv_val_metric)

        results = pd.DataFrame(
            data={
                "REPEAT": r_indices,
                "SPLIT": s_indices,
                "cv_train_metric": cv_train_metric,
                "cv_val_metric": cv_val_metric,
                "nb_samples_train": nb_samples_train,
                "nb_samples_val": nb_samples_val,
            }
        )

        return predictions_val, labels_val, results

    def run_no_cv(
        self, df_train_val: pd.DataFrame, df_test: pd.DataFrame
    ) -> tuple[dict, dict, pd.DataFrame]:
        """Trains model and test it in the case that there is no cross-validation.

        Parameters
        ----------
        df_train_val : pd.DataFrame
            Training set
        df_test : pd.DataFrame
            Test set

        Returns
        -------
        tuple[dict, dict, pd.DataFrame]
            prediction and label dictionaries with only one key '0' mapping to test set
            prediction and labels, and dataframe that contains the metrics on the test
            set.
        """
        logger.info("Train / Test train (no cross-validation)")
        logger.info("Training prediction model (along with encoder if fine-tuning)")
        logger.info("\n")

        # Train
        self.train(df_train_val)
        _, _, train_metric = self.predict_output_and_compute_eval_metric(df_train_val)

        # Log split info
        logger.info("Final Test")
        logger.info(f"Train set patients: {len(df_train_val)}")
        logger.info(f"Test set patients: {len(df_test)}")
        logger.info(f"Training metric: {train_metric:.3f}")

        # Get predictions on test set for bootstrapping
        X_test = df_test[self.rnaseq_variables]
        y_test = df_test[self.labels]
        X_prep_test = self.preprocessor.transform(X_test)

        if self.normalizations_augment is not None:
            X_prep_test, normalizations_col = self.augment_norm.concatenate_augmented(
                X_prep_test, fit_preprocessor=False
            )

        if self.finetuning:
            X_emb_test = X_prep_test.to_numpy()
        else:
            X_emb_test = rpz_transform_wrapper(self.rpz_model, X_prep_test)
        # Here add back the normalization columns saved previously.
        if self.normalizations_augment is not None:
            X_emb_test["normalization"] = normalizations_col
            y_test = df_test[self.labels]

        y_preds_test = self.pred_model.predict(X_emb_test)

        # Save experiment results
        if self.save_predictions:
            predictions_path = (
                Path(self.logging_path).parent / "test_predictions.parquet"
            )
            save_predictions(y_true=y_test, y_pred=y_preds_test, path=predictions_path)

        # Bootstrapping on test set
        scores = []
        sampling_rate = 1
        logger.info(
            f"Computing scores n={self.bootstrapping} times "
            f"with {sampling_rate} sampling rate"
        )
        for _ in range(self.bootstrapping):
            random_indices = np.random.choice(
                len(y_preds_test),
                size=int(len(y_preds_test) * sampling_rate),
                replace=True,
            )
            y_subsample_pred, y_subsample_test = (
                y_preds_test[random_indices],
                y_test.values[random_indices],
            )
            scores.append(self.metric(y_subsample_test, y_subsample_pred))

        mean_scores = np.mean(scores)
        ci_95_low, ci_95_high = np.quantile(scores, 0.025), np.quantile(scores, 0.975)

        # MLFlow: Log metrics
        mlflow.log_metric("train_metric", train_metric)
        mlflow.log_metric("test_metric", mean_scores)
        mlflow.log_metric("test_metric_95_CI_low", ci_95_low)
        mlflow.log_metric("test_metric_95_CI_high", ci_95_high)

        # Log test metric info
        logger.info(f"Test metric: {mean_scores:.3f}")
        logger.info(f"Test metric 95% CI: [{ci_95_low:.3f}, {ci_95_high:.3f}]")
        logger.info("\n")

        res = pd.DataFrame(
            data={
                "REPEAT": np.arange(self.bootstrapping),
                "test_metric": scores,
                "nb_patients_val": len(random_indices),
            }
        )

        return (
            {0: y_preds_test},
            {0: y_test},
            res,
        )  # dictionaries to be consistent with run cv

    def run(self) -> tuple[dict, dict, pd.DataFrame]:
        """Function to run the training and evaluation of tcga prediction tasks with
        pre-training and/or fine-tuning if desired.

        Returns
        -------
        tuple[dict, dict, pd.DataFrame]
            prediction and label dictionaries with one key per fold, and dataframe
            that contains the metrics of the run.
        """
        # Load data
        logger.info("Loading data...")
        load_data = instantiate(self.data_cfg["dataset"])
        df_data, rnaseq_variables = load_data()

        # Split all the data in train and test
        df_data = df_data.dropna(subset=["patient_id"] + self.labels)

        # For stratification.
        groups = df_data.patient_id.values
        if self.data_cfg.group_covariate:
            groups = create_group_envs(df_data, self.data_cfg.group_covariate)

            logger.info(
                "Stratifying CV folds on                 covariate:"
                f" {self.data_cfg.group_covariate}"
            )

        # we stratify on censorship only for OS or PFS tasks
        stratify = (len(self.labels) == 1) and (self.labels[0] in {"OS", "PFS"})
        df_train_val, df_test = test_split(
            df_all=df_data,
            groups=groups,
            stratify=df_data[self.labels].values if stratify else None,
            split_ratio=self.split_ratio,
            random_seed=self.test_split_random_seed,
        )
        self.rnaseq_variables = sorted(rnaseq_variables)

        # Log data info
        logger.info("Data info")
        logger.info(f"Number of barcodes: {df_data.index.nunique()}")
        logger.info(f"Number of samples: {df_data.sample_id.nunique()}")
        logger.info(f"Number of patients: {df_data.patient_id.nunique()}")
        for label_name in self.labels:
            label_count = df_data[label_name].nunique()
            logger.info(f"Number of labels {label_name}: {label_count}")
        logger.info(f"Number of genes: {len(rnaseq_variables)}")
        logger.info("\n")

        # Pretraining of preprocessor and RPZ model
        if self.pretraining:
            self.pretrain_rpz_model(df_train_val)

        if self.train_test_no_cv:
            return self.run_no_cv(df_train_val, df_test)
        return self.run_cv(df_train_val)
