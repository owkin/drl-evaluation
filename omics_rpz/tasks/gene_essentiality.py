"""Class for the gene essentiality task."""

import copy
import pickle
import time
from pathlib import Path
from typing import Optional

import mlflow
import numpy as np
import pandas as pd
from hydra.utils import instantiate
from loguru import logger

from omics_rpz.data import (
    create_fold_iterator,
    load_multiple_cohorts_gtex,
    load_multiple_cohorts_tcga,
    test_split,
)
from omics_rpz.transforms import (
    GaussianNoise,
    rpz_fit_transform_wrapper,
    rpz_transform_wrapper,
)
from omics_rpz.utils import save_predictions


class GeneEssentiality:
    """Class for the task of predicting gene essentiality like in the DeepDEP article:
    https://pubmed.ncbi.nlm.nih.gov/34417181/

    Parameters
    ----------

    data :
        Config for the dataset. Specified in the task config file, only the pre-train
        dataset is input, the loading functions for the other datasets are methods of
        this class for now.
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
    label : str
        label for the target that is predicted.
    pretraining : bool, optional
        To activate pretraining, by default False. Specified in the task config file,
        the CCLE rpz model is pretrained on the dataset specified above (if this flag is
        True) before being used in the prediction model training phase (the rpz model is
        then frozen or not according to the flag below).
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
        Specified in the task config file. If the flag is True, the rpz model is
        trained at the same time as the pred model in the pred model training phase
        (the rpz model has been trained in an unsupervised way beforehand). If the flag
        is False, the CCLE rep is frozen and fed as input to the pred model for
        training.
    noise_training : bool, optional
        Use of noise in training, by default False
    noise_pretraining : bool, optional
        Use noise in pretraining, by default False

    Raises
    ------
    AssertionError
        if pretraining activated and no pretrain dataset given.
    AssertionError
        if finetuning_rpz activated and no pretraining activated.
    AssertionError
        if pretraining activated and representation model is the multi-head AE
    AssertionError
        if finetuning activated and the prediction model is not MLP.

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
    rnaseq_variables, List[str]:
        RNAseq variables that are kept for the prediction task.
    is_hierarchical: bool
        Boolean indicating whether the prediction model used is a hierarchical model
        (used when instantiating the prediction model).
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
    get_targets_and_prediction
        Returns the target and the predictions for a given dataset.
    predict_output_and_compute_eval_metric
        Compute predictions from models and then evaluation metrics.
    concat_and_transform
        Generate cartesian product between cell lines and dependencies of interest.


    """

    def __init__(
        self,
        data,
        preprocessor,
        rpz_model,
        fgps_rpz_model,
        pred_model,
        logging_path,
        metric,
        save_predictions: bool = False,
        bootstrapping: int = 200,
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
        finetuning: bool = False,
        use_for_gtex_tcga_inference: bool = False,
        train_test_no_cv: bool = False,
        split_ratio: float = 0.2,
        test_split_random_seed: int = 0,
        n_repeats_cv: int = 3,
        n_random_seed_cv_start_at: int = 0,
        n_splits: int = 5,
    ):
        # pylint: disable=too-many-statements
        self.data_cfg = data
        self.preprocessor_cfg = preprocessor
        self.rpz_cfg = rpz_model
        self.fgps_rpz_cfg = fgps_rpz_model
        self.pred_cfg = pred_model
        self.logging_path = logging_path
        self.metric_cfg = metric
        self.save_predictions = save_predictions
        self.bootstrapping = bootstrapping
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
        self.finetuning = finetuning
        self.rpz_model = None
        self.fgps_rpz_model = None
        self.pred_model = None
        self.prefitted_scaler = None
        self.rpz_model_pretrained = None
        self.preprocessor = None
        self.use_for_gtex_tcga_inference = use_for_gtex_tcga_inference
        self.rnaseq_variables = None
        self.train_test_no_cv = train_test_no_cv
        self.split_ratio = split_ratio
        self.test_split_random_seed = test_split_random_seed
        self.n_repeats_cv = n_repeats_cv
        self.n_random_seed_cv_start_at = n_random_seed_cv_start_at
        self.n_splits = n_splits
        self.ensembling_normalization = (
            self.pred_cfg['_target_'] == 'omics_rpz.models.EnsembleNormalizations'
        )
        self.augment_norm = None

        self.metric = instantiate(self.metric_cfg)
        self.is_hierarchical = (
            self.pred_cfg._target_ == 'omics_rpz.models.HierarchicalRegression'
        )

        if pretraining and "pretrain_dataset" not in data:
            raise AssertionError
        if finetuning_rpz and not pretraining:
            raise AssertionError

        if (
            pretraining
            and rpz_model.get('_target_').split('.')[-1] == 'AutoEncoderMultiHead'
        ):
            raise AssertionError

        if finetuning and pred_model.get('_target_').split('.')[-1] != 'MLPPrediction':
            raise AssertionError
        if noise_training and noise_level_training == 0.0:
            raise AssertionError
        if noise_pretraining and noise_level_pretraining == 0.0:
            raise AssertionError

    # Technically gtex inference is not a task that we benchmark.
    def load_gtex_tcga_and_intersect_them(
        self, ccle_rnaseq: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load gtex and tcga and intersect the genes between all 3 (with ccle).

        Args:
            ccle_rnaseq (pd.DataFrame): CCLE gene expression.
        Returns:
            tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: 3 dfs with
                intersection of genes
            Will also update self.rnaseq_variables:
                list of genes to use after intersection with tcga, gtex.
        """
        logger.info("Loading TCGA and GTEx, calculating gene intersection with CCLE")

        gtex_rnaseq, shared_rnaseq_variables_gtex = load_multiple_cohorts_gtex()

        (
            tcga_rnaseq,
            shared_rnaseq_variables_tcga,
        ) = load_multiple_cohorts_tcga(cohorts="ALL_TCGA_COHORTS")

        rnaseq_variables_intersection = sorted(
            list(
                # CCLE rnaseq variables
                set(self.rnaseq_variables)
                & set(shared_rnaseq_variables_gtex)
                & set(shared_rnaseq_variables_tcga)
            )
        )
        self.rnaseq_variables = rnaseq_variables_intersection

        ccle_rnaseq = ccle_rnaseq[self.rnaseq_variables]
        tcga_rnaseq = tcga_rnaseq[self.rnaseq_variables]
        gtex_rnaseq = gtex_rnaseq[self.rnaseq_variables]

        return ccle_rnaseq, tcga_rnaseq, gtex_rnaseq

    def concat_and_transform(
        self,
        exp_prepro_reduced: pd.DataFrame,
        fingerprints_depoi_reduced: pd.DataFrame,
        dependencies_depoi: pd.DataFrame = None,
    ) -> tuple[pd.DataFrame, pd.Series]:
        """Generate cartesian product between cell lines and dependencies of interest.

        If the prediction model is hierarchical simply return X, y as it is.
        For all other prediction models, creates the dataframe of size
        (N_GENES*N_CELL_LINES, RNA_RPZ_DIM + FINGERPRINTS_RPZ_DIM) used for the gene
        essentiality task, using the exp_prepro expression df and the fingerprints df
        concatenated together similar to a cartesian product. To return y, we use the
        dataframe dependencies_depoi (Genes, Cell Lines) filled with essentiality scores
        from DepMap, and format it similarly ((N_GENES*N_CELL_LINES,1).
        See plot here: ggslide id: 1EsgcyFukOg92Eaxy1PmuRMC8OslHUFJO0y9QYUzttjQ

        Args:
            exp_prepro_reduced (pd.DataFrame): Size (N_CELL_LINES, RNA_RPZ_DIM)
                Transformed CCLE RNA data.

            fingerprints_depoi_reduced (pd.DataFrame): Size (N_GENES, FGPS_RPZ_DIM)
                Transformed fingerprints data.

            dependencies_depoi (pd.DataFrame):  Size (N_CELL_LINES, N_GENES)
                Dataframe (Genes, Cell Lines) filled with essentiality scores from
                DepMap.

        Returns:
            tuple[pd.DataFrame, pd.DataFrame]:
                features: Features dataframe for the task.
                essentiality_scores: Label dataframe for the task (only meaningful
                    when dependencies_depoi is not None).
        """
        time_0 = time.time()

        exp_prepro_reduced.sort_index(inplace=True)

        if self.is_hierarchical:
            # Hierarchical model
            features, essentiality_scores = (
                exp_prepro_reduced,
                dependencies_depoi.loc[exp_prepro_reduced.index],
            )
        else:
            fingerprints_depoi_reduced.sort_index(inplace=True)

            if dependencies_depoi is None:
                # create dummy df to do the cartesian product
                dependencies_depoi = pd.DataFrame(
                    0,
                    index=exp_prepro_reduced.index,
                    columns=fingerprints_depoi_reduced.index,
                )
                dependencies_depoi.index.names = ["DepMap_ID"]

            expanded_frame = self.pivot_essentiality_in_one_column(dependencies_depoi)
            if self.normalizations_augment:
                # Need a specific logic to align the y vector
                # Multiindex is required with normalizations otherwise the join cant be
                # performed
                expanded_frame = expanded_frame.join(
                    exp_prepro_reduced.reset_index().set_index(
                        ['DepMap_ID', 'normalization']
                    ),
                    how='inner',
                )
                expanded_frame = expanded_frame.reset_index().set_index(
                    ['DepMap_ID', 'DepOI']
                )
                expanded_frame = expanded_frame.join(
                    fingerprints_depoi_reduced, how="inner"
                )
                ref_normalization = expanded_frame['normalization'].iloc[0]
                features = expanded_frame.drop(["essentiality_score"], axis=1)
                essentiality_scores = expanded_frame[
                    expanded_frame["normalization"] == ref_normalization
                ]["essentiality_score"]

            else:
                expanded_frame = expanded_frame.join(
                    exp_prepro_reduced, how="inner"
                ).join(fingerprints_depoi_reduced, how="inner")
                features = expanded_frame.drop(["essentiality_score"], axis=1)
                essentiality_scores = expanded_frame["essentiality_score"]

        # Log assembling times
        delta_time = time.time() - time_0
        logger.info(f"Assembling took {delta_time:.2f} seconds in total")

        return features, essentiality_scores

    def pivot_essentiality_in_one_column(
        self, y_essentiality: pd.DataFrame
    ) -> pd.DataFrame:
        """Create one target column from all the essentiality columns.

        Args:
            y_essentiality (pd.DataFrame): df of shape cell lines x DepOi essentiality

        Returns:
            pd.DataFrame: DataFrame of length cell lines x DepOi essentiality.
        """
        y_essentiality = y_essentiality.reset_index().melt(id_vars=["DepMap_ID"])
        y_essentiality = y_essentiality.rename(
            columns={"variable": "DepOI", "value": "essentiality_score"}
        )
        y_essentiality = y_essentiality.set_index(["DepMap_ID", "DepOI"]).sort_index()
        return y_essentiality

    def infer_essentiality_score(
        self,
        df_rnaseq: pd.DataFrame,
        df_fgps_rpz: pd.DataFrame,
        file_to_save: str,
    ) -> pd.DataFrame:
        """Use model to infer essentiality scores.

        Can be used for gtex or tcga.

        Args:
            df_rnaseq (pd.DataFrame): gene expression data.
            df_fgps_rpz (pd.DataFrame): fingerprints matrix (see above) transformed by
                fgps_rpz_model.
            file_to_save (str): eg:'GTEx_scores.csv'. If set to None no saving
                will be done.
        Returns:
            df_essentiality_scores: Dataframe with essentiality score, with columns:
                ['(Sample, Gene)','Score']
        """
        logger.info("Pre-processing RNASeq data")
        X_rna = self.preprocessor.transform(df_rnaseq)

        if self.finetuning:
            X_rna_rpz = X_rna
        else:
            logger.info("RPZ inference on RNASeq data")
            X_rna_rpz = rpz_transform_wrapper(self.rpz_model, X_rna)

        logger.info("Assembling data for inference and gene fingerprints RPZ")

        X, _y = self.concat_and_transform(X_rna_rpz, df_fgps_rpz)

        logger.info("Essentiality Prediction")
        y_hat = self.pred_model.predict(X.to_numpy())

        df_index = pd.DataFrame(X.index, columns=['(Sample, Gene)'])
        df_scores = pd.DataFrame(y_hat, columns=['Score'])
        df_essentiality_scores = pd.concat([df_index, df_scores], axis=1)

        if file_to_save is not None:
            dir_path = Path(__file__).parent.parent.parent.absolute()
            dir_path = Path(dir_path).joinpath("outputs/")
            file = Path(dir_path).joinpath(file_to_save)
            df_essentiality_scores.to_csv(file)
        return df_essentiality_scores

    def pretrain_rpz_model(self, df_train_val) -> None:
        """Function to handle pretraining of the rpz_model if needed.

        Intersect genes between pretraining and training datasets. Fit preprocessor with
        training dataset to choose the most variant genes based on training dataset.
        Fit_transform of preprocessor and then rpz_model on pretraining dataset.

        If self.pretraining_ae_path is not None, load the scaler and the pretrained AE
        from the directory corresponding to the string and do not consider the
        df_train_val object.

        Args:
            df_train_val (pd.DataFrame): gene expression data for training/validation.
        """

        if self.pretrained_ae_path is not None:
            logger.info("Loading pretrained autoencoder...")
            pretrained_output_pt = Path(self.pretrained_ae_path)
            with open(pretrained_output_pt / "scaler.pkl", "rb") as pretrained_rpz_pkl:
                preprocessor = pickle.load(pretrained_rpz_pkl)
            with open(pretrained_output_pt / "ae.pkl", "rb") as pretrained_rpz_pkl:
                pretrained_rpz_model = pickle.load(pretrained_rpz_pkl)
            self.preprocessor = preprocessor
            self.rpz_model = pretrained_rpz_model

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
        X_pretrain = df_pretrain[self.rnaseq_variables]

        # Pre-training preprocessing
        logger.info("Pre-training Preprocessing")
        # We create the gene filters based on the training dataset.
        self.preprocessor = instantiate(self.preprocessor_cfg)
        self.preprocessor.fit(df_train_val[self.rnaseq_variables])
        # Now the gene list is stored in the self.preprocessor to be used for CCLE.
        X_prep_pretrain = self.preprocessor.fit_transform(X_pretrain)

        # Add noise AFTER pre-processing
        if self.noise_pretraining:
            # We add gaussian noise once all variables have been normalized.
            noiser = GaussianNoise(
                variables_to_perturbate=X_prep_pretrain.columns,
                gaussian_std=self.noise_level_pretraining,
                number_of_duplication=self.n_augment_pretraining,
            )
            X_prep_pretrain = noiser.transform(X_prep_pretrain)

        # Instantiate and fit rpz model
        logger.info("Pre-training RPZ model")
        self.rpz_model = instantiate(self.rpz_cfg)
        self.rpz_model.fit(X_prep_pretrain)

        if self.finetuning_rpz:
            self.rpz_model_pretrained = copy.deepcopy(self.rpz_model)

        return self

    def train(
        self,
        ccle_rnaseq: pd.DataFrame,
        df_fgps_rpz: pd.DataFrame,
        dependencies_depoi: pd.DataFrame,
        sample_info: pd.DataFrame,
        metrics_suffix: Optional[str] = None,
    ) -> None:
        """Handles training pipeline with a train dataset. Allows the use of the
        pipeline with different inputs (if crossval or test).

        Parameters
        ----------
        ccle_rnaseq : pd.DataFrame
            Training cell RNAseq dataset
        df_fgps_rpz : pd.DataFrame
            Gene fingerprints RPZ dataset
        dependencies_depoi : pd.DataFrame
            Dependencies dataset
        sample_info : pd.DataFrame
            Cell line info training dataset
        metrics_suffix : Optional[str], optional
            Optional name for logging in CV mode, by default None
        """
        time_0 = time.time()
        # Need to filter X_train on self.rnaseq_variables in the event of pretraining.
        X_ccle_rna = ccle_rnaseq[self.rnaseq_variables]

        # preprocessing
        logger.info("Pre-processing CCLE RNASeq data")
        if not self.pretraining:
            # Need to instantiate the preprocessor if not done in pretraining (cannot be
            # done elsewhere as in case of CV it needs a new preprocessor at every fold)
            self.preprocessor = instantiate(self.preprocessor_cfg)
            X_prep_train = self.preprocessor.fit_transform(X_ccle_rna)
        elif self.pretraining and not self.finetuning_rpz:
            X_prep_train = self.preprocessor.transform(X_ccle_rna)
        elif self.pretraining and self.finetuning_rpz:
            self.preprocessor = copy.deepcopy(self.prefitted_scaler)
            X_prep_train = self.preprocessor.fit_transform(X_ccle_rna)

        if self.normalizations_augment is not None:
            X_prep_train, normalizations_col = self.augment_norm.concatenate_augmented(
                X_prep_train, fit_preprocessor=True
            )

        # Add noise AFTER pre-processing
        if self.noise_training:
            logger.info("Adding noise to CCLE RNAseq")
            # We add gaussian noise once all variables have been normalized.
            noiser = GaussianNoise(
                variables_to_perturbate=X_prep_train.columns,
                gaussian_std=self.noise_level_training,
                number_of_duplication=self.n_augment_training,
            )
            X_prep_train, dependencies_depoi = noiser.transform_with_labels(
                X_prep_train, dependencies_depoi
            )

        # representation model
        if not self.pretraining:
            logger.info("Training CCLE RPZ method")
            # if there is no pretraining the representation model needs to be
            # instantiated and fitted on the training data.
            self.rpz_model = instantiate(self.rpz_cfg)
            self.rpz_model.fit(
                X=X_prep_train,
                metrics_suffix=metrics_suffix,
                df_labels=sample_info.join(dependencies_depoi, how='inner').loc[
                    X_prep_train.index
                ],
            )
            # the .loc enforces that X and df_labels have the same index, useful for
            # data augmentation cases

        if self.finetuning_rpz:
            # If pretrained, re-train the rpz model on the train dataset
            logger.info("Finetuning RPZ method")

            self.rpz_model = copy.deepcopy(self.rpz_model_pretrained)

            if self.rpz_cfg["early_stopping_use"]:
                # then early_stopping_use has been changed to False after pretraining
                # and num_epochs to number of epochs found by early stopping pretraining
                # so reset them in this case
                self.rpz_model.early_stopping_use = True
                self.rpz_model.num_epochs = self.rpz_cfg["max_num_epochs"]
                self.rpz_model.early_stopping_delta = self.rpz_cfg[
                    "early_stopping_delta"
                ]
                self.rpz_model.early_stopping_patience = self.rpz_cfg[
                    "early_stopping_patience"
                ]

            self.rpz_model.fit(
                X=X_prep_train,
                metrics_suffix=metrics_suffix,
                df_labels=sample_info.join(dependencies_depoi, how='inner').loc[
                    X_prep_train.index
                ],
                finetuning_rpz=True,
            )
            # the .loc enforces that X and df_labels have the same index, useful for
            # data augmentation cases

        # prediction model
        if self.finetuning:
            # with finetuning, the construction of the model is different as the
            # prediction model needs the representation model to be initialised.
            self.pred_cfg["_partial_"] = True
            # In the case of hierarchical or ensembling model need recursive = False to
            # instantitate pred_hierarchical or the ensemble.
            # For other models we use the default behavior recursive = True
            pred_model_constructor = instantiate(
                self.pred_cfg,
                _recursive_=not (self.is_hierarchical or self.ensembling_normalization),
            )
            self.pred_model = pred_model_constructor(
                # Deep copy of rpz model to ensure it doesn't change.
                auto_encoder=copy.deepcopy(self.rpz_model),
                end_dimensions_to_not_transform=self.fgps_rpz_cfg["repr_dim"],
            )
            df_ccle_rna_rpz = X_prep_train
        if not self.finetuning:
            # In the case of hierarchical or ensembling model need recursive = False to
            # instantitate pred_hierarchical or the ensemble.
            # For other models we use the default behavior recursive = True
            self.pred_model = instantiate(
                self.pred_cfg,
                _recursive_=not (self.is_hierarchical or self.ensembling_normalization),
            )
            df_ccle_rna_rpz = rpz_transform_wrapper(self.rpz_model, X_prep_train)

        # Here add back the normalization columns saved previously.
        if self.normalizations_augment is not None:
            df_ccle_rna_rpz["normalization"] = normalizations_col

        # Log preprocessing and rpz times
        delta_time1 = time.time() - time_0
        logger.info(f"Preprocessing and RPZ took {delta_time1:.2f} seconds in total")

        # Create dataset for prediction model
        logger.info("Assembling CCLE (RPZ or RNASeq), gene fgp RPZ and ground truth")

        df_training, y_training = self.concat_and_transform(
            df_ccle_rna_rpz,
            df_fgps_rpz,
            dependencies_depoi,
        )

        time_1 = time.time()
        # Prediction model training
        self.pred_model.fit(df_training, y_training)

        # Log training times
        delta_time2 = time.time() - time_1
        logger.info(f"Training model took {delta_time2:.2f} seconds in total")

    def get_targets_and_prediction(
        self,
        ccle_rnaseq: pd.DataFrame,
        df_fgps_rpz: pd.DataFrame,
        dependencies_depoi: pd.DataFrame,
    ) -> tuple[pd.Series, np.ndarray]:
        """Returns the target and the predictions for a given dataset.

        Parameters
        ----------
        ccle_rnaseq : pd.DataFrame
            cell RNAseq dataset
        df_fgps_rpz : pd.DataFrame
            Gene fingerprints RPZ dataset
        dependencies_depoi : pd.DataFrame
            Dependencies dataset
        Returns
        -------
        tuple[pd.DataFrame, np.ndarray]
            Targets and predictions.
        """
        # Need to filter ccle_rnase on self.rnaseq_variables in the event of
        # pretraining.
        X_ccle_rna = ccle_rnaseq[self.rnaseq_variables]

        # Preprocessing
        X_ccle_rna_preprocessed = self.preprocessor.transform(X_ccle_rna)

        if self.augment_norm and self.ensembling_normalization:
            (
                X_ccle_rna_preprocessed,
                normalization_col,
            ) = self.augment_norm.concatenate_augmented(
                X_ccle_rna_preprocessed,
                fit_preprocessor=False,
            )

        # RPZ model
        if self.finetuning:
            X_ccle_rna_rpz = X_ccle_rna_preprocessed
        else:
            X_ccle_rna_rpz = rpz_transform_wrapper(
                self.rpz_model, X_ccle_rna_preprocessed
            )

        # self.normalizations_augment and self.ensembling_normalization
        if self.normalizations_augment is not None:
            X_ccle_rna_rpz['normalization'] = normalization_col

        df_rna_fgps_combined, y_essentiality = self.concat_and_transform(
            X_ccle_rna_rpz,
            df_fgps_rpz,
            dependencies_depoi,
        )

        # Here y_essentiality is a pd.Series (with index), and the models predictions is
        # a np.array - without index.
        y_preds = self.pred_model.predict(df_rna_fgps_combined)
        if self.is_hierarchical:
            # Hierarchical model.
            y_true = self.pivot_essentiality_in_one_column(y_essentiality)[
                'essentiality_score'
            ]
        else:
            y_true = y_essentiality
        return y_true, y_preds

    def predict_output_and_compute_eval_metric(
        self,
        ccle_rnaseq: pd.DataFrame,
        df_fgps_rpz: pd.DataFrame,
        dependencies_depoi: pd.DataFrame,
    ) -> tuple[pd.Series, np.ndarray, float]:
        """Returns the prediction, ground truth label and evaluation metric of the
        trained model on a dataset.

        Parameters
        ----------
        ccle_rnaseq : pd.DataFrame
            cell RNAseq dataset
        df_fgps_rpz : pd.DataFrame
            Gene fingerprints RPZ dataset
        dependencies_depoi : pd.DataFrame
            Dependencies dataset
        Returns
        -------
        tuple[pd.Series, np.ndarray, float]
            predictions, labels and metric.
        """
        y_eval, y_preds_eval = self.get_targets_and_prediction(
            ccle_rnaseq, df_fgps_rpz, dependencies_depoi
        )
        # np.Series and np.array can be used here.
        return y_preds_eval, y_eval, self.metric(y_eval, y_preds_eval)[0]

    def run_no_cv(
        self,
        ccle_rnaseq_train_val: pd.DataFrame,
        ccle_rnaseq_test: pd.DataFrame,
        df_fgps_rpz: pd.DataFrame,
        dependencies_depoi: pd.DataFrame,
        sample_info_train_val: pd.DataFrame,
    ) -> tuple[dict, dict, pd.DataFrame]:
        """Trains model and test it in the case that there is no cross-validation.

        Parameters
        ----------
        ccle_rnaseq_train_val : pd.DataFrame
            Training cell RNAseq dataset
        ccle_rnaseq_test : pd.DataFrame
            Testing cell RNAseq dataset
        df_fgps_rpz : pd.DataFrame
            Gene fingerprints RPZ dataset
        dependencies_depoi : pd.DataFrame
            Dependencies dataset
        sample_info_train_val : pd.DataFrame
            Cell line info training dataset

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
        self.train(
            ccle_rnaseq_train_val,
            df_fgps_rpz,
            dependencies_depoi,
            sample_info_train_val,
        )
        _, _, train_metric = self.predict_output_and_compute_eval_metric(
            ccle_rnaseq_train_val, df_fgps_rpz, dependencies_depoi
        )

        # Log split info
        logger.info("Final Test")
        logger.info(f"Train set cell lines: {len(ccle_rnaseq_train_val)}")
        logger.info(f"Test set cell lines: {len(ccle_rnaseq_test)}")
        logger.info(f"Train metric: {train_metric:.3f}")

        # Get predictions on test set for bootstrapping
        y_test, y_pred_test = self.get_targets_and_prediction(
            ccle_rnaseq_test, df_fgps_rpz, dependencies_depoi
        )

        # Save experiment results
        if self.save_predictions:
            prediction_path = (
                Path(self.logging_path).parent / "test_predictions.parquet"
            )
            save_predictions(y_true=y_test, y_pred=y_pred_test, path=prediction_path)

        # Bootstrapping on test set
        scores = []
        sampling_rate = 1
        logger.info(
            f"Calc scores {self.bootstrapping} times ({sampling_rate} sampling rate)"
        )
        np.random.seed(0)
        for _ in range(self.bootstrapping):
            random_indices = np.random.choice(
                len(y_pred_test),
                size=int(len(y_pred_test) * sampling_rate),
                replace=True,
            )
            y_subsample_hat, y_subsample_test = (
                y_pred_test[random_indices],
                y_test.values[random_indices],
            )
            scores.append(
                self.metric(y_subsample_test.reshape(-1), y_subsample_hat.reshape(-1))[
                    0
                ]
            )

        # Metrics
        mean_metric = np.mean(scores)
        ci_95_low, ci_95_high = np.quantile(scores, 0.025), np.quantile(scores, 0.975)

        # MLFlow: Log metrics
        mlflow.log_metric("train_metric_corr_coeff", train_metric)
        mlflow.log_metric("test_metric_corr_coeff", mean_metric)
        mlflow.log_metric("test_metric_corr_coeff_95_CI_low", ci_95_low)
        mlflow.log_metric("test_metric_corr_coeff_95_CI_high", ci_95_high)

        # Logger: Log metrics
        logger.info(f"metric correlation bootstrapping mean is {mean_metric:.3f}")
        logger.info(f"95% Confidence Interval: [{ci_95_low:.3f}, {ci_95_high:.3f}]")

        res = pd.DataFrame(
            data={
                "REPEAT": np.arange(self.bootstrapping),
                "test_metric": scores,
                "nb_patients_val": len(random_indices),
            }
        )
        return (
            {0: y_pred_test},
            {0: y_test},
            res,
        )

    def run_cv(
        self,
        ccle_rnaseq_train_val: pd.DataFrame,
        df_fgps_rpz: pd.DataFrame,
        dependencies_depoi: pd.DataFrame,
        sample_info_train_val: pd.DataFrame,
    ) -> tuple[dict, dict, pd.DataFrame]:
        """Given one dataframe, runs a cross-validation for the task n_repeat times.

        Parameters
        ----------
        ccle_rnaseq_train_val : pd.DataFrame
            Training cell RNAseq dataset
        df_fgps_rpz : pd.DataFrame
            Gene fingerprints RPZ dataset
        dependencies_depoi : pd.DataFrame
            Dependencies dataset
        sample_info_train_val : pd.DataFrame
            Cell line info training dataset

        Returns
        -------
        tuple[dict, dict, pd.DataFrame]
            prediction and label dictionaries with one key per fold, and dataframe
            that contains the metrics on the validation sets.
        """
        # pylint: disable=duplicate-code
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
            # Here as we split before fgps concatenation we already split on the cell
            # ids and ensure we don't have the same cell in the train and test.
            splits_gen = create_fold_iterator(
                df_all=ccle_rnaseq_train_val,
                groups=None,
                stratify=None,
                random_seed=random_state,
            )
            for split, (train_indices, val_indices) in enumerate(splits_gen):
                ccle_rnaseq_train = ccle_rnaseq_train_val.iloc[train_indices]
                ccle_rnaseq_val = ccle_rnaseq_train_val.iloc[val_indices]
                depmap_ids_train = ccle_rnaseq_train.index.to_numpy()
                sample_info_train = sample_info_train_val.loc[depmap_ids_train]

                metrics_suffix = f"rep{random_state} split{split}"

                # Train model
                logger.info("Training...")
                self.train(
                    ccle_rnaseq_train,
                    df_fgps_rpz,
                    dependencies_depoi,
                    sample_info_train,
                    metrics_suffix,
                )
                # Predict
                logger.info("Predict...")
                (
                    _,
                    _,
                    train_metric,
                ) = self.predict_output_and_compute_eval_metric(
                    ccle_rnaseq_train, df_fgps_rpz, dependencies_depoi
                )
                (
                    y_pred_val,
                    y_val,
                    valid_metric,
                ) = self.predict_output_and_compute_eval_metric(
                    ccle_rnaseq_val, df_fgps_rpz, dependencies_depoi
                )

                # Log split info
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

                nb_samples_train.append(len(train_indices))
                nb_samples_val.append(len(val_indices))

                predictions_val[split], labels_val[split] = y_pred_val, y_val

        # Compute cross-validation metrics
        mean_cv_train_metric = np.mean(cv_train_metric)
        std_cv_train_metric = np.std(cv_train_metric)

        mean_cv_val_metric = np.mean(cv_val_metric)
        std_cv_val_metric = np.std(cv_val_metric)

        # Log cross-validation metrics
        logger.info("Cross-validation metrics")
        logger.info(
            f"Training metric: {mean_cv_train_metric:.3f} ({std_cv_train_metric:.3f})"
        )
        logger.info(
            f"Validation metric: {mean_cv_val_metric:.3f} ({std_cv_val_metric:.3f})"
        )
        logger.info("\n")

        # MLFlow: Log cross-val metrics
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

    def run(self) -> tuple[dict, dict, pd.DataFrame]:
        """Function to run the training and evaluation of the gene essentiality score
        with pre-training and/or fine-tuning if desired.

        Returns
        -------
        tuple[dict, dict, pd.DataFrame]
            prediction and label dictionaries with one key per fold, and dataframe
            that contains the metrics of the run.
        """

        # Load data CCLE
        logger.add(self.logging_path)
        logger.info("Loading CCLE and fingerprint data")
        load_data = instantiate(self.data_cfg["dataset"])
        ccle_rnaseq, fingerprints_depoi, dependencies_depoi, sample_info = load_data()

        # CCLE Rnaseq variables.
        self.rnaseq_variables = sorted(ccle_rnaseq.columns)

        # Load data GTEX and TCGA for inference
        if self.use_for_gtex_tcga_inference:
            logger.info("Loading CCLE and fingerprint data")
            (
                ccle_rnaseq,
                tcga_rnaseq,
                gtex_rnaseq,
            ) = self.load_gtex_tcga_and_intersect_them(ccle_rnaseq)

        # Log data info
        logger.info("Data info")
        logger.info(f"Number of cells: {len(ccle_rnaseq)}")
        logger.info(f"Number of RNAseq variables: {len(self.rnaseq_variables)}")
        logger.info(f"Number of genes: {len(fingerprints_depoi)}")
        logger.info("\n")

        # Fit Transform RPZ for gene fingerprints
        logger.info("Training and inferring gene fingerprint RPZ method")
        self.fgps_rpz_model = instantiate(self.fgps_rpz_cfg)
        df_fgps_rpz = rpz_fit_transform_wrapper(
            self.fgps_rpz_model, fingerprints_depoi, data_name="fgps"
        )

        # Split train/test cell RNAseq data and sample info

        # If the split is done prior to the concat we don't need to pass depmad_ids as
        # argument it will naturally split on that.

        # We can add here more groups on which we may want to do the split.
        ccle_rnaseq_train_val, ccle_rnaseq_test = test_split(
            df_all=ccle_rnaseq,
            # need to split by cell lines but it's already the case here because fpgs
            # haven't been added.
            # Could add primary disease here for instance.
            groups=None,
            split_ratio=self.split_ratio,
            random_seed=self.test_split_random_seed,
        )
        sample_info_train_val = sample_info.loc[ccle_rnaseq_train_val.index]

        # Pretraining of preprocessor and RPZ model
        if self.pretraining:
            self.pretrain_rpz_model(ccle_rnaseq_train_val)

        # Note: No need to split dependencies_depoi, for two reasons:
        # 1) Inner joins will be performed, automatically selecting only training data;
        # 2) It reduces the number of arguments of run_no_cv and run_cv.

        if self.train_test_no_cv:
            # Run training and predict without cross-validation
            pred_dict, label_dict, df_results = self.run_no_cv(
                ccle_rnaseq_train_val,
                ccle_rnaseq_test,
                df_fgps_rpz,
                dependencies_depoi,
                sample_info_train_val,
            )
        else:
            pred_dict, label_dict, df_results = self.run_cv(
                ccle_rnaseq_train_val,
                df_fgps_rpz,
                dependencies_depoi,
                sample_info_train_val,
            )

        # Inference mode on GTEx and TCGA
        if self.use_for_gtex_tcga_inference:
            logger.info("Gtex predictions of essentiality.")
            # Remove filename to prevent saving of files.
            _ = self.infer_essentiality_score(
                gtex_rnaseq, df_fgps_rpz, 'GTEx_scores.csv'
            )
            logger.info("TCGA predictions of essentiality.")
            _ = self.infer_essentiality_score(
                tcga_rnaseq, df_fgps_rpz, 'TCGA_scores.csv'
            )

        return pred_dict, label_dict, df_results
