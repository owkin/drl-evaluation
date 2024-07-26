"""Init file for the transforms functions of the package.

Includes data augmentation, preprocessing and scaling, and representation models.
"""

from .data_augmentation.gaussian_noise import GaussianNoise
from .preprocessors.rnaseq_preprocessor import RNASeqPreprocessor
from .preprocessors.scalers import MedianRatioScaler
from .representation_models.auto_encoder import AutoEncoder
from .representation_models.auto_encoder_multi_head import AutoEncoderMultiHead
from .representation_models.identity import Identity
from .representation_models.masked_auto_encoder import MaskedAutoencoder
from .representation_models.pca import PCA
from .representation_models.random_rpz import Random
from .representation_models.scvi import ScVI
from .representation_models.vae import VariationalAutoencoder
from .representation_models.wrappers import (
    rpz_fit_transform_wrapper,
    rpz_transform_wrapper,
)

try:
    from .representation_models.gnn import Gnn
except (ModuleNotFoundError, ImportError):
    pass
