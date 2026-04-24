# https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.GenericUnivariateSelect.html#sklearn.feature_selection.GenericUnivariateSelect
"""
Module implementing a wrapper for mutual information for a discrete target variable.
"""

import numpy as np
from ConfigSpace import ConfigurationSpace,  UniformFloatHyperparameter
from sklearn.feature_selection import GenericUnivariateSelect, mutual_info_classif

from auto_cen.constants import MIXED, FEATURERED, MUTUALINFO
from auto_cen.pipeline.base import BaseAlgorithm
from auto_cen.pipeline.transformations.base_preprocessor import BasePreprocessor

import warnings
from scipy.sparse import SparseEfficiencyWarning
warnings.simplefilter('ignore', SparseEfficiencyWarning)


class Mutual_Information(BasePreprocessor):
    """
    Applies estimation of mutual information for a discrete target variable to the data.
    Measures dependency between the variables."

    :param mode: Indicates which strategy is used to select the features.
    :param max_param_fraction: Controls how many features are selected.
                               Value range between [0.0,1.0].
                               The fraction is multiplied by the n_features of the dataset to
                               compute the number of features.
    :param seed: Random seed.
    :return: The transformed data.
    """

    def __init__(self, max_param_fraction: float = 0.7, seed: int = None):
        super().__init__(seed)
        self.max_param = 0
        self.max_param_fraction = max_param_fraction
        self.score_function = mutual_info_classif
        self.model = None

    def fit(self, X: np.array, y: np.array) -> BaseAlgorithm:
        n_features = X.shape[1]
        self.max_param = max(1, int(self.max_param_fraction * n_features))
        self.model = GenericUnivariateSelect(score_func=self.score_function, mode='k_best', param=self.max_param)
        self.model.fit(X, y)

        return self

    def transform(self, X: np.array = None, y: np.array = None) -> np.array:
        X_enc = self.model.transform(X)
        return X_enc, y

    def get_params(self, deep=True) -> dict:
        return {
            'max_param': self.max_param,
            'max_param_fraction': self.max_param_fraction,
            'selected_features':self.model.get_feature_names_out(),
            'seed': self.seed,
        }

    @staticmethod
    def get_specification_config() -> dict:
        return {'name': MUTUALINFO,
                'algorithm': FEATURERED,
                'is_deterministic': True,
                'input': MIXED,
                }

    @staticmethod
    def get_config_space() -> ConfigurationSpace:
        c_space = ConfigurationSpace()
        max_param_fraction = UniformFloatHyperparameter('max_param_fraction', lower=0, upper=1, default_value=0.6)
        c_space.add_hyperparameters([max_param_fraction])

        return c_space
