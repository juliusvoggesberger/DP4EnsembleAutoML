"""
Module implementing a wrapper for PCA.
"""
import numpy as np
from ConfigSpace import ConfigurationSpace, CategoricalHyperparameter, UniformFloatHyperparameter
from sklearn.decomposition import PCA

from auto_cen.constants import MIXED, FEATURERED, PCATR
from auto_cen.pipeline.base import BaseAlgorithm
from auto_cen.pipeline.transformations.base_preprocessor import BasePreprocessor


class PCAReduction(BasePreprocessor):
    """
    Applies PCA to the data.

    :param whiten: Whitening strategy to use. Either True or False.
    :param n_components: Number of components to used, specified by the variance to be explained.
                         Variance explained has to be in the interval [0.5, 0.99].
    :param seed: Random seed.
    :return: The transformed data.
    """

    def __init__(self, whiten: str, n_components:float, seed: int = None):

        super().__init__(seed)
        self.whiten = whiten
        self.n_components = n_components
        self.model = PCA(n_components=self.n_components, whiten=self.whiten, random_state=self.seed)


    def fit(self, X: np.array, y: np.array) -> BaseAlgorithm:
        self.model.fit(X)

        return self

    def transform(self, X: np.array = None, y: np.array = None) -> (np.array, np.array):
        X_enc = self.model.transform(X)
        return X_enc, y

    def get_params(self, deep=True) -> dict:

        return {
            'n_components': self.n_components,
            'whiten': self.whiten,
            'seed': self.seed,
        }

    @staticmethod
    def get_specification_config() -> dict:

        return {'name': PCATR,
                'algorithm': FEATURERED,
                'is_deterministic': False,
                'input': MIXED,
                }

    @staticmethod
    def get_config_space() -> ConfigurationSpace:

        c_space = ConfigurationSpace()
        n_components = UniformFloatHyperparameter('n_components', lower=0.5, upper=0.99)
        whiten = CategoricalHyperparameter('whiten', [True, False])
        c_space.add_hyperparameters([whiten, n_components])

        return c_space
