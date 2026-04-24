"""
Module implementing a wrapper for EditedNearestNeighbours.
"""
from collections import Counter
import numpy as np
from ConfigSpace import ConfigurationSpace, CategoricalHyperparameter, UniformIntegerHyperparameter
from imblearn.under_sampling import EditedNearestNeighbours

from auto_cen.constants import MIXED, SAMPLING, ENN
from auto_cen.pipeline.base import BaseAlgorithm
from auto_cen.pipeline.transformations.base_preprocessor import BasePreprocessor


class EditedNearestNeighboursSampling(BasePreprocessor):
    """
    Undersampling algorithm. --> Application for data sets with class overlap with class imbalance

    :param sampling_strategy: Indicates which sampling strategy is used to delete Instances.
                             Class: ['not minority', 'majority', 'all'].
    :param n_neighbors: Defines number of neighbors for the kNN-Alg.
    :param kind_sel: Strategy to be used to exclude samples.
    :param seed: Random seed.
    :return: The transformed data.
    """

    def __init__(self, sampling_strategy: str = 'not minority', n_neighbors:int =3, kind_sel: str = 'all', seed: int = None):
        super().__init__(seed)

        self.sampling_strategy = sampling_strategy
        self.n_neighbors = n_neighbors
        self.kind_sel = kind_sel
        self.class_distribution_start = {}
        self.class_distribution_transformed = {}
        self.model = None

    def fit(self, X: np.array, y: np.array) -> BaseAlgorithm:
        self.model = EditedNearestNeighbours(sampling_strategy=self.sampling_strategy, n_neighbors= self.n_neighbors, kind_sel= self.kind_sel)
        self.model.fit(X, y)
        return self

    def transform(self, X: np.array = None, y: np.array = None) -> np.array:
        # Assumes that we are then in the prediction phase
        if y is None:
            return X, y
        self.class_distribution_start = Counter(y)
        X_enc, y_enc = self.model.fit_resample(X, y)
        self.class_distribution_transformed = Counter(y_enc)
        return X_enc, y_enc

    def get_params(self, deep=True) -> dict:
        return {
            'sampling_strategy': self.sampling_strategy,
            'kind_sel': self.kind_sel,
            'n_neighbors':self.n_neighbors,
            'class_distribution_start': self.class_distribution_start,
            'class_distribution_transformed': self.class_distribution_transformed,
            'seed': self.seed,
        }

    @staticmethod
    def get_specification_config() -> dict:
        return {'name':ENN,
                'algorithm': SAMPLING,
                'is_deterministic': True,
                'input': MIXED,
                }

    @staticmethod
    def get_config_space() -> ConfigurationSpace:
        c_space = ConfigurationSpace()
        sampling_strategy = CategoricalHyperparameter('sampling_strategy', ['not minority', 'majority', 'all'], default_value='not minority')
        n_neighbors = UniformIntegerHyperparameter('n_neighbors', lower=2, upper=10,
                                                   default_value=3)
        kind_sel = CategoricalHyperparameter('kind_sel', ['all', 'mode'], default_value='all')
      
        c_space.add_hyperparameters([sampling_strategy, n_neighbors, kind_sel ])

        return c_space
