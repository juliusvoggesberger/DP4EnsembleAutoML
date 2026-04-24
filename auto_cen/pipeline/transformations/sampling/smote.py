from collections import Counter
import numpy as np
from ConfigSpace import ConfigurationSpace, CategoricalHyperparameter, UniformIntegerHyperparameter
from imblearn.over_sampling import SMOTE

from auto_cen.constants import MIXED, SAMPLING, ROS, SMOTE_S
from auto_cen.pipeline.base import BaseAlgorithm
from auto_cen.pipeline.transformations.base_preprocessor import BasePreprocessor


class SMOTESampling(BasePreprocessor):
    """
    Generates synthetic data instances based on SMOTE. Used for datasets that have a small amount of data or
    class overlap with class imbalance.

    :param sampling_strategy: Specifies the class for which data instances are to be generated.
                             ['minority', 'not majority', 'all']
    :param random_state: Set initiale point for the algorithm.
    :param k_neighbors: Defines number of neighbors for the kNN-Alg.
    :param seed: Random seed.
    :return: The transformed data.
    """

    def __init__(self, sampling_strategy: str,random_state: int=123, k_neighbors: int  =5, seed: int = None):
        super().__init__(seed)
        self.random_state = random_state
        self.k_neighbors =  k_neighbors
        self.sampling_strategy = sampling_strategy
        self.model = None
        self.class_distribution_start = {}
        self.class_distribution_transformed = {}

    def fit(self, X: np.array, y: np.array) -> BaseAlgorithm:
        self.model = SMOTE(sampling_strategy=self.sampling_strategy, random_state=self.random_state,  k_neighbors=self.k_neighbors)
        self.model.fit(X, y)
        return self

    def transform(self, X: np.array = None, y: np.array = None) -> np.array:
        # Assumes that we are then in the prediction phase
        if y is None:
            return X, y
        self.class_distribution_start = Counter(y)
        X_enc,y_enc = self.model.fit_resample(X, y)
        self.class_distribution_transformed = Counter(y_enc)
        return X_enc, y_enc

    def get_params(self, deep=True) -> dict:
        return {
            'sampling_strategy': self.sampling_strategy,
            'random_state':self.random_state,
            'k_neighbors': self.k_neighbors,
            'class_distribution_start': self.class_distribution_start,
            'class_distribution_transformed':   self.class_distribution_transformed,
            'seed': self.seed,
        }

    @staticmethod
    def get_specification_config() -> dict:
        return {'name': SMOTE_S,
                'algorithm': SAMPLING,
                'is_deterministic': True,
                'input': MIXED,
                }

    @staticmethod
    def get_config_space() -> ConfigurationSpace:
        c_space = ConfigurationSpace()
        sampling_strategy = CategoricalHyperparameter('sampling_strategy', ['minority', 'not majority', 'all'],default_value='minority')
        random_state = UniformIntegerHyperparameter('random_state', lower=1, upper=400,
                                                    default_value=123)
        k_neighbors = UniformIntegerHyperparameter('k_neighbors', lower=1, upper=10,
                                                   default_value=5)
        c_space.add_hyperparameters([sampling_strategy, random_state, k_neighbors])

        return c_space

