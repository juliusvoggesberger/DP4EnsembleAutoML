from collections import Counter
import numpy as np
from ConfigSpace import ConfigurationSpace, CategoricalHyperparameter, UniformFloatHyperparameter
from imblearn.under_sampling import TomekLinks

from auto_cen.constants import MIXED, SAMPLING, TOMEK_LINKS
from auto_cen.pipeline.base import BaseAlgorithm
from auto_cen.pipeline.transformations.base_preprocessor import BasePreprocessor


class TomekLinksSampling(BasePreprocessor):
    """
    Undersampling algorithm TomekLinks. --> Application for data sets with class overlap with class imbalance

    :param sampling_strategy: Indicates which sampling strategy is used to delete Instances.
                             Class: ['not minority', 'majority', 'all'],

    :param seed: Random seed.
    :return: The transformed data.
    """

    def __init__(self, sampling_strategy: str, seed: int = None):
        super().__init__(seed)
        self.sampling_strategy = sampling_strategy
        self.class_distribution_start = {}
        self.class_distribution_transformed = {}
        self.model = None

    def fit(self, X: np.array, y: np.array) -> BaseAlgorithm:
        self.model = TomekLinks(sampling_strategy=self.sampling_strategy) #sampling_strategy=self.sampling_strategy)
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
            'class_distribution_start': self.class_distribution_start,
            'class_distribution_transformed': self.class_distribution_transformed,
            'seed': self.seed,
        }

    @staticmethod
    def get_specification_config() -> dict:
        return {'name': TOMEK_LINKS,
                'algorithm': SAMPLING,
                'is_deterministic': True,
                'input': MIXED,
                }

    @staticmethod
    def get_config_space() -> ConfigurationSpace:
        c_space = ConfigurationSpace()
        sampling_strategy = CategoricalHyperparameter('sampling_strategy', ['not minority', 'majority', 'all'],
                                                      default_value='not minority')

        c_space.add_hyperparameters([sampling_strategy])

        return c_space
