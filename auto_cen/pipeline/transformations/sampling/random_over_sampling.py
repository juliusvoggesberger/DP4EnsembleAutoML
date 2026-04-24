import numpy as np
from ConfigSpace import ConfigurationSpace, CategoricalHyperparameter, UniformIntegerHyperparameter
from imblearn.over_sampling import RandomOverSampler
from collections import Counter
from auto_cen.constants import MIXED, SAMPLING, ROS
from auto_cen.pipeline.base import BaseAlgorithm
from auto_cen.pipeline.transformations.base_preprocessor import BasePreprocessor


class Random_Over_Sampling(BasePreprocessor):
    """
    Applies Random Oversampling to the data. Used for small datasets or data sets with
    class overlap with class imbalance

    :param sampling_strategy: Specifies the class for which data instances are to be generated.
                             ['minority', 'not majority', 'all']
    :param random_state: Set initiale point for the algorithm.
    :param seed: Random seed.
    :return: The transformed data.
    """

    def __init__(self, sampling_strategy: str, random_state: int = 123, seed: int = None):
        super().__init__(seed)
        self.random_state = random_state
        self.sampling_strategy = sampling_strategy
        self.class_distribution_start = {}
        self.class_distribution_transformed = {}
        self.model = None

    def fit(self, X: np.array, y: np.array) -> BaseAlgorithm:
        self.model = RandomOverSampler(sampling_strategy=self.sampling_strategy, random_state=self.random_state)
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
            'random_state': self.random_state,
            'class_distribution_start': self.class_distribution_start,
            'class_distribution_transformed': self.class_distribution_transformed,
            'seed': self.seed,
        }

    @staticmethod
    def get_specification_config() -> dict:
        return {'name': ROS,
                'algorithm': SAMPLING,
                'is_deterministic': False,
                'input': MIXED,
                }

    @staticmethod
    def get_config_space() -> ConfigurationSpace:
        c_space = ConfigurationSpace()
        sampling_strategy = CategoricalHyperparameter('sampling_strategy', ['minority', 'not majority', 'all'],
                                                      default_value='minority')
        random_state = UniformIntegerHyperparameter('random_state', lower=1, upper=400,
                                            default_value=123)
        c_space.add_hyperparameters([sampling_strategy, random_state])

        return c_space
