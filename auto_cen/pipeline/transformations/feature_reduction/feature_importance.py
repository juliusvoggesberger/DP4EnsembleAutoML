import numpy as np
from ConfigSpace import ConfigurationSpace, UniformFloatHyperparameter


from auto_cen.constants import MIXED,FEATURERED,FEATURE_IMP
from auto_cen.pipeline.base import BaseAlgorithm
from auto_cen.pipeline.transformations.base_preprocessor import BasePreprocessor
from sklearn.ensemble import RandomForestClassifier
import warnings
from scipy.sparse import SparseEfficiencyWarning
warnings.simplefilter('ignore',SparseEfficiencyWarning)

class Feature_Importance_FS(BasePreprocessor):
    """
    Applies estimation of feature Importance for a discrete target variable to the data.

    :param max_param_fraction: Controls how many features are selected.
                               Value range between [0.0,1.0].
                               The fraction is multiplied by the n_features of the dataset to
                               compute the number of features.
    :param seed: Random seed.
    :return: The transformed data.
    """

    def __init__(self,  max_param_fraction: float = 0.7, seed: int = None):
        super().__init__(seed)
        self.max_param = 0
        self.max_param_fraction = max_param_fraction
        self.features_to_keep = []
        self.model = None

    def fit(self, X: np.array, y: np.array) -> BaseAlgorithm:

        n_features = X.shape[1]
        self.max_param = max(1, int(self.max_param_fraction * n_features))
        self.model = RandomForestClassifier(n_estimators=100)
        self.model.fit(X, y)
        sorted_indices = np.argsort(self.model.feature_importances_)[::-1]

        # Get the top `n` indices
        self.features_to_keep = sorted_indices[:self.max_param]

        return self

    def transform(self, X: np.array = None, y: np.array = None) -> np.array:
        X_enc = X[:, self.features_to_keep]
        return X_enc,y

    def get_params(self, deep=True) -> dict:
        return {
            'max_param': self.max_param,
            'max_param_fraction': self.max_param_fraction,
            'selected_features':self.features_to_keep,
            'seed': self.seed,
        }

    @staticmethod
    def get_specification_config() -> dict:
        return {'name':FEATURE_IMP,
                'algorithm':FEATURERED,
                'is_deterministic': True,
                'input': MIXED,
                }

    @staticmethod
    def get_config_space() -> ConfigurationSpace:
        c_space = ConfigurationSpace()
        max_param_fraction = UniformFloatHyperparameter('max_param_fraction', lower=0, upper=1, default_value=0.4)
        c_space.add_hyperparameters([max_param_fraction])

        return c_space
