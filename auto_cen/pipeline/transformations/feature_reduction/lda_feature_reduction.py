"""
Module implementing a wrapper for feature reduction using LDA.
"""

from typing import Union

import numpy as np
from ConfigSpace import ConfigurationSpace, CategoricalHyperparameter,  UniformIntegerHyperparameter,UniformFloatHyperparameter,InCondition, EqualsCondition

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from auto_cen.constants import MIXED, LDAFR, FEATURERED
from auto_cen.pipeline.base import BaseAlgorithm
from auto_cen.pipeline.transformations.base_preprocessor import BasePreprocessor

import warnings
from scipy.sparse import SparseEfficiencyWarning
warnings.simplefilter('ignore',SparseEfficiencyWarning)

class LDAFeatureReduction(BasePreprocessor):
    """
    Wrapper for the sklearn feature reduction Linear Discriminant Analysis.

    :param max_component_fraction:  Controls how many dimensions are used.
                               Value range between [0.0,1.0].
                               The fraction is multiplied by the max_components of the dataset to
                               compute the number of dimensions.
    :param solver: Solver used by LDA.
                   Either singular value decomposition (svd), least squares (lsqr) or
                   eigenvalue decomposition (eigen).
    :param tol: Absolute threshold for some data value to be considered significant.
                Value range [1e-5, 1e-1].
                Used in solver svd. More information can be found in the sklearn documentation.
    :param shrinkage: Used for regularization for solver lsqr and eigen.
                      Can either be no shrinkage, manual shrinkage or auto shrinkage.
                      If set to manual, the value of the parameter 'manual' will be used.
    :param manual: Shrinkage value. Value range between [0.0,1.0].
    :param feature_mask: A list of feature indices. Used to select the features for the model.
                        Needed if Random Subspace Method is used.
    """

    def __init__(self, max_component_fraction:float,  solver: str, tol: float, shrinkage: Union[str, float] = None,
                 manual: float = 0,seed: int = None):
        super().__init__(seed)

        self.solver = solver
        self.shrinkage = shrinkage
        self.tol = tol
        self.manual = manual
        self.max_component_fraction = max_component_fraction
        self.n_components= 1

        if self.shrinkage == 'none':
            self.shrinkage = None
        elif self.shrinkage == 'manual':
            self.shrinkage = self.manual
        self.model = None


    def fit(self, X: np.array, y: np.array) -> BaseAlgorithm:
        y_classe = len(np.unique(y))
        max_components = min(y_classe - 1,  X.shape[1])
        self.n_components = max(1, int(max_components * self.max_component_fraction))
        self.model = LinearDiscriminantAnalysis(solver=self.solver, shrinkage=self.shrinkage,n_components=self.n_components,
                                                tol=self.tol)
        self.model.fit(X,y)

        return self

    def transform(self, X: np.array = None, y: np.array = None) -> np.array:
        X_enc = self.model.transform(X)

        return X_enc,y

    def get_params(self, deep=True) -> dict:
        return {
            'solver':self.solver ,
            'shrinkage':self.shrinkage,
            'tol': self.tol,
            'manual': self.manual,
            'max_component_fraction':self.max_component_fraction ,
            'n_components': self.n_components,
            'seed': self.seed,
        }

    @staticmethod
    def get_specification_config() -> dict:
        return {'name': LDAFR,
                'algorithm': FEATURERED,
                'is_deterministic': True,
                'input': MIXED,
                }

    @staticmethod
    def get_config_space() -> ConfigurationSpace:
        c_space = ConfigurationSpace()
        max_component_fraction = UniformFloatHyperparameter('max_component_fraction', lower=0, upper=1)
        solver = CategoricalHyperparameter('solver', ['svd', 'eigen'])
        tol = UniformFloatHyperparameter('tol', lower=1e-5, upper=1e-1, log=True, default_value=1e-4)
        shrinkage = CategoricalHyperparameter('shrinkage', ['none', 'auto', 'manual'], default_value="none")
        manual_shrinkage = UniformFloatHyperparameter('manual', lower=0.0, upper=1.0)

        cond_shrink_solve = InCondition(shrinkage, solver, ['eigen'])
        cond_manual_shrink = EqualsCondition(manual_shrinkage, shrinkage, 'manual')

        c_space.add_hyperparameters([max_component_fraction,solver, tol, shrinkage, manual_shrinkage])
        c_space.add_conditions([cond_shrink_solve, cond_manual_shrink])
        return c_space
