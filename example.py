"""
Runs the AutoML Framework
"""

import sys
import logging

from openml import datasets

from auto_cen.constants import ACCURACY, F1_MACRO, BALANCED_ACCURACY, DOUBLEFAULT, ROC_AUC_OVO, \
    DISAGREEMENT, DOUBLEFAULT_NORM
import auto_cen as ac
from auto_cen.optimization.bo import BayesianOptimization

# Activate Logging
logging.basicConfig(stream=sys.stdout)
logger = logging.getLogger('auto_cen')
logger.setLevel(logging.DEBUG)

if __name__ == '__main__':
    # Set the input parameters
    dataset_id = 30
    budget_m = 100
    budget_f = 20
    n_splits = 10
    cutoff_time = 600
    perf_metric = BALANCED_ACCURACY
    div_metric = DOUBLEFAULT_NORM
    ensemble_size = (2, 5, 10, 15, 20)
    seed = 123

    # Load dataset from OpenML
    dataset = datasets.get_dataset(dataset_id, download_data=True, download_qualities=False,
                                   download_features_meta_data=False)
    X, y, _, _ = dataset.get_data(dataset.default_target_attribute)

    # Check for Data Characteristics
    dc = {'small_data': False,
          'high_dimensional_data': False,
          'class_imbalance': False
          }

    class_imbalance_ratio = dataset.qualities["MinorityClassSize"] / dataset.qualities[
        "MajorityClassSize"]

    if X.shape[0] <= 5000:
        dc['small_data'] = True
    if X.shape[1] >= 100:
        dc['high_dimensional_data'] = True
    if class_imbalance_ratio <= 0.5:
        dc['class_imbalance'] = True

    # Set the foldername for the outputs
    SAVE_PATH = str(dataset_id) + "_M" + str(budget_m) + "_CV" + str(n_splits) + "_F" + str(
        budget_f) + "_CO" + str(cutoff_time) + "_IDIV_SH_PERF" + perf_metric + "_SIZE" + str(
        ensemble_size) + "_SEED" + str(seed)

    # Set up Auto-CEn
    el = ac.EnsembleLearner(ensemble_size, budget_m, budget_f,
                            cutoff_time=cutoff_time,
                            solver=BayesianOptimization,
                            n_splits=n_splits,
                            find_ensemble_size=True,
                            data_characteristics=dc,
                            perf_metric=perf_metric,
                            div_metric=div_metric,
                            eval_perf_metrics=[ACCURACY, BALANCED_ACCURACY, F1_MACRO, ROC_AUC_OVO],
                            eval_div_metrics=[DOUBLEFAULT_NORM, DOUBLEFAULT, DISAGREEMENT],
                            seed=seed)

    # Fit and evaluate the ensemble model
    el.fit_evaluate(X, y, n_processes=8, save_path=SAVE_PATH, stratify=y,
                    train_size=0.8, valid_size=0.0, test_size=0.2)
