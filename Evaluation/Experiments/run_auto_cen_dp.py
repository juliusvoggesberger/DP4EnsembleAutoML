"""
Runs the AutoML Framework
"""
import sys
import logging

from openml import datasets

import auto_cen as ao
from auto_cen.constants import BALANCED_ACCURACY, DOUBLEFAULT, DOUBLEFAULT_NORM, ACCURACY, F1_MACRO, \
    CORRELATION_COEFFICIENT_NORM, DISAGREEMENT, YULES_Q, ROC_AUC_OVO, ROC_AUC_OVR
from auto_cen.optimization.bo import BayesianOptimization

logging.basicConfig(stream=sys.stdout)
logger = logging.getLogger('auto_cen')
logger.setLevel(logging.DEBUG)

if __name__ == '__main__':
    dataset_id = int(sys.argv[1])  # Sets the OpenML ID of the dataset.
    model_budget = int(sys.argv[2])  # Sets the Budget of the Model optimization.
    seed = int(sys.argv[3])  # Sets the random seed.
    dc_flag = ""
    if len(sys.argv) > 4:
        dc_flag = str(
            sys.argv[4])  # If the flag is set to "SD", "HD", "CI" overwrite the standard approach.
    fusion_budget = 20  # Not really necessary as we don't optimize the decision fusion, but has to be set.
    cutoff_time = 600
    cross_valid = 10
    p_metric = BALANCED_ACCURACY
    d_metric = DOUBLEFAULT_NORM
    ens_size = (2, 5, 10, 15, 20)

    dataset = datasets.get_dataset(dataset_id, download_data=True, download_qualities=False,
                                   download_features_meta_data=False)
    X, y, _, _ = dataset.get_data(dataset.default_target_attribute)

    dc_flag = dc_flag if dc_flag in ["SD", "HD", "CI"] else None

    dc = {'small_data': False,
          'high_dimensional_data': False,
          'class_imbalance': False
          }

    class_imbalance_ratio = dataset.qualities["MinorityClassSize"] / dataset.qualities[
        "MajorityClassSize"]

    # Overwrites the data characteristic selection, if the flag is set.
    if dc_flag == "SD":
        dc['small_data'] = True
    elif dc_flag == "HD":
        dc['high_dimensional_data'] = True
    elif dc_flag == "CI":
        dc['class_imbalance'] = True
    else:
        # If the flag is not set, use the standard approach
        if X.shape[0] <= 5000:
            dc['small_data'] = True
        if X.shape[1] >= 100:
            dc['high_dimensional_data'] = True
        if class_imbalance_ratio <= 0.5:
            dc['class_imbalance'] = True

    SAVE_PATH = dc_flag + "_" + str(dataset_id) + "_M" + str(model_budget) + "_CV" + str(
        cross_valid) + "_F" + str(fusion_budget) + "_CO" + str(
        cutoff_time) + "_IDIV_SH_PERF" + p_metric + "_SIZE" + str(ens_size) + "_SEED" + str(seed)

    el = ao.EnsembleLearner(ens_size, model_budget, fusion_budget,
                            cutoff_time=cutoff_time,
                            solver=BayesianOptimization,
                            n_splits=cross_valid,
                            find_ensemble_size=True,
                            data_characteristics=dc,
                            perf_metric=p_metric,
                            div_metric=d_metric,
                            eval_perf_metrics=[BALANCED_ACCURACY, ACCURACY, F1_MACRO, ROC_AUC_OVO,
                                               ROC_AUC_OVR],
                            eval_div_metrics=[DOUBLEFAULT_NORM, DOUBLEFAULT,
                                              CORRELATION_COEFFICIENT_NORM, DISAGREEMENT, YULES_Q],
                            seed=seed)
    el.fit_evaluate(X, y, n_processes=8, save_path=SAVE_PATH, stratify=y,
                    train_size=0.8, valid_size=0.0, test_size=0.2)
