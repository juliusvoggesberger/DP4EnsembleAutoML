"""
Module consisting function for creating the configuration space.
"""
from typing import Union, Tuple

from ConfigSpace import ConfigurationSpace, UniformIntegerHyperparameter, CategoricalHyperparameter, \
    Constant

from auto_cen.constants import MUTUALINFO, LDAFR, PCATR, ROS, SMOTE_S, ENN, TOMEK_LINKS, \
    FEATURE_IMP, ENCODER
from auto_cen.pipeline.transformations.base_preprocessor import BasePreprocessor
from auto_cen.pipeline.transformations.pre_processing.encoding import Encoder


def create_cs(algorithms: list, data_characteristics: dict = None,
              ens_size: Union[int, Tuple] = None, has_cat: bool = False,
              default_value: str = None) -> ConfigurationSpace:
    """
    Creates a configuration space given a list of algorithms using the SMAC config space
    representation.

    :param algorithms: List of tuples containing the algorithm names and the algorithm models.
    :param ens_size: If this parameter is a tuple, this will be used to define the interval
                     in which the ensemble size will be searched.
                     Else if it is an int, the ensemble size will be added as a constant.
                     If None, nothing will be done.
    :param has_cat: True, if any of the features are categorical.
                            Used to decide if an encoder is needed.

    :return: List of algorithms with configuration space.
    """
    c_space = _create_cs(algorithms, default_value)

    # Add preprocessing search spaces (preprocessing, feature engineering, implicit diversity)
    if data_characteristics:
        subspaces = _add_preprocessors(has_cat, data_characteristics)
        for space in subspaces:
            if space[1] is not None:
                c_space.add_configuration_space(*space)

    # Ensemble size as part of the search space
    if isinstance(ens_size, Tuple):
        ens_size = UniformIntegerHyperparameter('ENS:size', lower=ens_size[0], upper=ens_size[1])
        c_space.add_hyperparameter(ens_size)
    # Fixed Ensemble size
    elif ens_size is not None:
        ens_size = Constant('ENS:size', value=ens_size)
        c_space.add_hyperparameter(ens_size)
    return c_space


def _add_preprocessors(has_cat: bool, data_characteristics: dict) -> list:
    """
    Creates subspaces for preprocessing methods.
    Currently only encoding is supported.

    :param has_cat: True, if any of the features are categorical.
                            Used to decide if an encoder is needed.
    :return: List of subspaces.
    """
    enc_cs = [("NOEnc", None)]
    if has_cat:
        enc_cs = [(ENCODER, Encoder)]

    shared_cs = True

    shared_cs_methods = []
    small_data_methods = []
    high_dimensional_methods = []
    class_overlap_os_methods = []

    shared_cs_cs = [("NO", None)]
    small_data = [("NOSD", None)]
    high_dimensional = [("NOHD", None)]
    class_overlap_os = [("NOUS", None)]

    if data_characteristics['small_data']:
        if shared_cs:
            shared_cs_cs = []
            shared_cs_methods += [MUTUALINFO, FEATURE_IMP, ROS, SMOTE_S]
        elif data_characteristics['class_imbalance']:
            small_data = []
            small_data_methods = [MUTUALINFO, FEATURE_IMP]
        else:
            small_data = []
            small_data_methods = [MUTUALINFO, FEATURE_IMP, ROS, SMOTE_S]
    if data_characteristics['high_dimensional_data']:

        if shared_cs:
            shared_cs_cs = []
            shared_cs_methods += [MUTUALINFO, FEATURE_IMP, PCATR, LDAFR]
        elif data_characteristics['small_data']:
            high_dimensional = []
            high_dimensional_methods = [PCATR, LDAFR]
        else:
            high_dimensional = []
            high_dimensional_methods = [MUTUALINFO, FEATURE_IMP, PCATR, LDAFR]
    if data_characteristics['class_imbalance']:

        if shared_cs:
            shared_cs_cs = []
            shared_cs_methods += [ROS, SMOTE_S, ENN, TOMEK_LINKS]
        else:
            class_overlap_os = []
            class_overlap_os_methods = [ROS, SMOTE_S, ENN, TOMEK_LINKS]

    shared_cs_methods = list(set(shared_cs_methods))
    for subclass in BasePreprocessor.__subclasses__():
        sc_spec = subclass.get_specification_config()

        if shared_cs and sc_spec['name'] in shared_cs_methods:
            shared_cs_cs.append((sc_spec['name'], subclass))
        elif sc_spec['name'] in small_data_methods:
            small_data.append((sc_spec['name'], subclass))
        elif sc_spec['name'] in high_dimensional_methods:
            high_dimensional.append((sc_spec['name'], subclass))
        elif sc_spec['name'] in class_overlap_os_methods:
            class_overlap_os.append((sc_spec['name'], subclass))

    print(data_characteristics)
    print(f"Small data: {small_data}")
    print(f"High Dimensional: {high_dimensional}")
    print(f"Oversampling methods: {class_overlap_os}")
    print(f"Shared Methods: {shared_cs_cs}")
    return [("Enc", _create_cs(enc_cs)),("Shared", _create_cs(shared_cs_cs)), ("Ovs", _create_cs(class_overlap_os)),
            ("Fs", _create_cs(small_data)), ("Dr", _create_cs(high_dimensional)),
            ("Us", None)]


def _create_cs(algorithms: list, default_value: str = None) -> ConfigurationSpace:
    """
    Creates a configuration space given a list of algorithms using the SMAC config space
    representation.

    :param algorithms: List of tuples containing the algorithm name and the algorithm model.
    :return: List of algorithms with configuration space.
    """
    if not algorithms:
        return None

    c_space = ConfigurationSpace()
    root = CategoricalHyperparameter('algorithm', [a[0] for a in algorithms],
                                     default_value=default_value)
    c_space.add_hyperparameter(root)
    for algo in algorithms:
        if algo[1] is not None:
            # Needed, if a root node without an algorithm is passed.
            # This is the case, if e.g. it should be possible that no diversity or preprocessing
            # method should be used.
            c_space.add_configuration_space(algo[0], algo[1].get_config_space(),
                                            parent_hyperparameter={'parent': root,
                                                                   'value': algo[0]})
    return c_space
