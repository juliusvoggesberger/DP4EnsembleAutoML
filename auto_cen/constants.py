SUPPORTED_CLASSIFIER = ["RF", "XT", "GB", "AB", "BNB", "DTREE", "GNB", "KNN_CLASS", "LDA", "LSVM",
                        "MLP", "MNB", "PA", "QDA", "SGD", "SVM"]
SUPPORTED_COMBINER = ["MAMV", "WV", "MLE"]

SOLVER = ["BO"]

# Algorithm Type
CLASSIFICATION = "classification"
COMBINATION = "combination"
PREPROCESSOR = "preprocessor"
FEATURERED = "feature_reduction"
PIPELINE = "pipeline"
SAMPLING = "sampling"

# Input Data Types
NUMERICAL = "continuous"
CATEGORICAL = "categorical"
MIXED = "mixed"

# Classification Problem
BINARY = "binary"
MULTICLASS = "multiclass"
MULTILABEL = "multilabel"

# Output Types (+ Input Types for Combiners)
LABELS = "labels"  # If classifier Output: A label vector, If combiner In-/Output: A label matrix
CONTINUOUS_OUT = "continuous_out"

# Evaluation Metrics
ACCURACY = "accuracy"
BALANCED_ACCURACY = "balanced_accuracy"
PRECISION = "precision"
PRECISION_MICRO = "precision_micro"
PRECISION_MACRO = "precision_macro"
RECALL = "recall"
RECALL_MICRO = "recall_micro"
RECALL_MACRO = "recall_macro"
F1_MICRO = "f1_micro"
F1_MACRO = "f1_macro"
JACCARD_MICRO = "jaccard_micro"
JACCARD_MACRO = "jaccard_macro"
AP_MICRO = "apmicro"
AP_MACRO = "apmacro"
MC = "meanconfidence"
ROC_AUC_OVR = "roc_auc_ovr"
ROC_AUC_OVO = "roc_auc_ovo"

# Diversity Metrics
YULES_Q = "yulesq"
YULES_Q_NORM = "yulesq_norm"
CORRELATION_COEFFICIENT = "correlation"
CORRELATION_COEFFICIENT_NORM = "correlation_norm"
DISAGREEMENT = "disagreement"
DOUBLEFAULT = "doublefault"
DOUBLEFAULT_NORM = "doublefault_norm"
KAPPA = "kappa-error"

ENCODER = "ENC"

# Feature Engineering methods
LDAFR = "PRE_LDA"
PCATR = "PCA"
MUTUALINFO = "MUTUAL_INFORMATION"
FEATURE_IMP = 'feature_importance'

# Sampling Methods
ROS = "RANDOM_OVER_SAMPLING"
SMOTE_S = 'smote'
TOMEK_LINKS = 'tomek_links'
ENN = 'edited_nearest_neighbours'

# Combiner types
UTILITY_COMBINER = 'utility'
EVIDENCE_COMBINER = 'evidence'
TRAINABLE_COMBINER = 'trainable'

FILEPATH_MODELS = "files/models/"

SILHOUETTE = "SIL"
