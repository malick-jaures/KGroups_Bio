"""

"""
#############################################
############### General setup ###############
#############################################
EXPERIMENT_NAME = "FS" # FS => feature selection | FA => feature addition

ALPHA = 0.7

SMOOTHING = True

FA_RATE = 0.05 # Feature Addition Rate in Percentage between 0.0 (0%) and 1 (100%)

RANDOM_STATE = 0

N_JOBS = -1 # -1 means all available CPUs or any integer > 0

N_SPLITS = 5  # Default = 10

TEST_SIZE = 0.3

VERBOSE = True

MIN_FEATURES, MAX_FEATURES, STEP = 2, 100, 1

INCLUDE_LOWEST_OPTIONS = [
    False, 
    # True
]

TRAINING_DATA_SCALER = "Standardizer" # Options {"MinMax", "Robust", "Standardizer", "Normalizer"}

SORT_DATASETS_IN_REVERSED_ORDER = False

QUICK_SEARCH = False # Default = False

RETURN_TRAIN_SCORE = True

IS_VERBOSE_TEST = True 

CLUSTERS_AGG_FUNC_NAME_OPTIONS = [
    'max',
    # 'mean'
] 

FEATURES_COL_NAME='features'

####################################################
############### Classification setup ###############
####################################################

STRATIFIED_CV = False 

SCORING_CLF = ["accuracy"] # "f1_weighted", "accuracy"

ORIGINAL_SCSIG_SETUP_ONLY = True # SCSIG - Supervised Cosine Similarity Information Gain

################################################
############### Regression setup ###############
################################################

USE_STRATIFIED_SHUFFLE_CV_REG = False

SCORING_REG = ["r2"]

####################################################
############### Synthetic data setup ###############
####################################################
# n_samples, n_classes, n_features = X.shape[0], 2, 500
# n_informative, n_redundant = int(0.3*n_features), int(0.05*n_features)
