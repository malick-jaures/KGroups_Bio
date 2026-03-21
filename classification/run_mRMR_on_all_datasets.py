"""
# 0. Packages
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, RobustScaler, MaxAbsScaler, Normalizer, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import jaccard_score, accuracy_score, r2_score, roc_auc_score, f1_score, balanced_accuracy_score
import sklearn

import os
from os.path import dirname, join as pathjoin
import scipy.io as sio

from sklearn.feature_selection import f_classif, f_regression, mutual_info_classif, mutual_info_regression
from sklearn.feature_selection import SelectPercentile, SelectKBest
from glob import glob
from tqdm import tqdm

import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.append("../utils/")
from util_selector import *
from util_preprocessing import *
from util_postprocessing import *
from util_models import *

sys.path.append("../")
from experiment_setup import *

np.random.seed(RANDOM_STATE)
sklearn.random.seed(RANDOM_STATE)

print("Number of CPUs in the system:", os.cpu_count())

"""# 1. Datasets"""

data_dir = pathjoin('.', 'inputs', 'datasets.mat', 'bio_data')
datasets_list = get_datasets_list_from_dir_and_subdirs(data_dir)
    
if SORT_DATASETS_IN_REVERSED_ORDER:
    datasets_list.reverse() 
        
######################################################
## Modify/Upgrade the constants and some variables

SELECTION_METHOD_NAME = "mRMR"
output_path_root = "./outputs/dataframes/{}-{}-{}-{}CV".format(EXPERIMENT_NAME, TRAINING_DATA_SCALER, SCORING_CLF[0], N_SPLITS)  # Path without the final bar 

REGRESSION = False
######################################################

check_or_create_directory(output_path_root)

# For Gini importance-based FFS
# _, default_RF_param_grid = get_RandomForest_default_params_and_param_grid(regression = REGRESSION, random_state=RANDOM_STATE)

"""Experiments start here"""

for mat_fname in tqdm([datasets_list[7]], colour="green"): # Use this to test the experiment on one datasets
# for mat_fname in tqdm(datasets_list, colour="green"): # Use this to run the experiment on all the dataset
    
    X, y = load_matlab_dataset(mat_fname)
    
    X = convert_X_array_into_dataframe(X)
    y = convert_y_array_into_dataframe(y)
    

    """## Some dataset metadata"""
    
    feature_names = list(X.columns) # important
    
    """## Dataset scaling"""
    # See experiment_setup.py file for the details

    scaler = data_scaler(scaler_name=TRAINING_DATA_SCALER)
        
    X = scaler.fit_transform(X)
    X = pd.DataFrame(data=X, columns=feature_names)
    print("Done!")    
    
    """# 2. Finetuning Number of Selected Features
    
    ## 2.0 Setting up some variables
    """

    dataset_name = mat_fname.split("/")[-1].split(".")[0]
    if VERBOSE:
        print(" ######################## Dataset name: {} -- Shape: {} ######################## ".format(dataset_name, X.shape))
    try:
    
        """## 2.1. 'MID', Mutual information, Mutual information, Difference,
        """
        if VERBOSE:
            print(" ------- MID=> Mutual information, Mutual information, Difference ------- ")
        df_mRMR_MID = run_experiments_using_mrmr(
            method="MID", X=X, y=y, model_list = clf_list, scoring = "accuracy", n_splits=N_SPLITS, test_size=TEST_SIZE, 
            min_features = MIN_FEATURES, max_features = MAX_FEATURES, step = STEP,
            stratified_cv=STRATIFIED_CV, return_train_score=RETURN_TRAIN_SCORE, 
            random_state=RANDOM_STATE, quick_search=QUICK_SEARCH, n_jobs=N_JOBS, verbose=VERBOSE,
            regression = REGRESSION, n_neighbors=3, cv_rmrm = 3, 
        )
        df_mRMR_MID["dataset_name"] = dataset_name


        ## Saving experimental data
        df_mRMR_MID.to_csv("{}/MID-{}-{}.csv".format(output_path_root, SELECTION_METHOD_NAME, dataset_name), index=False)

        del df_mRMR_MID; gc.collect()
        
        """
        ---------------------------------------------------------------------------------
        ## 2.2. 'MIQ', Mutual information, Mutual information, Ratio,
        """
        if VERBOSE:
            print(" ------- MIQ=> Mutual information, Mutual information, Ratio ------- ")
            
        df_mRMR_MIQ = run_experiments_using_mrmr(
            method="MIQ", X=X, y=y, model_list = clf_list, scoring = "accuracy", n_splits=N_SPLITS, test_size=TEST_SIZE, 
            min_features = MIN_FEATURES, max_features = MAX_FEATURES, step = STEP,
            stratified_cv=STRATIFIED_CV, return_train_score=RETURN_TRAIN_SCORE, 
            random_state=RANDOM_STATE, quick_search=QUICK_SEARCH, n_jobs=N_JOBS, verbose=VERBOSE,
            regression = REGRESSION, n_neighbors=3, cv_rmrm = 3, 
        )
        df_mRMR_MIQ["dataset_name"] = dataset_name
        
        ## Saving experimental data
        df_mRMR_MIQ.to_csv("{}/MIQ-{}-{}.csv".format(output_path_root, SELECTION_METHOD_NAME, dataset_name), index=False)

        del df_mRMR_MIQ; gc.collect()
        
        """## 
        ---------------------------------------------------------------------------------
        2.3. 'FCD', F-Statistic, Correlation, Difference,
        """
        if VERBOSE:
            print(" ------- FCD=> F-Statistic, Correlation, Difference ------- ")
            
        df_mRMR_FCD = run_experiments_using_mrmr(
            method="FCD", X=X, y=y, model_list = clf_list, scoring = "accuracy", n_splits=N_SPLITS, test_size=TEST_SIZE, 
            min_features = MIN_FEATURES, max_features = MAX_FEATURES, step = STEP, 
            stratified_cv=STRATIFIED_CV, return_train_score=RETURN_TRAIN_SCORE, 
            random_state=RANDOM_STATE, quick_search=QUICK_SEARCH, n_jobs=N_JOBS, verbose=VERBOSE,
            regression = REGRESSION, n_neighbors=3, cv_rmrm = 3, 
        )
        df_mRMR_FCD["dataset_name"] = dataset_name

        
        ## Saving experimental data
        df_mRMR_FCD.to_csv("{}/FCD-{}-{}.csv".format(output_path_root, SELECTION_METHOD_NAME, dataset_name), index=False)

        del df_mRMR_FCD;  gc.collect()
        
        """## 
        ---------------------------------------------------------------------------------
        2.4. 'FCQ', F-Statistic, Correlation, Ratio,
        """
        if VERBOSE:
            print(" ------- FCQ=> F-Statistic, Correlation, Ratio ------- ")
            
        df_mRMR_FCQ = run_experiments_using_mrmr(
            method="FCQ", X=X, y=y, model_list = clf_list, scoring = "accuracy", n_splits=N_SPLITS, test_size=TEST_SIZE, 
            min_features = MIN_FEATURES, max_features = MAX_FEATURES, step = STEP,
            stratified_cv=STRATIFIED_CV, return_train_score=RETURN_TRAIN_SCORE, 
            random_state=RANDOM_STATE, quick_search=QUICK_SEARCH, n_jobs=N_JOBS, verbose=VERBOSE,
            regression = REGRESSION, n_neighbors=3, cv_rmrm = 3,  
        )
        df_mRMR_FCQ["dataset_name"] = dataset_name


        ## Saving experimental data
        df_mRMR_FCQ.to_csv("{}/FCQ-{}-{}.csv".format(output_path_root, SELECTION_METHOD_NAME, dataset_name), index=False)

        del df_mRMR_FCQ; gc.collect()

        """## 
        ---------------------------------------------------------------------------------
        2.5. 'RFCQ', Random Forests, Correlation, Ratio,
        """
        if VERBOSE:
            print(" ------- RFCQ=> Random Forests, Correlation, Ratio ------- ")
        
        df_mRMR_FCQ = run_experiments_using_mrmr(
            method="RFCQ", X=X, y=y, model_list = clf_list, scoring = "accuracy", n_splits=N_SPLITS, test_size=TEST_SIZE, 
            min_features = MIN_FEATURES, max_features = MAX_FEATURES, step = STEP, 
            stratified_cv=STRATIFIED_CV, return_train_score=RETURN_TRAIN_SCORE, 
            random_state=RANDOM_STATE, quick_search=QUICK_SEARCH, n_jobs=N_JOBS, verbose=VERBOSE,
            regression = REGRESSION, n_neighbors=3, cv_rmrm = 3, selector_model = RandomForestClassifier
        )
        df_mRMR_FCQ["dataset_name"] = dataset_name

        ## Saving experimental data
        df_mRMR_FCQ.to_csv("{}/RFCQ-{}-{}.csv".format(output_path_root, SELECTION_METHOD_NAME, dataset_name), index=False)
        
        del df_mRMR_FCQ; gc.collect()
        
    except  Exception as err:
        print("Errors occurred for dataset: {}\n".format(mat_fname.split("/")[-1]), err )
