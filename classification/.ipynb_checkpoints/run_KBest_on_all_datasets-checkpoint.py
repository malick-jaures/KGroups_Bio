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

SELECTION_METHOD_NAME = "KBest" 
output_path_root = "./outputs/dataframes/{}-{}-{}-{}CV".format(EXPERIMENT_NAME, TRAINING_DATA_SCALER, SCORING_CLF[0], N_SPLITS)  # Path without the final bar 
######################################################

check_or_create_directory(output_path_root)

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
    
        if VERBOSE:
            print(" ------- Cosine Similarity ------- ")
            
        df_selectKBest = run_experiments_using_kbest(
            score_func=cosine_similarity, supervised = True, custom_class=True,
            X=X, y=y, model_list=clf_list, scoring = SCORING_CLF, n_splits=N_SPLITS, 
            test_size=TEST_SIZE, min_features= MIN_FEATURES, max_features = MAX_FEATURES, step=STEP, 
            stratified_cv=STRATIFIED_CV, quick_search=QUICK_SEARCH,
            return_train_score=RETURN_TRAIN_SCORE,  
            valuation_method_name="cosine_sim", random_state=RANDOM_STATE, verbose = VERBOSE, n_jobs = N_JOBS,
        )
        df_selectKBest["dataset_name"] = dataset_name
        
        df_selectKBest.to_csv("{}/cosine_sim-{}-{}.csv".format(output_path_root, SELECTION_METHOD_NAME, dataset_name), index=False)

        del df_selectKBest; gc.collect()
        
        """
        ---------------------------------------------------------------------------------
        ## 2.3. Mutual Information
        """
        if VERBOSE:
            print(" ------- Mutual Information ------- ")
            
        df_selectKBest_MI = run_experiments_using_kbest(
            score_func=mutual_info_classif, supervised = True, custom_class=True,
            X=X, y=y, model_list=clf_list, scoring = SCORING_CLF, n_splits=N_SPLITS, 
            test_size=TEST_SIZE, min_features= MIN_FEATURES, max_features = MAX_FEATURES, step=STEP, 
            stratified_cv=STRATIFIED_CV, quick_search=QUICK_SEARCH,
            return_train_score=RETURN_TRAIN_SCORE,  
            valuation_method_name="mutual_info", random_state=RANDOM_STATE, verbose = VERBOSE, n_jobs = N_JOBS,
        )
        df_selectKBest_MI["dataset_name"] = dataset_name
        
        ## Saving experimental data
        df_selectKBest_MI.to_csv("{}/mutual_info-{}-{}.csv".format(output_path_root, SELECTION_METHOD_NAME, dataset_name), index=False)

        del df_selectKBest_MI; gc.collect()
        
        """## 
        ---------------------------------------------------------------------------------
        2.4. FValue
        """
        if VERBOSE:
            print(" ------- F-Value ------- ")
            
        df_selectKBest_Fv = run_experiments_using_kbest(
            score_func=f_classif, supervised = True, custom_class=True,
            X=X, y=y, model_list=clf_list, scoring = SCORING_CLF, n_splits=N_SPLITS, 
            test_size=TEST_SIZE, min_features= MIN_FEATURES, max_features = MAX_FEATURES, step=STEP, 
            stratified_cv=STRATIFIED_CV, quick_search=QUICK_SEARCH,
            return_train_score=RETURN_TRAIN_SCORE,  
            valuation_method_name="Fvalue", random_state=RANDOM_STATE, verbose = VERBOSE, n_jobs = N_JOBS,
        )
        df_selectKBest_Fv["dataset_name"] = dataset_name
        
        ## Saving experimental data
        df_selectKBest_Fv.to_csv("{}/Fvalue-{}-{}.csv".format(output_path_root, SELECTION_METHOD_NAME, dataset_name), index=False)

        del df_selectKBest_Fv;  gc.collect()
        
        """## 
        ---------------------------------------------------------------------------------
        2.3. Variance
        """
        if VERBOSE:
            print(" ------- Variance ------- ")
            
        df_selectKBest_VAR = run_experiments_using_kbest(
            score_func=calc_variance, supervised = False, custom_class=True,
            X=X, y=y, model_list=clf_list, scoring = SCORING_CLF, n_splits=N_SPLITS, 
            test_size=TEST_SIZE, min_features= MIN_FEATURES, max_features = MAX_FEATURES, step=STEP, 
            stratified_cv=STRATIFIED_CV, quick_search=QUICK_SEARCH,
            return_train_score=RETURN_TRAIN_SCORE,  
            valuation_method_name="variance", random_state=RANDOM_STATE, verbose = VERBOSE, n_jobs = N_JOBS,
        )
        df_selectKBest_VAR["dataset_name"] = dataset_name

        ## Saving experimental data
        df_selectKBest_VAR.to_csv("{}/variance-{}-{}.csv".format(output_path_root, SELECTION_METHOD_NAME, dataset_name), index=False)

        del df_selectKBest_VAR; gc.collect()

        """## 
        ---------------------------------------------------------------------------------
        2.4. Entropy
        ### a) KGroups
        """
        if VERBOSE:
            print(" ------- Entropy ------- ")
            
        df_selectKBest_Entropy = run_experiments_using_kbest(
            score_func=calc_entropy, supervised = False, custom_class=True,
            X=X, y=y, model_list=clf_list, scoring = SCORING_CLF, n_splits=N_SPLITS, 
            test_size=TEST_SIZE, min_features= MIN_FEATURES, max_features = MAX_FEATURES, step=STEP, 
            stratified_cv=STRATIFIED_CV, quick_search=QUICK_SEARCH,
            return_train_score=RETURN_TRAIN_SCORE,  
            valuation_method_name="entropy", random_state=RANDOM_STATE, verbose = VERBOSE,n_jobs = N_JOBS,
        )
        df_selectKBest_Entropy["dataset_name"] = dataset_name

        ## Saving experimental data
        df_selectKBest_Entropy.to_csv("{}/entropy-{}-{}.csv".format(output_path_root, SELECTION_METHOD_NAME, dataset_name), index=False)
        
        del df_selectKBest_Entropy; gc.collect()

        """## 
        ---------------------------------------------------------------------------------
        2.7. Random Forest Gini importance
        """
        if VERBOSE:
            print(" ------- Random Forest Gini importance ------- ")
            
        df_selectKBest_RF = run_experiments_using_kbest(
            score_func=None, supervised = True, custom_class=False,
            X=X, y=y, model_list=clf_list, scoring = SCORING_CLF, n_splits=N_SPLITS, 
            test_size=TEST_SIZE, min_features= MIN_FEATURES, max_features = MAX_FEATURES, step=STEP, 
            stratified_cv=STRATIFIED_CV, quick_search=QUICK_SEARCH,
            return_train_score=RETURN_TRAIN_SCORE,  
            valuation_method_name="gini_importance", random_state=RANDOM_STATE, n_jobs = N_JOBS,
            from_model=True, selector_model = RandomForestClassifier, verbose = VERBOSE,
        )
        df_selectKBest_RF["dataset_name"] = dataset_name

        ## Saving experimental data
        df_selectKBest_RF.to_csv("{}/gini-{}-{}.csv".format(output_path_root, SELECTION_METHOD_NAME, dataset_name), index=False)
        
        del df_selectKBest_RF; gc.collect()
        
    except  Exception as err:
        print("Errors occurred for dataset: {}\n".format(mat_fname.split("/")[-1]), err )
