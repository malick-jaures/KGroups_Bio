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

EXPERIMENT_NAME = "FS_P{}S".format(ALPHA) if SMOOTHING else "FS_P{}".format(ALPHA)
SELECTION_METHOD_NAME = "KGroups_max"
output_path_root = "./outputs/dataframes/{}-{}-{}-{}CV".format(EXPERIMENT_NAME, TRAINING_DATA_SCALER, SCORING_CLF[0], N_SPLITS)  # Path without the final bar 


######################################################

check_or_create_directory(output_path_root)

# For Gini importance-based FFS
default_RF_params , _ = get_model_default_params_and_param_grid(model=RandomForestClassifier, random_state=RANDOM_STATE)

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
    sim_scores_dict = compute_similarity_measures(X=X, y=y,random_state=RANDOM_STATE)
    res_cos = sim_scores_dict["mutual_info_or_cos_sim"] 
    # res = sim_scores_dict["SCSIG"] 
    res_fvalue = sim_scores_dict["Fvalue"] 
    res_bmi = sim_scores_dict["mutual_info_or_cos_sim"] 
    res_var = sim_scores_dict["variance"]
    res_entropy = sim_scores_dict["entropy"]

    dataset_name = mat_fname.split("/")[-1].split(".")[0]
    if VERBOSE:
        print(" ######################## Dataset name: {} -- Shape: {} ######################## ".format(dataset_name, X.shape))
    try:
    
        """## 2.1. Cosine Similarity
        """
        if VERBOSE:
            print(" ------- Cosine Similarity ------- ")
        df_kgroups = run_experiments_using_kgroups(
            scores_df=res_cos, value_column="cos_sim", aggby = None, X=X, y=y, model_list = clf_list, 
            scoring = SCORING_CLF, bins_start=None, bins_stop=None, 
            include_lowest_options=INCLUDE_LOWEST_OPTIONS, 
            break_ties=True, min_features = MIN_FEATURES, max_features = MAX_FEATURES, step=STEP, 
            quick_search=QUICK_SEARCH, return_train_score=RETURN_TRAIN_SCORE,
            clusters_agg_func_name_options=CLUSTERS_AGG_FUNC_NAME_OPTIONS, n_splits=N_SPLITS, test_size=TEST_SIZE,
            # selection_algorithm_name="KGroups",
            stratified_cv=STRATIFIED_CV, alpha=ALPHA,
            valuation_method_name="cosine_sim",  random_state=RANDOM_STATE,
            verbose=VERBOSE, features_col_name=FEATURES_COL_NAME, n_jobs = N_JOBS,
        )
        df_kgroups["dataset_name"] = dataset_name


        ## Saving experimental data
        df_kgroups.to_csv("{}/cosine_sim-{}-{}.csv".format(output_path_root, SELECTION_METHOD_NAME, dataset_name), index=False)

        del df_kgroups; gc.collect()
        
        """
        ---------------------------------------------------------------------------------
        ## 2.3. Mutual Information
        """
        if VERBOSE:
            print(" ------- Mutual Information ------- ")
            
        df_kgroups_MI = run_experiments_using_kgroups(
            scores_df=res_bmi, value_column="mutual_info", aggby = None, X=X, y=y, model_list = clf_list, 
            scoring = SCORING_CLF, bins_start=None, bins_stop=None, 
            include_lowest_options=INCLUDE_LOWEST_OPTIONS, 
            break_ties=True, min_features = MIN_FEATURES, max_features = MAX_FEATURES, step=STEP, 
            quick_search=QUICK_SEARCH, return_train_score=RETURN_TRAIN_SCORE,
            clusters_agg_func_name_options=CLUSTERS_AGG_FUNC_NAME_OPTIONS, n_splits=N_SPLITS, test_size=TEST_SIZE,
            # selection_algorithm_name="KGroups",
            stratified_cv=STRATIFIED_CV, alpha=ALPHA,
            valuation_method_name="mutual_info",  random_state=RANDOM_STATE,
            verbose=VERBOSE, features_col_name=FEATURES_COL_NAME, n_jobs = N_JOBS,
        )
        df_kgroups_MI["dataset_name"] = dataset_name
        
        ## Saving experimental data
        df_kgroups_MI.to_csv("{}/mutual_info-{}-{}.csv".format(output_path_root, SELECTION_METHOD_NAME, dataset_name), index=False)

        del df_kgroups_MI; gc.collect()
        
        """## 
        ---------------------------------------------------------------------------------
        2.4. FValue
        """
        if VERBOSE:
            print(" ------- F-Value ------- ")
            
        df_kgroups_Fv = run_experiments_using_kgroups(
            scores_df=res_fvalue, value_column="fvalue", aggby = None, X=X, y=y, model_list = clf_list, 
            scoring = SCORING_CLF, bins_start=None, bins_stop=None, 
            include_lowest_options=INCLUDE_LOWEST_OPTIONS, 
            break_ties=True, min_features = MIN_FEATURES, max_features = MAX_FEATURES, step=STEP, 
            quick_search=QUICK_SEARCH, return_train_score=RETURN_TRAIN_SCORE,
            clusters_agg_func_name_options=CLUSTERS_AGG_FUNC_NAME_OPTIONS, n_splits=N_SPLITS, test_size=TEST_SIZE,
            # selection_algorithm_name="KGroups",
            stratified_cv=STRATIFIED_CV, alpha=ALPHA,
            valuation_method_name="Fvalue",  random_state=RANDOM_STATE,
            verbose=VERBOSE, features_col_name=FEATURES_COL_NAME, n_jobs = N_JOBS,
        )
        df_kgroups_Fv["dataset_name"] = dataset_name

        
        ## Saving experimental data
        df_kgroups_Fv.to_csv("{}/Fvalue-{}-{}.csv".format(output_path_root, SELECTION_METHOD_NAME, dataset_name), index=False)

        del df_kgroups_Fv;  gc.collect()
        
        """## 
        ---------------------------------------------------------------------------------
        2.5. Variance
        """
        if VERBOSE:
            print(" ------- Variance ------- ")
            
        df_kgroups_VAR = run_experiments_using_kgroups(
            scores_df=res_var, value_column="variance", aggby = None, X=X, y=y, model_list = clf_list, 
            scoring = SCORING_CLF, bins_start=None, bins_stop=None, 
            include_lowest_options=INCLUDE_LOWEST_OPTIONS, 
            break_ties=True, min_features = MIN_FEATURES, max_features = MAX_FEATURES, step=STEP, 
            quick_search=QUICK_SEARCH, return_train_score=RETURN_TRAIN_SCORE,
            clusters_agg_func_name_options=CLUSTERS_AGG_FUNC_NAME_OPTIONS, n_splits=N_SPLITS, test_size=TEST_SIZE,
            # selection_algorithm_name="KGroups",
            stratified_cv=STRATIFIED_CV, alpha=ALPHA,
            valuation_method_name="variance",  random_state=RANDOM_STATE,
            verbose=VERBOSE, features_col_name=FEATURES_COL_NAME, n_jobs = N_JOBS,
        )
        df_kgroups_VAR["dataset_name"] = dataset_name


        ## Saving experimental data
        df_kgroups_VAR.to_csv("{}/variance-{}-{}.csv".format(output_path_root, SELECTION_METHOD_NAME, dataset_name), index=False)

        del df_kgroups_VAR; gc.collect()

        """## 
        ---------------------------------------------------------------------------------
        2.6. Entropy
        """
        if VERBOSE:
            print(" ------- Entropy ------- ")
            
        df_kgroups_Entropy = run_experiments_using_kgroups(
            scores_df=res_entropy, value_column="entropy", aggby = None, X=X, y=y, model_list = clf_list, 
            scoring = SCORING_CLF, bins_start=None, bins_stop=None, 
            include_lowest_options=INCLUDE_LOWEST_OPTIONS, 
            break_ties=True, min_features = MIN_FEATURES, max_features = MAX_FEATURES, step=STEP, 
            quick_search=QUICK_SEARCH, return_train_score=RETURN_TRAIN_SCORE,
            clusters_agg_func_name_options=CLUSTERS_AGG_FUNC_NAME_OPTIONS, n_splits=N_SPLITS, test_size=TEST_SIZE,
            # selection_algorithm_name="KGroups",
            stratified_cv=STRATIFIED_CV, alpha=ALPHA,
            valuation_method_name="entropy",  random_state=RANDOM_STATE,
            verbose=VERBOSE, features_col_name=FEATURES_COL_NAME, n_jobs = N_JOBS,
        )
        df_kgroups_Entropy["dataset_name"] = dataset_name

        ## Saving experimental data
        df_kgroups_Entropy.to_csv("{}/entropy-{}-{}.csv".format(output_path_root, SELECTION_METHOD_NAME, dataset_name), index=False)
        
        del df_kgroups_Entropy; gc.collect()

        """## 
        ---------------------------------------------------------------------------------
        2.7. Random Forest Gini importance
        """
        if VERBOSE:
            print(" ------- Random Forest Gini importance ------- ")
            
        rf_clf = RandomForestClassifier(**default_RF_params)
        rf_clf.fit(X,y)
        res_gini = pd.DataFrame({FEATURES_COL_NAME:rf_clf.feature_names_in_ , "gini_importance":rf_clf.feature_importances_})
        res_gini["mutual_info"] = mutual_info_classif(X, y, random_state=RANDOM_STATE, n_jobs=-1,)
        res_gini.fillna(0., inplace=True)
        
        df_kgroups_RF = run_experiments_using_kgroups(
            scores_df=res_gini, value_column="gini_importance", aggby = None, X=X, y=y, model_list = clf_list, 
            scoring = SCORING_CLF, bins_start=None, bins_stop=None, 
            include_lowest_options=INCLUDE_LOWEST_OPTIONS, 
            break_ties=True, min_features = MIN_FEATURES, max_features = MAX_FEATURES, step=STEP, 
            quick_search=QUICK_SEARCH, return_train_score=RETURN_TRAIN_SCORE,
            clusters_agg_func_name_options=["max"], n_splits=N_SPLITS, test_size=TEST_SIZE,
            # selection_algorithm_name="KGroups",
            stratified_cv=STRATIFIED_CV, alpha=ALPHA,
            valuation_method_name="gini_importance",  random_state=RANDOM_STATE,
            verbose=VERBOSE, features_col_name=FEATURES_COL_NAME, n_jobs = N_JOBS,
        )
        df_kgroups_RF["dataset_name"] = dataset_name

        ## Saving experimental data
        df_kgroups_RF.to_csv("{}/gini-{}-{}.csv".format(output_path_root, SELECTION_METHOD_NAME, dataset_name), index=False)
        
        del df_kgroups_RF; gc.collect()
        
    except  Exception as err:
        print("Errors occurred for dataset: {}\n".format(mat_fname.split("/")[-1]), err )
