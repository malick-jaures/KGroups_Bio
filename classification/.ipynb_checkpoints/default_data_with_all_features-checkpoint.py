"""
# 0. Packages
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, RobustScaler, MaxAbsScaler, Normalizer, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import jaccard_score, accuracy_score, r2_score, roc_auc_score, f1_score, balanced_accuracy_score
import sklearn
import seaborn as sns

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

"""# 1. Datasets"""

data_dir = pathjoin('.', 'inputs', 'datasets.mat', 'bio_data')
datasets_list = get_datasets_list_from_dir_and_subdirs(data_dir)
    
if SORT_DATASETS_IN_REVERSED_ORDER:
    datasets_list.reverse() 


USE_STRATIFIED_SHUFFLE_CV = True

df_list = []

"""Experiments start here"""

for mat_fname in tqdm(datasets_list, colour="green"):
    
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
    
    X = pd.DataFrame(data=X, columns=feature_names)

    # print(X.min().min(), X.max().max())

    
    results_dict = cross_validate_models(
        X=X,
        y=y, 
        model_list=clf_list, 
        scoring="accuracy",
        stratified_cv=USE_STRATIFIED_SHUFFLE_CV, 
        # n_splits=n_splits, test_size=test_size, RANDOM_STATE=RANDOM_STATE,
        # return_train_score=return_train_score, return_estimator=False
    )

    df = convert_cross_validate_resultsDict_into_stats_dataframe(
        resultsDict=results_dict, 
        num_desired_features=np.nan, 
        num_selected_features=np.nan, 
        selection_algorithm_name=np.nan,
        valuation_method_name=np.nan,
    ).dropna(axis=1)
    df["dataset_name"] = mat_fname.split("/")[-1].split(".")[0]
    
    df_list += [df]
        
    ## Saving experimental data
    pd.concat(df_list).to_csv("./outputs/dataframes/all_features-all_datasets.csv", index=False)
        
