from sklearn.linear_model import LinearRegression, LogisticRegression #, RidgeClassifier, BayesianRidge, Ridge 
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.svm import SVC, LinearSVC, SVR, LinearSVR
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from xgboost import XGBClassifier, XGBRegressor
from sklearn.naive_bayes import GaussianNB

# import pandas as pd
# import numpy as np

########################### List of Models and their Names #################################

####### Classification Models
clf_list = [
    KNeighborsClassifier, 
    # DecisionTreeClassifier, 
    LinearSVC, 
    # LogisticRegression, 
    RandomForestClassifier, 
    XGBClassifier, 
    # SVC, 
    MLPClassifier,  
    GaussianNB,
    # GradientBoostingClassifier,
]
clf_names = [clf_func.__name__.replace("Classifier", "").replace("Regressor", "") for clf_func in clf_list]

####### Regression Models
reg_list = [
    KNeighborsRegressor, 
    # DecisionTreeRegressor,
    LinearSVR, 
    # LinearRegression,   
    RandomForestRegressor, 
    XGBRegressor, 
    # SVR, 
    MLPRegressor, 
    # GradientBoostingRegressor, 
]
reg_names = [reg_func.__name__.replace("Classifier", "").replace("Regressor", "") for reg_func in reg_list]