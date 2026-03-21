from sklearn.feature_selection import SelectKBest, SelectFromModel #,SelectPercentile
from sklearn.model_selection import train_test_split
# from tqdm import tqdm
# from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from feature_engine.selection import MRMR
from typing import Union

import pandas as pd
import numpy as np
import sklearn
import time

import gc  
# gc.disable()
# gc.isenabled()

import os
import sys
# sys.path.append("./")
# from util_preprocessing import format_time
# from util_postprocessing import *
# from util_models import *

########################### Some Functions #################################
def format_time(elapsed, num_decimal=6):
    from datetime import timedelta
    '''
    Takes a time in seconds and returns a string hh:mm:ss.ms (ms stands for milliseconds)
    '''
    # Round to the nearest second.
    elapsed_rounded = round((elapsed), num_decimal)
    # Format as hh:mm:ss
    return str(timedelta(seconds=elapsed_rounded))

########################### Similarities Measures #################################

### Lambda functions
from scipy.stats import entropy, variation

calc_entropy = lambda x: entropy(x.value_counts()) # options: base={None|2|10}
calc_variation = lambda x: variation(x, nan_policy='omit') 
calc_variance = lambda x: np.nanvar(x, axis=0)

########################### Cosine similarities 
def cosine_similarity(X, y, return_df=False, absolute=False):
    import pandas as pd
    import numpy as np
    from scipy.spatial.distance import cosine as cosine_dist
    
    if not isinstance(X, pd.core.frame.DataFrame):
        X = pd.DataFrame(data=X, columns=["col_{}".format(i) for i in range(X.shape[1])])
        
    sim_scores = [1-cosine_dist(X[col].values,y.values) for col in X.columns]
    if absolute:
        sim_scores = np.abs(sim_scores)

    if return_df:
        return pd.DataFrame({"features":X.columns, "cos_sim":sim_scores}).fillna(0.0) 
    else:
        return pd.Series(data=sim_scores, index=X.columns)

def kendalltau_X_y(X, y, return_df=False, absolute=False):
    import pandas as pd
    import numpy as np
    import scipy.stats
    if not isinstance(X, pd.core.frame.DataFrame):
        X = pd.DataFrame(data=X, columns=["col_{}".format(i) for i in range(X.shape[1])])
    # scores = [scipy.stats.pointbiserialr(X[col].values,y.values)[0] for col in X.columns]
    scores = [scipy.stats.kendalltau(X[col].values,y.values)[0] for col in X.columns]
    if absolute:
        scores = np.abs(scores)
    if return_df:
        return pd.DataFrame({"features":X.columns, "corr":scores}).fillna(0.0) 
    else:
        return pd.Series(data=scores, index=X.columns)

def spearmanr_X_y(X, y, return_df=False, absolute=False):
    import pandas as pd
    import numpy as np
    import scipy.stats
    if not isinstance(X, pd.core.frame.DataFrame):
        X = pd.DataFrame(data=X, columns=["col_{}".format(i) for i in range(X.shape[1])])
    scores = [scipy.stats.spearmanr(X[col].values,y.values)[0] for col in X.columns]
    if absolute:
        scores = np.abs(scores)
    if return_df:
        return pd.DataFrame({"features":X.columns, "corr":scores}).fillna(0.0) 
    else:
        return pd.Series(data=scores, index=X.columns)

########################### CustomFeatureSelectors #################################

from sklearn.base import BaseEstimator
from sklearn.feature_selection import SelectorMixin
# from functools import partial

class CustomSelectKBest(SelectorMixin, BaseEstimator):
    def __init__(self, score_func, k=10):
        self.score_func = score_func
        self.feature_scores_ = None
        self.features_sorted_by_scores_ = None
        self.feature_pvalues_ = None
        self.feature_scores_sorted_ = None
        self.feature_names_in_ = None
        self.feature_names_in_sorted_ = None
        self.feature_names_out_ = None
        if k is None or k < 1:
            raise ValueError("k should be greater than or equal to 1")
        self.k = k
        self.X = None
        
    def fit(self, X, y=None, **kwargs):
        self.X = X.copy()
        self.feature_names_in_ = [i for i in range(X.shape[1])] if not isinstance(X, pd.core.frame.DataFrame) else X.columns
        
        self.n_features_in_ = X.shape[1]
        _score_func_output = self.score_func(X, **kwargs) if y is None else self.score_func(X, y, **kwargs)
        if isinstance(_score_func_output, tuple):
            self.feature_scores_ = _score_func_output[0]
            self.feature_pvalues_ = _score_func_output[1]
        else:
            self.feature_scores_ = _score_func_output
        self.features_sorted_by_scores_ = list(
            pd.Series(self.feature_scores_, index=self.feature_names_in_).sort_values(ascending=False).items()
        )
        self.feature_scores_sorted_ = [item[1] for item in self.features_sorted_by_scores_]
        self.feature_names_in_sorted_ = [item[0] for item in self.features_sorted_by_scores_] ## Sorted names by score values 

        self.feature_names_out_=  self.feature_names_in_sorted_[:self.k]
        del _score_func_output; gc.collect()
        return self
        
    def transform(self, X, k=None):
        _k = self.k if k is None else k
        self.feature_names_out_=  self.feature_names_in_sorted_[:_k]
        return X[self.feature_names_out_]
        
    def fit_transform(self, X, y=None,k=None, **kwargs):
        self.fit(X=X, y=y)
        return self.transform(X=X, k=self.k if k is None else k)
        
    def inverse_transform(self):
        return self.X
        
    def _get_support_mask(self):
        return pd.Series(self.feature_names_in_).isin(self.feature_names_out_).values
        
    def get_feature_names_out(self):
        return self.feature_names_out_
    
########################### Model training functions #################################

def auto_set_models_default_params(model_list, n_jobs=-1, random_state=10,):
    default_params_dict = dict()
    for model in model_list:
        model_name = model.__name__.replace("Classifier", "").replace("Regressor", "")  
        model_params = model().get_params().keys()
        if 'n_jobs' in model_params:
            default_params_dict[model_name] = {'n_jobs':n_jobs}
        if 'random_state' in model_params:
            if model_name in default_params_dict.keys():
                default_params_dict[model_name]['random_state'] = random_state
            else:
                default_params_dict[model_name] = {'random_state':random_state}
    del (model_name, model_params); gc.collect()
    return default_params_dict

# from sklearn.metrics import make_scorer
def cross_validate_models(
    X, y, model_list:list=None, scoring: Union[list, dict]=None, n_splits=10, test_size=0.3, random_state=0,
    stratified_cv=True, return_train_score=True, return_estimator=False, n_jobs=-1,
):
    """
    scoring=> list or dict of strings / scorer(score_func, kwargs)
    """
    from sklearn.model_selection import cross_validate, StratifiedShuffleSplit
    import numpy as np
    import sklearn
    np.random.seed(random_state)
    sklearn.random.seed(random_state)

    models_default_params = auto_set_models_default_params(model_list, n_jobs=n_jobs, random_state=random_state,)
    
    scores_dict = dict()
    for _model in model_list:
        model_name = _model.__name__.replace("Classifier", "").replace("Regressor", "")
        model_params = models_default_params[model_name] if model_name in models_default_params.keys() else dict()
        model = _model(**model_params)
        if stratified_cv:
            cv = StratifiedShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=random_state)
            scores = cross_validate(
                model, X=X, y=y, scoring=scoring, cv=cv, n_jobs=n_jobs if "n_jobs" in model_params.keys() else 1, 
                #pre_dispatch = '2*n_jobs',
                return_train_score=return_train_score, return_estimator=return_estimator,
            )
        else:
            scores = cross_validate(
                model, X=X, y=y, scoring=scoring, cv=n_splits, n_jobs=n_jobs if "n_jobs" in model_params.keys() else 1, 
                #pre_dispatch = '2*n_jobs',
                return_train_score=return_train_score, return_estimator=return_estimator,
            )
        scores_dict[model_name] = pd.DataFrame(scores)
    return scores_dict
 
###########################   Postprocessing   #################################

### Results Formatting
def convert_cross_validate_resultsDict_into_stats_dataframe(
    resultsDict:dict=None, num_desired_features=None, num_selected_features=None, 
    selection_algorithm_name:str = None, valuation_method_name:str = None, 
):
    import pandas as pd
    df_list = []
    for k in resultsDict.keys():
        df_out = pd.DataFrame()
    
        _df_mean = pd.DataFrame(
            [(name, round(value,4) if name.endswith("_time") else 100*round(value,4) ) for (name, value) in resultsDict[k].mean(axis=0).items()]
        ).T
        _df_mean.columns = _df_mean.iloc[0,:]
        _df_mean.drop(0, axis=0, inplace=True)
        
        _df_std = pd.DataFrame(
            [(name, round(value,4) if name.endswith("_time") else 100*round(value,4) ) for (name, value) in resultsDict[k].std(axis=0).items()]
        ).T
        _df_std.columns = _df_std.iloc[0,:]
        _df_std.drop(0, axis=0, inplace=True)
        
        _df_std.columns = ["{}_std".format(col) for col in _df_std.columns]
        reordered_cols = []
        
        for col1, col2 in zip(_df_mean.columns, _df_std.columns):
            reordered_cols += [col1, col2]

        df_out = pd.concat([_df_mean, _df_std], axis=1)[reordered_cols]
        
        df_out["model_name"] = k
        df_out["num_desired_features"] = num_desired_features
        df_out["num_selected_features"] = num_selected_features
        df_out["selection_algorithm"] = selection_algorithm_name
        df_out["valuation_method"] = valuation_method_name
        
        df_list += [df_out]

    return pd.concat(df_list).reset_index(drop=True)
    
########################### Value-based Features clustering   #################################

def nonlinear_binning(x, n_bins=10, method="log", strength=1.0):
    """
    Create non-linear bin edges with smaller bins for small values
    and larger bins for large values.

    Parameters
    ----------
    x : array-like
        Input data (1D).
    n_bins : int
        Number of bins.
    method : str, optional
        Transformation method: 'log' or 'power'.
    strength : float, optional
        Controls non-linearity:
        - For 'log': higher -> stronger compression of large values.
        - For 'power': <1 makes bins finer at small values.

    Returns
    -------
    bins : ndarray
        Bin edges (length n_bins+1).
    """
    import numpy as np

    if strength <= 0.:
        raise ValueError("strength must be a strictly positive float number!")
    
    x = np.asarray(x)
    x_min, x_max = x.min(), x.max()

    if method == "log":
        # Ensure shift so values are positive before log
        shift = 1 - x_min if x_min <= 0 else 0
        x_shifted = x + shift

        # Transform to log-space
        log_min, log_max = np.log1p(x_shifted.min()), np.log1p(x_shifted.max())

        # Uniform bins in [log_min, log_max], apply strength
        u = np.linspace(0, 1, n_bins + 1) ** strength
        
        bins_transformed = log_min + u * (log_max - log_min) ## added to fix range issue

        # Map back, remove shift
        bins = np.expm1(bins_transformed) - shift

        # Force exact match of data range
        bins[0], bins[-1] = x_min, x_max ## added to fix range issue

    elif method == "power":
        # Normalize to [0,1], apply power warp, map back
        u = np.linspace(0, 1, n_bins + 1) ** strength
        bins = u * (x_max - x_min) + x_min

    else:
        raise ValueError("method must be 'log' or 'power'")

    return bins


def create_bins_with_steps(start, stop, step):
    """
    Credit to: https://stackoverflow.com/questions/53212834/numpy-arange-include-endpoint
    
    """
    import numpy as np
    L = stop-start
    n = int(L/step)
    stop_ = start+n*step
    return np.linspace(start,stop_,n+1)

def create_bins(start:float, stop:float, n_bins:int, alpha:float=1.0, smoothing:bool=False) -> np.ndarray:
    if alpha <= 0. or alpha > 5.:
        raise ValueError("alpha must be a strictly positive float number less than or equal to 5; preferably in (0.0; 3.]!")
    elif alpha==1.:
        return np.linspace(start,stop,n_bins+1, endpoint=True)
    else:
        # Map to [0,1] using norm_x = (x - x_min) / (x_max - x_min), then apply power transformation
        # norm_x = (x - x_min) / (x_max - x_min)
        if smoothing:
            if alpha<1.:
                bins_transformed = np.linspace(0, 1, n_bins + 1, endpoint=True) ** (alpha**(1./3.))
                bins = bins_transformed * (stop - start) + start
                return bins
            elif alpha>1.:
                bins_transformed = np.linspace(0, 1, n_bins + 1, endpoint=True) ** (alpha**(4./5.))
                bins = bins_transformed * (stop - start) + start
                return bins
        else:
            bins_transformed = np.linspace(0, 1, n_bins + 1, endpoint=True) ** alpha
            bins = bins_transformed * (stop - start) + start
            return bins


## Which is better point_change (pt_change) or percentage_change (pct_change)?
def handle_inf_values(df, keep_inf_values_seperate=False, pct_change=0.0):
    df.fillna(0.0, inplace=True)
    
    series = np.isinf(df.select_dtypes(include=np.number)).sum()
    
    for col in series[series>0].index:
        max_val = df[col][~np.isinf(df[col])].max()
        min_val = df[col][~np.isinf(df[col])].min()

        if keep_inf_values_seperate:
            df[col] = np.where(np.isposinf(df[col]), max_val*(1+pct_change), df[col])
            df[col] = np.where(np.isneginf(df[col]), min_val*(1-pct_change), df[col])
        else:
            df[col] = np.where(np.isposinf(df[col]), max_val, df[col])
            df[col] = np.where(np.isneginf(df[col]), min_val, df[col])
    return df

def cluster_features(df, value_column:str=None, n_clusters=20, bins=None, include_lowest=False, 
                     clusters_col_suffix="_clusters", clusters_label_prefix="ft_cluster_", 
                     include_infs=False, pct_change=0.0, keep_infs_seperate=False, #handle_infs = False,
                     verbose = False, alpha:float=1.0, smoothing:bool=False,
                    ):
    import numpy as np
    import pandas as pd

    max_val, min_val = None, None
    if bins is None:
        if np.isinf(df[value_column]).sum() > 0 :
            max_val, min_val = df[value_column][~np.isinf(df[value_column])].max(), df[value_column][~np.isinf(df[value_column])].min()
            bins = create_bins(min_val, max_val, n_clusters, alpha=alpha, smoothing=smoothing)
            if include_infs:
                if np.isposinf(df[value_column]).sum() > 0 :
                    bins = np.array(list(bins) + [max_val+(bins[-1]-bins[-2])], dtype=np.float64)
                    df[value_column] = np.where(np.isposinf(df[value_column]), max_val+(bins[-1]-bins[-2]), df[value_column])
                    
                if np.isneginf(df[value_column]).sum() > 0 :
                    bins = np.array([min_val-(bins[1]-bins[0])]+ list(bins), dtype=np.float64)
                    df[value_column] = np.where(np.isneginf(df[value_column]), min_val-(bins[1]-bins[0]), df[value_column])
            else:
                df[value_column] = np.where(np.isposinf(df[value_column]), max_val, df[value_column])
                df[value_column] = np.where(np.isneginf(df[value_column]), min_val, df[value_column])
        else:
            bins = create_bins(df[value_column].min(), df[value_column].max(), n_clusters, alpha=alpha, smoothing=smoothing)

    bins = np.unique(bins) # Remove possible duplicates from bins
    labels=None
    if len(bins) > 1:
        if include_lowest: # and np.isneginf(df[value_column]).sum() == 0:
            bins = np.array([bins[0]-(bins[1]-bins[0])]+ list(bins), dtype=np.float64)
            labels= ["{}{}".format(clusters_label_prefix, i) for i in range(bins.shape[0]-1)]
        else:
            labels= ["{}{}".format(clusters_label_prefix, i+1) for i in range(bins.shape[0]-1)]
            
        try: 
            df[value_column+clusters_col_suffix] = pd.cut(
                df[value_column], bins=bins, 
                right=True, labels=labels,
            )
        except Exception as err:
            if verbose:
                print("Exception {} occurred:\n".format(type(err)), err)
            else:
                pass
    else:
        df[value_column+clusters_col_suffix] = "{}{}".format(clusters_label_prefix, 1)
        
    df[value_column+clusters_col_suffix] = df[value_column+clusters_col_suffix].fillna(labels[0])
    return df, bins, labels

def get_features_from_clusters(df, groupby:str=None, aggby:str=None, features_col_name="features", break_ties= False, tie_breaker_columns:list = None, verbose = False,):
    df_out = df.groupby(groupby).agg({aggby:max}).dropna().reset_index().merge(df, how="inner")
    
    if (tie_breaker_columns is not None) and len(tie_breaker_columns) > 0 and break_ties and (df_out.shape[0] != df_out[aggby].nunique()):
        if verbose:
            print("\nApplying the tie breaker(s) ...")
        ft_cluster_dict = df_out.groupby(groupby).agg({features_col_name:"unique"}).to_dict()[features_col_name]
        
        selected_features = []
        
        used_ties_breaker = []
        for ft_cluster in ft_cluster_dict.keys():
            if len(ft_cluster_dict[ft_cluster]) > 0:
                if len(ft_cluster_dict[ft_cluster]) == 1:
                    selected_features += list(ft_cluster_dict[ft_cluster])
                else:
                    _df = df_out[df_out[features_col_name].isin(ft_cluster_dict[ft_cluster])]
                    for col in tie_breaker_columns:
                        used_ties_breaker += [col]
                        _df = _df[_df[col] == _df[col].max()]
                        if _df.shape[0] == 1:
                            break
                    selected_features += list(_df[features_col_name].values)
        if verbose:        
            print("{}/{} tie breaker(s) have/has been applied!".format(len(set(used_ties_breaker)), len(tie_breaker_columns)))
        
        return selected_features
    else:
        return df_out[features_col_name].values
        

def agg_features_from_clusters(X_data, clusters_df, clusters_col_name:str=None, features_col_name:str="features"):
    import pandas as pd
    import numpy as np
    ft_cluster_dict = clusters_df.groupby(clusters_col_name).agg({features_col_name:"unique"}).to_dict()[features_col_name]
    
    df_new = pd.DataFrame()
    for ft_cluster in ft_cluster_dict.keys():
        if len(ft_cluster_dict[ft_cluster]) > 0:
            df_new[ft_cluster] = np.mean(X_data[ft_cluster_dict[ft_cluster]], axis=1)
            
    return df_new.dropna(axis=1)


###########################   Finetuning k - Experimental data generation   #################################

###########################   KGroups 

def _range_gen(num_features, min_features:int = 2, max_features:int = 100, step:int = 1, quick_search=False,):
    _start = step if min_features==0 or min_features is None else min_features
    _end = num_features+1 if max_features is None else min(max_features, num_features)+1
    _range = None
    if step == 1 and quick_search:
        _range = set(np.arange(5, _end, 5)).union(set(np.arange(4, _end, 4))).union(set(np.arange(3, _end, 3))).union({2})
    else:
        _range = np.arange(_start,_end , step)
    return _range

def finetune_k_using_kgroups(
    scores_df, value_column:str = None,
    X=None, y=None, model_list: list = None, scoring: Union[list, dict]  = ["f1_weighted"], n_splits=10, test_size=0.3, 
    bins_start=None, bins_stop=None,  aggby: str = None, min_features:int = 2, max_features:int = 100, step:int = 1,
    include_lowest=False, features_col_name='features', break_ties=False, 
    clusters_col_suffix:str="_clusters", clusters_label_prefix:str="ft_cluster_",
    stratified_cv=True, return_train_score=True, 
    clusters_agg_func_name:str='mean', valuation_method_name="cosine_sim",
    random_state=0, quick_search=False, n_jobs=-1, alpha:float=1.0, smoothing:bool=False,
):
    stats_df_list = []

    for n_clusters in _range_gen(X.shape[1], min_features, max_features, step, quick_search):
        _features = None
        _results_dict = None
        _stats_df = pd.DataFrame()

        try:
            ## Group features
            _scores_df, _, _ = cluster_features(
                df=scores_df, 
                value_column=value_column, 
                n_clusters=n_clusters, 
                bins = None if bins_start is None or bins_stop is None else create_bins(bins_start, bins_stop, n_clusters, alpha=alpha, smoothing=smoothing) ,
                include_lowest=include_lowest,
                clusters_col_suffix=clusters_col_suffix,
                clusters_label_prefix=clusters_label_prefix,
                alpha=alpha, smoothing=smoothing,
            )
    
            if clusters_agg_func_name.strip().lower()=='max':
                _groupby = "{}{}".format(value_column, clusters_col_suffix)
                _aggby = value_column if aggby is None else aggby
                
                _features = get_features_from_clusters(
                    _scores_df, groupby=_groupby,aggby=_aggby, 
                    tie_breaker_columns = list(_scores_df.drop([features_col_name,_groupby, _aggby], axis=1).columns),
                    break_ties= break_ties,
                )
                
                _results_dict = cross_validate_models(
                    X=X[_features],
                    y=y, 
                    model_list=model_list, 
                    scoring=scoring,
                    stratified_cv=stratified_cv, 
                    n_splits=n_splits, test_size=test_size, random_state=random_state,
                    return_train_score=return_train_score, return_estimator=False,
                    n_jobs=n_jobs,
                )
    
                _stats_df = convert_cross_validate_resultsDict_into_stats_dataframe(
                    resultsDict=_results_dict, 
                    num_desired_features=n_clusters, 
                    num_selected_features=len(_features), 
                    selection_algorithm_name= "KGroups_{}".format(clusters_agg_func_name),
                    valuation_method_name=valuation_method_name,
                )
                
            elif clusters_agg_func_name.strip().lower() =='mean':
                X_new = agg_features_from_clusters(
                    X_data=X, 
                    clusters_df=_scores_df, 
                    clusters_col_name="{}{}".format(value_column, clusters_col_suffix), 
                    features_col_name=features_col_name
                )
                
                _results_dict = cross_validate_models(
                    X=X_new,
                    y=y, 
                    model_list=model_list, 
                    scoring=scoring,
                    stratified_cv=stratified_cv, 
                    n_splits=n_splits, test_size=test_size, random_state=random_state,
                    return_train_score=return_train_score, return_estimator=False,
                    n_jobs=n_jobs,
                )
                
                _stats_df = convert_cross_validate_resultsDict_into_stats_dataframe(
                    resultsDict=_results_dict, 
                    num_desired_features=n_clusters, 
                    num_selected_features=X_new.shape[1], 
                    selection_algorithm_name= "KGroups_{}".format(clusters_agg_func_name),
                    valuation_method_name=valuation_method_name,
                )
            else:
                pass
        
            stats_df_list += [_stats_df]
        except Exception as err:
            print("Exception {} occurred:\n".format(type(err)), err)

    del (_results_dict,_stats_df); gc.collect()
    if len(stats_df_list)>0:
        return pd.concat(stats_df_list).reset_index(drop=True)
    else:
        return pd.DataFrame()

def run_experiments(func, verbose=True, **kwargs):
    """
    Func doc goes here!
    """
    start_time = time.time()
    start_process_time = time.process_time()
    df = func(**kwargs)
    end_process_time = time.process_time()
    end_time = time.time()
    
    df["CPU_time"] = format_time((end_process_time-start_process_time))
    df["Wall_time"] = format_time((end_time-start_time))

    if verbose:
        print("Execution times:")
        print("  => CPU_time:",format_time((end_process_time-start_process_time)))
        print("  => Wall_time:",format_time((end_time-start_time)))
        print()
        
    del (end_process_time,start_process_time,end_time,start_time); gc.collect()
    return df
    
## To be deleted
def run_experiments_using_kgroups(
    scores_df, value_column:str = None,
    X=None, y=None, model_list: list = None, scoring: Union[list, dict]  = ["f1_weighted"], n_splits=10, test_size=0.3, 
    bins_start=None, bins_stop=None,  aggby: str = None, min_features:int = 2, max_features:int = 100, step:int = 1,
    features_col_name='features', break_ties=False, 
    clusters_col_suffix:str="_clusters", clusters_label_prefix:str="ft_cluster_",
    stratified_cv=True, return_train_score=True, 
    valuation_method_name="cosine_sim", verbose=True,
    random_state=0, quick_search=False, n_jobs=-1, alpha:float=1., smoothing:bool=False,
    include_lowest_options:list=[False, True], clusters_agg_func_name_options:list=['max','mean'],
):
    df_list = []
    df = pd.DataFrame()
    
    for clusters_agg_func_name in clusters_agg_func_name_options:
        for include_lowest in include_lowest_options:
            start_time = time.time()
            start_process_time = time.process_time()
            df = finetune_k_using_kgroups(
                scores_df=scores_df, value_column = value_column,
                X=X, y=y, model_list = model_list, scoring = scoring, n_splits=n_splits, test_size=test_size, 
                bins_start=bins_start, bins_stop=bins_stop,  aggby = aggby, min_features= min_features, max_features = max_features, step = step,
                include_lowest=include_lowest, features_col_name=features_col_name, break_ties=break_ties, 
                clusters_col_suffix=clusters_col_suffix, clusters_label_prefix=clusters_label_prefix,
                stratified_cv=stratified_cv, return_train_score=return_train_score, 
                clusters_agg_func_name=clusters_agg_func_name, valuation_method_name=valuation_method_name,
                random_state=random_state, quick_search=quick_search, n_jobs=n_jobs, alpha=alpha, smoothing=smoothing
            )
            end_process_time = time.process_time()
            end_time = time.time()
            
            df["include_lowest"] = include_lowest
            
            df["CPU_time"] = format_time((end_process_time-start_process_time))
            df["Wall_time"] = format_time((end_time-start_time))
        
            df_list += [df]
            if verbose:
                print("KGroups_{} execution times (include_lowest = {}):".format(clusters_agg_func_name, "True" if include_lowest else "False"))
                print("  => CPU_time:",format_time((end_process_time-start_process_time)))
                print("  => Wall_time:",format_time((end_time-start_time)))
                print()
        
    df_out = pd.concat(df_list, ignore_index=True, axis=0)
    del (df_list,df,end_process_time,start_process_time,end_time,start_time); gc.collect()
    return df_out



###########################   KBest 

def finetune_k_using_kbest(
    score_func, X=None, y=None, model_list: list = None, scoring: Union[list, dict]  = ["f1_weighted"], n_splits=10, test_size=0.3, 
    min_features:int = 2, max_features:int = 100, step:int = 1, stratified_cv=True, return_train_score=True,
    valuation_method_name="cosine_sim", random_state=0, supervised=True, custom_class=False,
    quick_search=False, n_jobs=-1, from_model:bool=False, selector_model = None,
):
    stats_df_list = []
        
    selector = None
    if score_func is not None and custom_class:
        if supervised:
            selector = CustomSelectKBest(score_func=score_func).fit(X, y)
        else:
            selector = CustomSelectKBest(score_func=score_func).fit(X)
    for n_clusters in _range_gen(X.shape[1], min_features, max_features, step, quick_search):
        _results_dict = None
        _stats_df = pd.DataFrame()

        if score_func is not None and not custom_class:
            selector = SelectKBest(score_func=score_func, k=n_clusters ).fit(X, y)
        if score_func is None and from_model:
            if selector_model is None:
                raise TypeError("selector_model should be a Classifier/Regressor class not NoneType!")
            _model = selector_model(
                **get_model_default_params_and_param_grid(model=selector_model, random_state=random_state)[0]
            )
            _model.fit(X,y)
            
            selector = SelectFromModel(
                _model, max_features=n_clusters, 
                threshold=-1*np.inf, prefit=True, norm_order=1, importance_getter='auto',
            ).fit(X,y)
        
        _results_dict = cross_validate_models(
            X=selector.transform(X, k=n_clusters) if custom_class else selector.transform(X), 
            y=y, 
            model_list=model_list, 
            scoring=scoring,
            stratified_cv=stratified_cv, 
            n_splits=n_splits, test_size=test_size, random_state=random_state,
            return_train_score=return_train_score, return_estimator=False,
            n_jobs=n_jobs,
        )

        _stats_df = convert_cross_validate_resultsDict_into_stats_dataframe(
            resultsDict=_results_dict, 
            num_desired_features=n_clusters, 
            num_selected_features=n_clusters, 
            selection_algorithm_name="KBest",
            valuation_method_name=valuation_method_name,
        )
            
        stats_df_list += [_stats_df]
        
    del (_results_dict,_stats_df); gc.collect()
    return pd.concat(stats_df_list).reset_index(drop=True)

## To be deleted
def run_experiments_using_kbest(
    score_func, X=None, y=None, model_list: list = None, scoring: Union[list, dict]  = ["f1_weighted"], n_splits=10, test_size=0.3, 
    min_features:int = 2, max_features:int = 100, step:int = 1, stratified_cv=True, return_train_score=True, 
    valuation_method_name="cosine_sim", random_state=0, supervised=True, custom_class=False,
    quick_search=False, verbose = False,n_jobs=-1, 
    from_model:bool=False, selector_model = None,
):
    """
    Func doc goes here!
    """
    start_time = time.time()
    start_process_time = time.process_time()
    df = finetune_k_using_kbest(
        score_func=score_func, X=X, y=y, model_list = model_list, scoring = scoring, n_splits=n_splits, 
        test_size=test_size, min_features = min_features, max_features = max_features, step = step, 
        stratified_cv=stratified_cv, return_train_score=return_train_score,
        valuation_method_name=valuation_method_name, random_state=random_state,
        supervised=supervised, custom_class=custom_class, 
        quick_search=quick_search, n_jobs=n_jobs, from_model=from_model, selector_model=selector_model
    )
    end_process_time = time.process_time()
    end_time = time.time()
    
    df["CPU_time"] = format_time((end_process_time-start_process_time))
    df["Wall_time"] = format_time((end_time-start_time))

    if verbose:
        print("KBest execution times:")
        print("  => CPU_time:",format_time((end_process_time-start_process_time)))
        print("  => Wall_time:",format_time((end_time-start_time)))
        print()
        
    del (end_process_time,start_process_time,end_time,start_time); gc.collect()
    return df


############################# mRMR

def get_model_default_params_and_param_grid(model, random_state=0):
    model_name = model.__name__.replace("Classifier", "").replace("Regressor", "")
    
    default_params = model().get_params()
    default_params.update(
        **auto_set_models_default_params(
            [model], 
            random_state=random_state, 
        )[model_name]
    )
    
    default_param_grid = dict()
    for k in default_params.keys():
        default_param_grid[k] = [default_params[k]]

    return default_params, default_param_grid
    
def finetune_k_using_mrmr(
    method:str, X=None, y=None, model_list: list = None, scoring: str = "accuracy", n_splits=5, test_size=0.3, 
    min_features:int = 2, max_features:int = 100, step:int = 1, stratified_cv=False, return_train_score=True, 
    random_state=0, quick_search=False, n_jobs=-1, 
    regression:bool = False, n_neighbors:int=3, cv_rmrm:int = 3, selector_model = None,
):
    stats_df_list = []
    

    for n_clusters in _range_gen(X.shape[1], min_features, max_features, step, quick_search):
        _results_dict = None
        _stats_df = pd.DataFrame()

        ## mRMR
        default_RF_param_grid = None
        if method == "RFCQ":
            _ , default_RF_param_grid = get_model_default_params_and_param_grid(model=selector_model, random_state=random_state,)
    
        selector = MRMR(
            method = method,
            max_features = int(n_clusters),
            n_neighbors=n_neighbors,
            scoring = scoring,
            cv=cv_rmrm,
            regression = regression,
            param_grid = default_RF_param_grid if method == "RFCQ" else None,
            random_state = random_state,
            confirm_variables = False,
            n_jobs = n_jobs,
        )

        selector.fit(X,y)
        
        _results_dict = cross_validate_models(
            X=selector.transform(X), 
            y=y, 
            model_list=model_list, 
            scoring=[scoring],
            stratified_cv=stratified_cv, 
            n_splits=n_splits, test_size=test_size, random_state=random_state,
            return_train_score=return_train_score, return_estimator=False,
            n_jobs=n_jobs,
        )

        _stats_df = convert_cross_validate_resultsDict_into_stats_dataframe(
            resultsDict=_results_dict, 
            num_desired_features=n_clusters, 
            num_selected_features=n_clusters, 
            selection_algorithm_name="mRMR",
            valuation_method_name=method,
        )
            
        stats_df_list += [_stats_df]
        
    del (_results_dict,_stats_df); gc.collect()
    return pd.concat(stats_df_list).reset_index(drop=True)

## To be deleted
def run_experiments_using_mrmr(
    method:str, X=None, y=None, model_list: list = None, scoring: str = "accuracy", n_splits=5, test_size=0.3, 
    min_features:int = 2, max_features:int = 100, step:int = 1, stratified_cv=False, return_train_score=True, 
    random_state=0, quick_search=False, n_jobs=-1, verbose:bool=True,
    regression:bool = False, n_neighbors:int=3, cv_rmrm:int = 3, selector_model = None,
):
    """
    Func doc goes here!
    """
    start_time = time.time()
    start_process_time = time.process_time()
    df = finetune_k_using_mrmr(
        method=method, X=X, y=y, model_list = model_list, scoring = scoring, n_splits=n_splits, test_size=test_size, 
        min_features = min_features, max_features = max_features, step = step, stratified_cv=stratified_cv, 
        return_train_score=return_train_score, 
        random_state=random_state, quick_search=quick_search, n_jobs=n_jobs, 
        regression = regression, n_neighbors=n_neighbors, cv_rmrm = cv_rmrm, selector_model=selector_model
    )
    end_process_time = time.process_time()
    end_time = time.time()
    
    df["CPU_time"] = format_time((end_process_time-start_process_time))
    df["Wall_time"] = format_time((end_time-start_time))

    if verbose:
        print("mRMR_{} execution times:".format(method))
        print("  => CPU_time:",format_time((end_process_time-start_process_time)))
        print("  => Wall_time:",format_time((end_time-start_time)))
        print()

    del (end_process_time,start_process_time,end_time,start_time); gc.collect()
    return df