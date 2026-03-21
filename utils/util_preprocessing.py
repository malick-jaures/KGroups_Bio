import pandas as pd
import numpy as np
import sys
import os
import time

from sklearn.preprocessing import OneHotEncoder
from glob import glob
import gc 

from sklearn.preprocessing import MinMaxScaler, RobustScaler, MaxAbsScaler, Normalizer, StandardScaler
from sklearn.feature_selection import f_classif, f_regression, mutual_info_classif, mutual_info_regression

import os
from os.path import dirname, join as pathjoin
import scipy.io as sio

import sys
sys.path.append("./")
from util_selector import *
# from util_preprocessing import *
# from util_postprocessing import *
# from util_models import *

# sys.path.append("../")
# from experiment_setup import *
########################### Some Variables #################################

########################### Some Functions #################################

# def format_time(elapsed):
#     from datetime import timedelta
#     '''
#     Takes a time in seconds and returns a string hh:mm:ss.ms (ms stands for milliseconds)
#     '''
#     # Round to the nearest second.
#     # elapsed_rounded = int(round((elapsed)))
#     elapsed_rounded = round((elapsed), 6)
#     # Format as hh:mm:ss
#     return str(timedelta(seconds=elapsed_rounded))

def feature_ratio_to_step(num_features, ratio=0.05):
    if ratio < 0. or ratio > 1.:
        raise ValueError("rate should be between [0., 1.]")
    return max(round(num_features*ratio), 1)

def get_experiment_name(name):
    if name.strip().lower() in ["fa", "featureaddition", "feature_addition", "feature-addition", "addition"]:
        return "FA"
    elif name.strip().lower() in ["fs", "featureselection", "feature_selection", "feature-selection", "selection"]:
        return "FS"
    else:
        return "FS"

def data_scaler(scaler_name):
    scaler = None
    if scaler_name.lower().strip() == "robust":
        print("Applying Robust scaler...")
        scaler = RobustScaler()
    elif scaler_name.lower().strip() == "standardizer":
        print("Applying Standard scaler...")
        scaler = StandardScaler()
    elif scaler_name.lower().strip() == "minmax":
        print("Applying MinMax scaler...")
        scaler = MinMaxScaler()
    elif scaler_name.lower().strip() == "normalizer":
        print("Applying Normalizer...")
        scaler = Normalizer()
    else:
        print("Default setup - Applying Standard scaler...")
        scaler = StandardScaler()
    return scaler

def check_or_create_directory(dir_path:str="./outputs"):
    from pathlib import Path
    # creating a new directory
    path = Path(dir_path)
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)
        print("Created directory: {}".format(path))
    else:
        print("Directory already exists: {}".format(path))

def get_datasets_list_from_dir_and_subdirs(data_dir, extension=".mat"):

    datasets_list = glob(pathjoin(data_dir,"*","*"+extension)) + glob(pathjoin(data_dir,"*"+extension))
    datasets_list.sort() 
        
    # datasets_with_negative_labels = []
    # for filename in glob(pathjoin(data_dir,"*","*.mat")):
    #     contents = sio.loadmat(filename)
    #     X, y = contents['X'], contents['Y']
    #     y = y.reshape(1,-1)[0]
    #     # print("------- {} -------".format(filename.split("/")[-1]))
    #     # print(X.shape,X.min().min(), X.max().max())
    #     labels_index = list(pd.Series(y).value_counts(dropna=False).index)
    #     # print(labels_index)
    #     if -1 in labels_index:
    #         datasets_with_negative_labels+=[filename.split("/")[-1]]

    return  datasets_list #, datasets_with_negative_labels

def load_matlab_dataset(filename):
    mat_contents = sio.loadmat(filename)
    
    X, y = mat_contents['X'], mat_contents['Y']
    y = y.reshape(1,-1)[0]
    y = pd.Series(y)
    labels_index = pd.Series(y).unique()
    if -1 in labels_index:
        y = np.where(y==-1,0,y)
        labels_index = pd.Series(y).unique()
    if 0 not in labels_index:
        y = y - 1
    # else:
    return X, y
        

def compute_similarity_measures(X, y, random_state=0):
    ## For cosine similarity based feature valuation
    res_cos = cosine_similarity(X=X, y=y, return_df=True)
    res_cos["mutual_info"] = mutual_info_classif(X, y, n_jobs=-1, random_state=random_state)
    res_cos.fillna(0., inplace=True)
    
    ## For cosine similarity and/or mutual info. based feature valuation
    res = res_cos[["features","cos_sim", "mutual_info"]]    
    res["cos_sim"] = np.abs(res["cos_sim"])
    
    # res_impr = res[["features","cos_sim", "mutual_info"]]

    ## For Fvalue based feature valuation
    res_fvalue = pd.DataFrame(
        {"features":X.columns,
         "fvalue":f_classif(X, y)[0]}
    ).fillna(0.0)
    
    res_fvalue["mutual_info"] = res_cos["mutual_info"]

    # ## For mutual info. based feature valuation
    # res_bmi = res_cos[["features","cos_sim", "mutual_info"]]
    
    ## For variance based feature valuation
    res_var = res_cos[
        [
            "features",
            "mutual_info",
            "cos_sim",
            # "mutual_info",
        ]
    ]
    # res_var["fvalue"] = res_fvalue["fvalue"].values
    res_var["variance"] = X.apply(calc_variance).values
    res_var.fillna(0., inplace=True)

    ## For Entropy based feature valuation
    res_entropy = res_cos[
        [
            "features", 
            "mutual_info",
        ]
    ]
    res_entropy["entropy"] = X.apply(calc_entropy).values
    res_entropy.fillna(0., inplace=True)

    return {
        "mutual_info_or_cos_sim": res_cos,
        "Fvalue": res_fvalue,
        "SCSIG": res, # SCSIG - Supervised Cosine Similarity Information Gain
        "variance": res_var,
        "entropy": res_entropy,
    }

def compute_similarity_measures_as_mrmr(X, y, random_state=0):

    ## For Fvalue based feature valuation
    res_fvalue = pd.DataFrame(
        {"features":X.columns,
         "fvalue":f_classif(X, y)[0]}
    ).fillna(0.0)

    df = kendalltau_X_y(X, y, return_df=True)
    df["corr_spear"] = spearmanr_X_y(X, y, return_df=False).values
    
    res_fvalue["corr"] = df[["corr_spear","corr"]].mean(axis=1).values
    del df; gc.collect()
    # res_fvalue["mutual_info"] = res_cos["mutual_info"]  ## should compute pearson corr

    ## For cosine similarity based feature valuation
    res_bmi = res_fvalue[["features", "corr"]]
    res_bmi["mutual_info"] = mutual_info_classif(X, y, n_jobs=-1, random_state=random_state)
    res_bmi.fillna(0., inplace=True)

    return {
        "mutual_info_or_cos_sim": res_bmi,
        "Fvalue": res_fvalue,
    }
    
########################### 

def convert_X_array_into_dataframe(X_data):
    if isinstance(X_data, np.ndarray):
        return pd.DataFrame(data=X_data, columns=["col_{}".format(i) for i in range(X_data.shape[1])])
    elif isinstance(X_data, pd.core.frame.DataFrame):
        return X_data
    else:
        raise Exception("Input dependent data should be a numpy array or pandas DataFrame")

def convert_y_array_into_dataframe(y_data):
    if isinstance(y_data, np.ndarray):
        if len(y_data.shape) == 1:
            return pd.Series(y_data, name="target")
        elif len(y_data.shape) == 2:
            return pd.DataFrame(data=y_data, columns=["target_{}".format(i) for i in range(y_data.shape[1])])
    elif isinstance(y_data, pd.core.series.Series):
        return y_data
    else:
        raise Exception("Input dependent data should be a numpy array, or pandas Series or DataFrame")


###########################
def check_mem():
    # These are the usual ipython objects, including this one you are creating
    import sys
    ipython_vars = ['In', 'Out', 'exit', 'quit', 'get_ipython', 'ipython_vars']

    # Get a sorted list of the objects and their sizes
    print(sorted([(x, sys.getsizeof(globals().get(x))) for x in globals() if not x.startswith('_') and x not in sys.modules and x not in ipython_vars], key=lambda x: x[1], reverse=True))

########################### Load data from public data repositories #################################

def load_data_from_opeml(data_id, dataset_format='dataframe'):
  """
  Credit to: https://gist.github.com/pod3275/6d22a2b7bf6e628791ff892eef3e80c5
  
  * Check on https://www.openml.org/ to get dataset IDs
  * Possible values: {'dataframe', 'array}. Default: 'dataframe'
  """
  import openml
  # print("Data loading...")
  dataset = openml.datasets.get_dataset(data_id)

  X, y, categorical_indicator, attribute_names = dataset.get_data(
      dataset_format=dataset_format,
      target=dataset.default_target_attribute
  )

  # print("Data load complete.")
  if dataset_format=='dataframe':
    return (X, y)
  else:
    return (X, y, categorical_indicator, attribute_names)

def load_data_from_uci(data_id, return_metadata=False):
    """
    Credit to: => https://stackoverflow.com/questions/66945361/how-to-get-the-data-from-uci-machine-learning-repository
               => https://github.com/uci-ml-repo/ucimlrepo
    """
    from ucimlrepo import fetch_ucirepo 
      
    # data_with_metadata is just a pyhton dict
    data_with_metadata = fetch_ucirepo(id=data_id) 
    
    X = data_with_metadata.data.features 
    y = data_with_metadata.data.targets 
    return (X, y, data_with_metadata) if return_metadata else (X, y)

def load_data_from_csv_files(data_id, data_name="openMLdatasetID_{}_X.csv", target_name="openMLdatasetID_{}_y.csv"):
    _X = pd.read_csv("./inputs/datasets.csv/"+data_name.format(data_id))
    _y = pd.read_csv("./inputs/datasets.csv/"+target_name.format(data_id))
    return _X , _y.iloc[:,0] if _y.shape[1]==1 else _y

def save_data_to_csv_files(data_id, data, target, data_name="openMLdatasetID_{}_X.csv", target_name="openMLdatasetID_{}_y.csv"):
    data.to_csv("./inputs/datasets.csv/"+data_name.format(data_id), index=False)
    target.to_csv("./inputs/datasets.csv/"+target_name.format(data_id), index=False)
    print(data_name.split(".")[0][:-2].format(data_id)+" has been successfully saved locally in a CSV format!")
########################### Preprocessiong #################################

def reduce_dataframe_size(df):
    def reduce_series_size(series):
        if "int" in str(series.dtype):
            series = pd.to_numeric(series, errors='coerce', downcast="integer")
        elif "float" in str(series.dtype):
            series = pd.to_numeric(series, errors='coerce', downcast="float")
        else:
            series = series.astype("int64")
            series = pd.to_numeric(series, errors='coerce', downcast="integer")
        return series
        
    if isinstance(df, pd.core.frame.DataFrame):
        for col in df.columns:
            df[col] = reduce_series_size(df[col])
        return df
    elif isinstance(df, pd.core.series.Series):
        return reduce_series_size(df)
    else:
        raise TypeError("Object type not supported")


def check_for_missing_values(df, threshold:np.float16=.0):
    def check_series_missing_values(series, threshold:np.float16=.0):
        missing_type_list = [] 
        try:
            v = (series.value_counts(dropna=False)*100./series.shape[0]).loc["?"]
            if v > threshold:
                missing_type_list += ["?"] 
        except:
            pass
        try:
            v = (series.value_counts(dropna=False)*100./series.shape[0]).loc[np.nan]
            if v > threshold:
                missing_type_list += ["NaN"] 
        except:
            pass
        return missing_type_list
        
    missing_data_cols = {
        "?": [],
        "NaN": []
    }
    if isinstance(df, pd.core.frame.DataFrame):
        for col in df.columns:
            missing_type_list = check_series_missing_values(df[col], threshold=threshold)
            if "?" in missing_type_list:
                missing_data_cols["?"] += [col]
            if "NaN" in missing_type_list:
                missing_data_cols["NaN"] += [col]
        return missing_data_cols 
    elif isinstance(df, pd.core.series.Series):
        return check_series_missing_values(df, threshold=threshold)
    else:
        raise TypeError("Object type not supported")


def convert_sparse_to_dense(X,y):
    """
    Convert Sparse DF to Dense DF. See => https://pandas.pydata.org/docs/user_guide/sparse.html
    """
    print("Before changes=>\nAny Sparse: {}\nAll Sparse: {}".format(X.dtypes.apply(pd.api.types.is_sparse).any(), X.dtypes.apply(pd.api.types.is_sparse).all()))
    
    if X.dtypes.apply(pd.api.types.is_sparse).any():
        X = X.sparse.to_dense()
        y = y.sparse.to_dense()
    
        print("After changes=>\nAny Sparse: {}".format(X.dtypes.apply(pd.api.types.is_sparse).any()))

    return X, y

def drop_single_value_columns(df):
    cols_to_drop = []
    for col in df.columns:
        if df[col].nunique() == 1:
            cols_to_drop += [col]
    df.drop(cols_to_drop, axis=1, inplace=True)
    return df

def try_converting_category_and_string_cols_to_numeric(df):
    col_list = []
    for col in df.select_dtypes(include=[object, "category"]).columns:
        try:
             df[col] = pd.to_numeric( df[col], errors='raise', downcast="float")
        except:
            col_list += [col]
    return df, col_list
    
def one_hot_encoding_categories(df):
    if isinstance(df, pd.core.frame.DataFrame):
        cols_encoders_dict = dict()
        _encoded_df_list = []
        _cols_to_delete = []
        _encoded_df = None
        
        df, categorical_cols_list = try_converting_category_and_string_cols_to_numeric(df)
        
        for colname in categorical_cols_list:
            _cols_to_delete += [colname] 
            one_hot_enc = OneHotEncoder(sparse_output=False, handle_unknown='ignore', dtype=np.uint8)
            one_hot_enc.fit(df[colname].values.reshape(-1,1))
            
            _encoded_df = pd.DataFrame(
                data= one_hot_enc.transform(df[colname].values.reshape(-1,1)),
                columns=[colname+"_"+val for val in one_hot_enc.categories_[0]],
            ) 
            _encoded_df_list += [_encoded_df]
            cols_encoders_dict[colname] = one_hot_enc
    
        df.drop(_cols_to_delete, axis=1, inplace=True)
        del (_encoded_df, _cols_to_delete); gc.collect()
        
        return  pd.concat([df]+_encoded_df_list, axis=1) , cols_encoders_dict
    elif isinstance(df, pd.core.series.Series):
        one_hot_enc = OneHotEncoder(sparse_output=False, handle_unknown='ignore', dtype=np.uint8)
        one_hot_enc.fit(df.values.reshape(-1,1))
        return pd.DataFrame(
            data= one_hot_enc.transform(df.values.reshape(-1,1)),
            columns=one_hot_enc.categories_,
        ) , one_hot_enc 

########################### Load data from public data repositories => preprocessing => saving locally #################################

def data_cleansing_and_local_storage(data_id_list, loading_from="openml", max_tol_missing_rate:np.float16=.0):
    X, y = None, None
    data_name, target_name = None, None
    
    if loading_from.strip().lower() == "openml":
        data_name, target_name = "openMLdatasetID_{}_X.csv", "openMLdatasetID_{}_y.csv"
    elif loading_from.strip().lower() == "uci":
        data_name, target_name = "UCIdatasetID_{}_X.csv", "UCIdatasetID_{}_y.csv"
        
    for data_id in data_id_list:
        if loading_from.strip().lower() == "openml":
            X, y = load_data_from_opeml(data_id, dataset_format='dataframe')
        elif loading_from.strip().lower() == "uci":
            X, y, _ = load_data_from_uci(uci_id, return_metadata=True)
        print(X.shape, y.shape)
        # print(X.info(), y.shape))

        ## Handling duplicated rows
        dupl_rows = X.duplicated()
        X, y = X[~dupl_rows].reset_index(drop=True, allow_duplicates=False), y[~dupl_rows].reset_index(drop=True, allow_duplicates=False)

        ## Check for columns with unique equal to num_rows
        series = X.nunique()
        print("This dataset has {} column(s) with number of unique values equal to num_rows".format(len(series[(series == X.shape[0])].index)))

        ## Checking for missing values in X and y
        missing_data_cols = check_for_missing_values(X, threshold=max_tol_missing_rate)
        print("? => {}\nNaN => {}".format(len(missing_data_cols["?"]),len(missing_data_cols["NaN"])))
        
        missing_data_targets = check_for_missing_values(y, threshold=max_tol_missing_rate)
        if isinstance(missing_data_targets, dict):
            print("? => {}\nNaN => {}".format(len(missing_data_targets["?"]),len(missing_data_targets["NaN"])))
        else:
            print(missing_data_targets)
        
        ## Handling missing values in X
        if len(missing_data_cols["NaN"]) > 0:
            X.drop(missing_data_cols["NaN"], axis=1, inplace=True)
        
        if len(missing_data_cols["?"])>0:
            X.replace("?", np.nan, inplace=True)
            X.drop(missing_data_cols["?"], axis=1, inplace=True)
        
        ## Handling missing values in y
        if isinstance(missing_data_targets, dict):
            if len(missing_data_targets["NaN"]) > 0:
                y.drop(missing_data_targets["NaN"], axis=1, inplace=True)
            
            if len(missing_data_targets["?"])>0:
                y.replace("?", np.nan, inplace=True)
                y.drop(missing_data_targets["?"], axis=1, inplace=True)
        
        # Check for sparse dtypes and convert them into dense dtypes     
        X,y = convert_sparse_to_dense(X,y)

        X = drop_single_value_columns(X)
        
        # X.select_dtypes(include=[object, "category"])

        X, _ = one_hot_encoding_categories(X)
        # y, _ = one_hot_encoding_categories(y) # If y also need to be encoded

        print("Overview of the dataset before saving")
        print(X.info())
        gc.collect()
        save_data_to_csv_files(
            data_id, X, y, 
            data_name = data_name, 
            target_name =target_name,
        )
        print("\n")