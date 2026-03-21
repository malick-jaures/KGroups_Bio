## Import libraries
import pandas as pd #; pd.set_option("display.max_columns",None); pd.set_option("display.max_rows",None)
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import gc
import time

import warnings
warnings.filterwarnings("ignore")

from pandas.api.types import CategoricalDtype

filename = None
out_path_root = "./outputs/"
in_path_root = "./inputs/"

def load_data(with_cols=True, filename=filename):
    if with_cols:
        return pd.read_csv(filename, 
                   on_bad_lines="warn", 
                  )
    else:
        ### Loop the data lines
        with open(filename, 'r') as temp_f:
            # get No of columns in each line
            col_count = [ len(l.split(",")) for l in temp_f.readlines() ]

        ### Generate column names  (names will be 0, 1, 2, ..., maximum columns - 1)
        column_names = [i for i in range(0, max(col_count))]
            
        return pd.read_csv(filename, 
                   skiprows=1,
                   header=None, 
                   delimiter=",", 
                   names=column_names,
                  ), col_count
    
def load_text_data(num_cols=None, filename=filename):
    with open(filename, 'r') as temp_f:
        if num_cols is None:
            return load_data(with_cols=False)[0]
        else:
            ### Loop the data lines
            with open(filename, 'r') as temp_f:
                splitted_data = [ line.split(",") for line in temp_f.readlines() if len(line.split(",")) <=num_cols ]
            return pd.DataFrame(splitted_data[1:]), splitted_data[0]
            

def convert_columns_to_categorical(df, columns, ordered=False):
    for col in columns:
        df[col] = df[col].astype(CategoricalDtype(ordered=ordered))
    return df


def find_empty_cells(df):
    cols_with_empty_cells = []
    for col in df.columns:
        if df[df[col] == ''][col].shape[0] > 0:
            cols_with_empty_cells += [col]
    return cols_with_empty_cells


def drop_outliers(df, colname=None, return_outliers=False, q1=None, q3=None):
    import numpy as np
    import pandas as pd
    
    iqr, filter_cond = None, None
    if type(df) == pd.core.series.Series:
        if q1 is None or q3 is None:
            q1, q3 = np.percentile(df, 25), np.percentile(df, 75)
        iqr = q3-q1    
        filter_cond = (df>=q1-1.5*iqr)&(df<=q3+1.5*iqr)
        if return_outliers:
            return df[~filter_cond].reset_index(drop=True), q1, q3
        else:
            return df[filter_cond].reset_index(drop=True), q1, q3
    elif colname is not None and type(df) == pd.core.frame.DataFrame:
        if q1 is None or q3 is None:
            q1, q3 = np.percentile(df[colname], 25), np.percentile(df[colname], 75)
        iqr = q3-q1
        filter_cond = (df[colname]>=q1-1.5*iqr)&(df[colname]<=q3+1.5*iqr)
        df = df[filter_cond]
        if return_outliers:
            return df[~filter_cond].reset_index(drop=True), q1, q3
        else:
            return df[filter_cond].reset_index(drop=True), q1, q3
    else:
        raise TypeError("Either pass a Series or a DataFrame with the column to use as a filter")

def drop_mul_cols_outliers(df, colnames:"list"=None, drop_all=False, return_outliers=False):
    import numpy as np
    
    if colnames is None:
        raise TypeError("A list of column names is expected")
    elif len(colnames) == 1:
        return drop_outliers(df, colname=colnames[0])
    elif len(colnames) > 1:
        filter_all = True #Use to filter out if the entry is an outlier in at least one of the columns
        filter_any = False #Use to filter out if the entry is an outlier in all the columns
        for col in colnames :
            q1, q3 = np.percentile(df[col], 25), np.percentile(df[col], 75)
            iqr = q3-q1    
            filter_out =  (df[col]>=q1-1.5*iqr)&(df[col]<=q3+1.5*iqr)
            
            filter_all = filter_all & filter_out
            filter_any = filter_any | filter_out
        
        if drop_all:
            if return_outliers:
                return df[~filter_all].reset_index(drop=True)
            else:
                return df[filter_all].reset_index(drop=True)
        else:
            if return_outliers:
                return df[~filter_any].reset_index(drop=True)
            else:
                return df[filter_any].reset_index(drop=True)
    else:
        raise TypeError("Argument(s) given is (are) not supported")
        
        
def describe_df(df):
    try:
        stats = df.describe(include=[np.number])
        print("Numerical columns statistics")
        display(stats)
    except:
        pass
    try:
        stats = df.describe(include=[object])
        print("\nString/Object columns statistics")
        display(stats)
    except:
        pass
    try:
        stats = df.describe(include=["category"])
        print("\nCategorical columns statistics")
        display(stats)
    except:
        pass
    del stats; gc.collect()

    
def plot_normal_vs_outliers(series, q1=None, q3=None, plot_pct=True, y_label="y_label", ft="0.0f"):
    import pandas as pd
    normal, outliers = None, None
    if q1 is None:
        ## Normal people
        normal, q1, q3 = drop_outliers(series, q1, q3)
        normal = normal.describe()
        ## Outliers
        outliers, q1, q3 = drop_outliers(series, return_outliers=True, q1=q1, q3=q3)
        outliers = outliers.describe()
    else:
        ## Normal people
        normal = drop_outliers(series, q1, q3)[0].describe()
        ## Outliers
        outliers = drop_outliers(series, return_outliers=True, q1=q1, q3=q3)[0].describe()
  
    
    ax = sns.barplot(x=["Normal ({:{}}-{:{}})".format(normal.loc["min"], ft, normal.loc["max"], ft), "Outliers ({:{}}-{:{}})".format(outliers.loc["min"], ft, outliers.loc["max"], ft)], 
                y=[normal["count"]*100/series.shape[0], outliers["count"]*100/series.shape[0]] if plot_pct else [normal["count"], outliers["count"]]
               )
    annotate_barplot(ax, xytext = (0, -9), ft=ft)
    ax.set_ylabel("Percentage" if plot_pct else y_label)
    plt.savefig(out_path_root+series.name +"-"+"normal_vs_outliers.png", dpi=300)
    return pd.DataFrame({"normal":normal.values, "outliers":outliers.values}, index=normal.index)

def annotate_barplot(ax, xytext = (0, -9), ft="0.1f", orient="v", ha = 'center', va = 'center', 
                     rotate_text=False, color="red", fontsize=12, fontweight="bold", rotation= 60):
    if rotate_text:
        for r in ax.patches:
            h = r.get_height()
            plt.text(r.get_x() + r.get_width() / 2. +xytext[0], h+xytext[1] , format(h,ft), ha=ha, va=va, 
                 color=color, fontsize=fontsize, fontweight=fontweight, rotation= rotation
            ) 
    else:
        for p in ax.patches:
            ax.annotate(format(p.get_width() if orient=="h" else p.get_height(), f'{ft}'), 
                           (p.get_width() , p.get_y() + p.get_height()/2. ) if orient=="h" else (p.get_x() + p.get_width() / 2., p.get_height()),
                           ha = ha, va = va, 
                           xytext = xytext, 
                           textcoords = 'offset points')
            
#########################################################################################
def convert_empty_arrays_into_NAs(df):  ## This is done is the "remove_NAs_within_sets" function
    func = lambda x: np.nan if isinstance(x, np.ndarray) and x.shape[0]==0 else x
    if isinstance(df, pd.DataFrame) :
        for col in df.columns:
            df[col] = df[col].apply(func)
        return df
    else:
        return df.apply(func)

def contains_array(series):
    for v in series.value_counts().index:
        if isinstance(v, np.ndarray) or isinstance(v, frozenset) or isinstance(v, set) or isinstance(v, list):
            return True
    return False

def remove_NAs_within_sets(df, column_list:list=None, use_loop = False):
    def remove_NAs(x, use_loop = use_loop):
        out = None
        if use_loop:
            out = []
            for v in x:
                if isinstance(v, str):
                    out += [v]
                else:
                    if not np.isnan(v):
                        out += [v]
        else:
            out = set(x) - set([np.nan])
            
        if len(out) == 0:
            return np.nan
        else:
#             return frozenset(out)
            return set(out) if use_loop else out
#             return list(out)

    column_list = df.columns if column_list is None else column_list
    if isinstance(df, pd.DataFrame) :
        for col in column_list:
            t0 = time.time()
#             if df[col].dtype == object and contains_array(df[col]):
            try:
                df[col] = df[col].apply(remove_NAs)
            except:
                pass
            print(f"{col} - Elapsed time: {format_time(time.time() - t0)}\n")
        return df
    else:
        return df.apply(remove_NAs)
    
def create_mapping_dict(series, use_value_counts_order=False):
    if contains_array(series):
        mapping_dict = {0: np.nan} if series.isna().sum() >0 else {}
        value_counts_series = series.value_counts(dropna=True)
        if use_value_counts_order:
            i = value_counts_series.shape[0] if len(mapping_dict)==1 else value_counts_series.shape[0]-1
            for idx in value_counts_series.index:
                mapping_dict[i] = idx
                i-=1
        else:
            i = 1 if len(mapping_dict)==1 else 0
            for idx in value_counts_series.index:
                mapping_dict[i] = idx
                i+=1
        return mapping_dict
    else:
        mapping_dict = { np.nan: 0} if series.isna().sum() >0 else {}
        value_counts_series = series.value_counts(dropna=True)
        if use_value_counts_order:
            i = value_counts_series.shape[0] if len(mapping_dict)==1 else value_counts_series.shape[0]-1
            for idx in value_counts_series.index:
                mapping_dict[idx] = i
                i-=1
        else:
            i = 1 if len(mapping_dict)==1 else 0
            for idx in value_counts_series.index:
                mapping_dict[idx] = i
                i+=1
        return mapping_dict

def apply_mapping_dict(series, use_value_counts_order=False, mapping_dict=None ):
    if mapping_dict is None or len(mapping_dict) == 0:
        mapping_dict = create_mapping_dict(series, use_value_counts_order=use_value_counts_order)
    if contains_array(series):
        mapping_dict_val_list = list(mapping_dict.values()) 
        for idx, val in series.iteritems():
            try:
                series.loc[idx] = mapping_dict_val_list.index(val)
            except:
                print(f"An error occurred for {series.name} at the {idx}th index: {val}")
        return series, mapping_dict
    else:
        return series.map(mapping_dict), mapping_dict
    
    
def replace_sets_with_ids(df, columns:list=[], all_mapping_dicts=None, use_value_counts_order=False):
    all_mapping_dicts = dict() if all_mapping_dicts is None else all_mapping_dicts
    columns = df.columns if len(columns) == 0 else columns
    if isinstance(df, pd.DataFrame) :
        for col in columns:
            t0 = time.time()
#             if isinstance(df[col][0], set):
            if df[col].dtype == object:
                try:
                    df[col], all_mapping_dicts[col]  = apply_mapping_dict(df[col], use_value_counts_order)
                except:
                    pass
            print(f"{col} - Elapsed time: {format_time(time.time() - t0)}\n")
        return df, all_mapping_dicts
    else:
        return apply_mapping_dict(df, use_value_counts_order)
    
def encode_columns(df, columns:'list'=None, use_value_counts_order=False, all_mapping_dicts=None):
    all_mapping_dicts = dict() if all_mapping_dicts is None else all_mapping_dicts
    if isinstance(df, pd.DataFrame) :
        for col in columns:
            try:
                df[col], all_mapping_dicts[col]  = apply_mapping_dict(df[col], use_value_counts_order)
            except:
                pass
        return df, all_mapping_dicts
    else:
        return apply_mapping_dict(df, use_value_counts_order)
    
def add_datetime_features(df, columns:list=[], use_fillna=False, na_value=-1, year_multiplier=2000 ):
    import pandas as pd
    import numpy as np
    from workalendar.europe import Ireland
    irish_holidays = Ireland()
    
    def isweekend(x):
        if np.isnan(x) or x is pd.NaT :
            return np.nan
        elif x==5 or x==6:
            return 1
        else:
            return 0

    for col in columns:
        t0 = time.time()
        if use_fillna:
            df[col+"_year"] = df[col].dt.year.fillna(na_value*year_multiplier)
            df[col+"_quarter"] = df[col].dt.quarter.fillna(na_value)
            df[col+"_month"] = df[col].dt.month.fillna(na_value)
            try:
                df[col+"_week"] = df[col].apply(lambda x: np.nan if x is pd.NaT else x.isocalendar().week).fillna(na_value)
            except:
                df[col+"_week"] = df[col].apply(lambda x: np.nan if x is pd.NaT else x.week).fillna(na_value)
            df[col+"_day"] = df[col].dt.day.fillna(na_value)
            df[col+"_hour"] = df[col].dt.hour.fillna(na_value)
            df[col+"_minute"] = df[col].dt.minute.fillna(na_value)
            df[col+"_dayofweek"] = df[col].dt.dayofweek.fillna(na_value)
            df[col+"_isweekend"] = df[col+"_dayofweek"].apply(isweekend).fillna(na_value)
            df[col+"_isholiday"] = df[col].apply(lambda x: np.nan if x is pd.NaT else int(irish_holidays.is_holiday(x)) ).fillna(na_value)
        else:
            df[col+"_year"] = df[col].dt.year
            df[col+"_quarter"] = df[col].dt.quarter
            df[col+"_month"] = df[col].dt.month
            try:
                df[col+"_week"] = df[col].apply(lambda x: np.nan if x is pd.NaT else x.isocalendar().week) 
            except:
                df[col+"_week"] = df[col].apply(lambda x: np.nan if x is pd.NaT else x.week)
            df[col+"_day"] = df[col].dt.day
            df[col+"_hour"] = df[col].dt.hour
            df[col+"_minute"] = df[col].dt.minute
            df[col+"_dayofweek"] = df[col].dt.dayofweek
            df[col+"_isweekend"] = df[col+"_dayofweek"].apply(isweekend)
            df[col+"_isholiday"] = df[col].apply(lambda x: np.nan if x is pd.NaT else int(irish_holidays.is_holiday(x)) )
        print(f"{col} - Elapsed time: {format_time(time.time() - t0)}\n")
    return df

def reduce_dataframe_size(df):
    for col in df.columns:
        if "int" in str(df[col].dtype):
            df[col] = pd.to_numeric(df[col], errors='coerce', downcast="integer")
        elif "float" in str(df[col].dtype):
            df[col] = pd.to_numeric(df[col], errors='coerce', downcast="float")
        else:
            df[col] = df[col].astype("int64")
            df[col] = pd.to_numeric(df[col], errors='coerce', downcast="integer")
    return df


def format_time(elapsed):
    from datetime import timedelta
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))
    # Format as hh:mm:ss
    return str(timedelta(seconds=elapsed_rounded))

def get_series_mode(series):
    import pandas as pd
    import numpy as np
    out = pd.Series.mode(series)
    if out.shape[0] == 0 :
        return np.nan
    else:
        return out.iloc[0]
    
#######################################################################################ß

from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, Normalizer 
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.utils import class_weight

def stratified_sampling(df, stratify_by:'str'=None  , frac=0.01, random_state=None):
    import pandas as pd
    from sklearn.model_selection import train_test_split
    
    _ , x,_ ,y = train_test_split(
        df.drop(stratify_by, axis=1).values, 
        df[stratify_by].values, 
        test_size=frac, 
        random_state=random_state, 
        shuffle=True, 
        stratify = df[stratify_by]
    )
    df_out = pd.DataFrame(x,columns = df.drop(stratify_by, axis=1).columns )
    df_out[stratify_by] = y
    return df_out

class SimpleNormalizer:
    import numpy as np
    def __init__(self, return_values=True):
        self.max_arr = None
        self.return_values = return_values
    
    def fit(self,x):
        self.max_arr = np.max(x)
        
    def transform(self,x):
        out = x/self.max_arr
        return out.values if self.return_values else out
        
    def fit_transform(self,x):
        self.max_arr = np.max(x)
        out = x/self.max_arr
        return out.values if self.return_values else out
    
    
def create_scalers_dict(return_values=False):
    return {
#         "DefaultData": None,
        "simple_normalizer": SimpleNormalizer(return_values=return_values),
        "Normalizer":Normalizer(), 
        "MinMaxScaler":MinMaxScaler(),
        "RobustScaler":RobustScaler(), 
        "StandardScaler":StandardScaler()
    }


def random_search_tuning(x_train=None, y_train=None, default_params=None, params=None, scoring='f1_weighted',
                         model_func=None, use_sample_weight=False, random_state=None, max_iter = 5_000
                        ):
    
    clf = RandomizedSearchCV(
        estimator=model_func(**default_params),
        param_distributions=params,
        scoring= scoring, #['accuracy', 'f1_micro', 'f1_macro', 'f1_weighted'], # neg_mean_absolute_error, accuracy, neg_log_loss, balanced_accuracy, f1_macro
        n_iter=max_iter,
        n_jobs=-1,
        verbose=1,
        cv=5,
        refit=False,
        random_state = random_state,
    )
    if model_func.__name__ == 'MLPClassifier':
        clf.fit(x_train, y_train, 
#                 class_weight=classes_weights if use_sample_weight else None
               )
    else:
        clf.fit(x_train, y_train, sample_weight=samples_weights if use_sample_weight else None)

    return clf


def run_search(x_train=None, y_train=None, default_params=None, params=None, 
               model_func=None, use_sample_weight=False,  scaler_dict=None, 
               random_state=None, max_iter = 5_000):
    score_dict ={}
    params_dict = {}
    if scaler_dict is None:
        t0 = time.time()
        try:
            clf = random_search_tuning(
                x_train=x_train, 
                y_train=y_train, default_params=default_params, params=params, 
                model_func=model_func, use_sample_weight=use_sample_weight, 
                random_state=random_state, max_iter=max_iter
            )

            score_dict["DefaultData"] = round(clf.best_score_, 6)
            params_dict["DefaultData"] = clf.best_params_
            print(f"DefaultData=>\n  Best score: {clf.best_score_:0.6f}\n  Best hyperparameter combination:\n{clf.best_params_}" )
            print(f"Elapsed time: {format_time(time.time() - t0)}\n")
        except Exception as exc:
            print(exc.with_traceback(None))
        return score_dict, params_dict
    
    for scaler_func in list(scaler_dict.keys()) :
        t0 = time.time()
        try:
            if scaler_func == "DefaultData":
                clf = random_search_tuning(
                    x_train=x_train, 
                    y_train=y_train, default_params=default_params, params=params, 
                    model_func=model_func, use_sample_weight=use_sample_weight, 
                    random_state=random_state, max_iter=max_iter
                )
            else:
                clf = random_search_tuning(
                    x_train=scaler_dict[scaler_func].fit_transform(x_train), 
                    y_train=y_train, default_params=default_params, params=params, 
                    model_func=model_func, use_sample_weight=use_sample_weight, 
                    random_state=random_state, max_iter=max_iter
                )

            score_dict[f"{scaler_func}"] = round(clf.best_score_, 6)
            params_dict[f"{scaler_func}"] = clf.best_params_
            print(f"{scaler_func}=>\n  Best score: {clf.best_score_:0.6f}\n  Best hyperparameter combination:\n{clf.best_params_}" )
            print(f"Elapsed time: {format_time(time.time() - t0)}\n")
        except:
            print(f"Error occured for {scaler_func}")

    return score_dict, params_dict

def print_scores(clf=None, x_test=None, y_test=None, y_pred=None):
    y_pred = clf.predict(x_test)
    avg_list = ["micro", "macro", "weighted"]
    metrics_list = [accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score, roc_auc_score]
    for score_func in metrics_list:
        print("------------------------------------")
        if "accuracy_score" not in score_func.__name__:
            for avg in avg_list:
                print(f"{avg} {score_func.__name__}: {score_func(y_test, y_pred, average=avg):0.4%}\n")
            
        else:
            if score_func.__name__ == "balanced_accuracy_score":
                samples_weights = class_weight.compute_sample_weight(
                    class_weight='balanced',
                    y=y_test
                )
                print(f"{score_func.__name__}: {score_func(y_test, y_pred, sample_weight=samples_weights):0.4%}\n") 
                print(f"class weights: {np.unique(samples_weights)}")
            else:
                print(f"{score_func.__name__}: {score_func(y_test, y_pred):0.4%}\n") 
    
    print("====================================\n")
    print(
        "precision_score: ", precision_score(y_test, y_pred, average=None), "\n",
        "recall_score: ", recall_score(y_test, y_pred, average=None),  "\n",
        "f1_score: ", f1_score(y_test, y_pred, average=None),  "\n",
        "roc_auc_score: ", roc_auc_score(y_test, y_pred, average=None)
    )
#######################################################################################ß
"""
Credit to: https://gist.github.com/pod3275/6d22a2b7bf6e628791ff892eef3e80c5
"""
def get_data(data_id, dataset_format='dataframe'):
  """
  * Check on https://www.openml.org/ to get dataset IDs
  * Possible values: {'dataframe', 'array}. Default: 'dataframe'
  """
  import openml
  print("Data loading...")
  dataset = openml.datasets.get_dataset(data_id)

  X, y, categorical_indicator, attribute_names = dataset.get_data(
      dataset_format=dataset_format,
      target=dataset.default_target_attribute
  )

  print("Data load complete.")
  if dataset_format=='dataframe':
    return (X, y)
  else:
    return (X, y, categorical_indicator, attribute_names)

    
#######################################################################################
def normalise_data(df):
    for col in df.columns:
        df[col] = df[col]/df[col].max()
    return df

def compute_MIC(x, y):
    from minepy import MINE
    mine = MINE(alpha=0.6, c=15, est="mic_approx")
    mine.compute_score(x, y)
    return mine.mic()


def calculate_jaccard_score(y_true, y_pred):
    return round(len(set(y_true).intersection(set(y_pred)) ) / len(set(y_true).union(set(y_pred)) ), 4)

#######################################################################################

def train_test_splitting(df, label_col:str="Urban" , under_class_split_rate=0.3, random_state=12):    
    _series = df[label_col].value_counts(dropna=True)
    idx_list = _series.index
    
    test_df_list = []
    train_df_list = []
    
    test_df_temp = df[df[label_col] == idx_list[np.argmin(_series)]].sample(frac=under_class_split_rate, random_state=random_state, replace = False,)
    test_df_list += [test_df_temp]
    train_df_list += [df[df[label_col] == idx_list[np.argmin(_series)]].drop(test_df_temp.index, axis=0)]
    
    num_test_samples = test_df_temp.shape[0]
    
    _df = df[~(df[label_col] == idx_list[np.argmin(_series)])]
    
    for _y in _df[label_col].unique():
        test_df_temp = _df[_df[label_col] == _y].sample(n=num_test_samples, random_state=random_state)
        test_df_list += [test_df_temp]
        train_df_list += [_df[_df[label_col] == _y].drop(test_df_temp.index, axis=0)]

    return pd.concat(train_df_list).reset_index(drop=True), pd.concat(test_df_list).reset_index(drop=True)


def train_data_settings(df_train, label_col = "Urban", aug_rate=0.5, random_state=12):
    """
    - This function uses Using RandomUnderSampler and SMOTE handle unbalanced data.
    """
    
    from imblearn.over_sampling import SMOTE
    from imblearn.under_sampling import RandomUnderSampler #, NearMiss
    
    _series = df_train[label_col].value_counts(dropna=True)
    idx_list = list(_series.index)
    values_list = _series.values
    
    values_list[np.argmin(_series)], (values_list[np.argmin(_series)]*aug_rate),  round(values_list[np.argmin(_series)] * (1+aug_rate))
    
    minority_class_id = idx_list[np.argmin(_series)]
    
    del idx_list[idx_list.index(minority_class_id)]
    
    sampling_strategy_dict = dict()
    for idx in idx_list:
        sampling_strategy_dict[idx] = round(min(values_list) * (1+aug_rate))
    
    randUnderSampler = RandomUnderSampler(
        sampling_strategy=sampling_strategy_dict,
        random_state=random_state,
        replacement=False,
    )
    
    smote = SMOTE(random_state = random_state) 

    try:
        X_res, y_res = randUnderSampler.fit_resample(df_train.drop(label_col, axis=1), df_train[label_col])
    except Exception:
        print(f"Cannot proceed!")
    X_train, y_train = smote.fit_resample(X_res, y_res) 

    from collections import Counter
    print('Resampled dataset shape (after undersampling): %s ' % Counter(y_res))
    print('Resampled dataset shape (after oversampling): %s  ' % Counter(y_train))

    return X_train, y_train

def return_scores(clf=None, x_test=None, y_test=None, y_pred=None):
    
    def add_to_dict(dct, k, v):
        if k in dct.keys():
            dct[k] += [v]
        else:
            dct[k] = [v]

    scores_dict = dict()
    
    y_pred = clf.predict(x_test) if y_pred is None else y_pred
    avg_list = ["micro", "macro", "weighted"]
    metrics_list = [accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score, roc_auc_score]
    for score_func in metrics_list:
        if "accuracy_score" not in score_func.__name__:
            for avg in avg_list:
                add_to_dict(scores_dict, score_func.__name__, round(score_func(y_test, y_pred, average=avg), 6))
            
        else:
            if score_func.__name__ == "balanced_accuracy_score":
                samples_weights = class_weight.compute_sample_weight(
                    class_weight='balanced',
                    y=y_test
                )
                add_to_dict(scores_dict, score_func.__name__, round(score_func(y_test, y_pred, sample_weight=samples_weights), 6))
                
                # print(f"class weights: {np.unique(samples_weights)}")
            else:
                add_to_dict(scores_dict, score_func.__name__, round(score_func(y_test, y_pred), 6))
                
    add_to_dict(scores_dict, "precision_score", round(precision_score(y_test, y_pred, average=None), 6))
    add_to_dict(scores_dict, "recall_score", round(recall_score(y_test, y_pred, average=None), 6))
    add_to_dict(scores_dict, "f1_score", round(f1_score(y_test, y_pred, average=None), 6))
    add_to_dict(scores_dict, "roc_auc_score", round(roc_auc_score(y_test, y_pred, average=None), 6))

    return scores_dict

#####################################  From util_selector.py ##################################################ß
from sklearn.linear_model import LinearRegression, LogisticRegression, RidgeClassifier, BayesianRidge, Ridge 
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, LinearSVC, SVR, LinearSVR
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import roc_auc_score, r2_score, accuracy_score
from sklearn.naive_bayes import GaussianNB

from sklearn.feature_selection import SelectKBest #,SelectPercentile

from sklearn.model_selection import train_test_split
from tqdm import tqdm

import pandas as pd
import numpy as np
import sklearn
import time

########################### Model training functions #################################

def train_models(X_train=None, y_train=None, X_test=None, y_test=None, 
                 model_list=None, selector=None, score_func=None
                ):
    """
    Not for NeurIPS
    """
    scores_list = []
    for _model in model_list:
        model = _model()
        model.fit(X_train if selector is None else selector.transform(X_train), y_train)
        scores_list += [
            # Check eda_util.py for ideas
            score_func(
                y_test, model.predict(X_test if selector is None else selector.transform(X_test)),
                average = "binary" if len(y_test.shape) == 1 else "weighted"
            )
        ]

    display({k:round(v,6) for k,v in zip(model_names,scores_list)})
    return scores_list

########################### Selector Evaluation #################################

def evaluate_selector(model, features:list, X_train, y_train, X_test, y_test, score_func=None, random_state=1, n_digits=4):
    # Training
    model.fit(X_train[features], y_train)
    # Make predictions
    y_pred = model.predict(X_test[features])
    y_pred_train = model.predict(X_train[features])
    return np.float32(round(score_func(y_test, y_pred), n_digits)),  np.float32(round(score_func(y_train, y_pred_train), n_digits))

def evaluate_selector_generalisation(model_list:list, features:list, X_train, y_train, X_test, y_test, score_func=None, selector_name="selector", random_state=1, n_digits=4, testing_scores_dict=dict(), training_scores_dict=dict(), elapsed_time_dict=dict() ):
    """
    score_func = roc_auc_score => for classification
    score_func = r2score => for regression
    """
    
    for model_abst in tqdm(model_list, colour="green"):
        try:
            model = model_abst(random_state=random_state)
        except:
            pass
        finally:
            model = model_abst()
        model_name = model.__str__().split("(")[0]
        # print("------------{}------------".format(model_name))
        t0 = time.time()
        score_testing, score_training = evaluate_selector(model=model, features=features,  X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, score_func=score_func, n_digits=n_digits)
        t1 = time.time()
        if model_name in testing_scores_dict.keys():
            testing_scores_dict[model_name] += [score_testing]
            training_scores_dict[model_name] += [score_training]
            elapsed_time_dict[model_name] += [np.float32(round((t1-t0),3))]
            
            
        else:
            testing_scores_dict[model_name] = [score_testing]
            training_scores_dict[model_name] = [score_training]
            elapsed_time_dict[model_name] = [np.float32(round((t1-t0),3))]
    # print(elapsed_time_dict)
    return testing_scores_dict, training_scores_dict, elapsed_time_dict


def eval_selector_generalisation_stats_test(model_list:list, features:list, X_data, y_data, test_size=0.33, score_func=None, selector_name="selector", n_digits=4, n_iter = 100, stratify=None, save_outputs:bool=False ):
    
    test_scores_dict, train_scores_dict, train_test_duration_dict = dict(), dict(), dict()

    for random_state in tqdm(range(n_iter), colour="red"):
        np.random.seed(random_state)
        sklearn.random.seed(random_state)
        
        X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=test_size, shuffle=True, random_state=random_state , stratify=stratify)
        
        test_scores_dict, train_scores_dict, train_test_duration_dict = evaluate_selector_generalisation(
            model_list=model_list, features=features, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, score_func=score_func, selector_name=selector_name, random_state=random_state, n_digits=n_digits, testing_scores_dict=test_scores_dict, training_scores_dict=train_scores_dict, elapsed_time_dict = train_test_duration_dict
        )
    if save_outputs:
        print("Saving experiment outputs ...")
        pd.DataFrame(test_scores_dict).to_csv("test_scores_"+selector_name+".csv", index=False) 
        pd.DataFrame(train_scores_dict).to_csv("train_scores_"+selector_name+".csv", index=False)
        pd.DataFrame(train_test_duration_dict).to_csv("train_test_duration_"+selector_name+".csv", index=False) 
        print("Done!!!")
    return pd.DataFrame(test_scores_dict), pd.DataFrame(train_scores_dict), pd.DataFrame(train_test_duration_dict)