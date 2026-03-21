import pandas as pd
import numpy as np
import os
import time

from os.path import dirname, join as pathjoin
from glob import glob
from tqdm import tqdm

# import matplotlib.pyplot as plt
# import seaborn as sns
# sns.set(style="whitegrid")

import gc 

import sys
sys.path.append("./utils/")
# from util_selector import *
from util_preprocessing import check_or_create_directory
# from util_postprocessing import *
from util_models import *

########################### Some Variables #################################



###########################   Postprocessing   #################################

def get_filename_list(dir_name, valuation_method, selection_method):
    filename_list = None
    if selection_method == "mRMR":
        if valuation_method == 'mutual_info':
            filename_list = glob(pathjoin(dir_name, "{}-{}-{}.csv".format("MI*", selection_method, '*')))
        elif  valuation_method == 'Fvalue':
            filename_list = glob(pathjoin(dir_name, "{}-{}-{}.csv".format("FC*", selection_method, '*')))
        elif  valuation_method == 'gini':
            filename_list = glob(pathjoin(dir_name, "{}-{}-{}.csv".format("RFC*", selection_method, '*')))
        else:
            pass
    else:
        filename_list = glob(pathjoin(dir_name, "{}-{}-{}.csv".format(valuation_method, selection_method, '*')))
    return filename_list

def extract_num_features(x):
    try:
        _x = x.split("(")[-1].split(")")[0]
    except:
        _x = x
    finally:
        return _x

def extract_performance_scores(x):
    try:
        _x = x.split("±")[0]
    except:
        _x = x
    finally:
        return _x

def pivot_dataframe(df, scoring="f1_weighted"):
    return df.pivot(
        columns=['selection_algorithm'],
        index = [
            'dataset_name', 'model_name', 'include_lowest',#'use_fix_range'
        ],
        values = ['test_{}'.format(scoring)],
    )
     
def clean_up_dataframe(df, level_to_drop:int=0):
    try:
        df = df.droplevel(level_to_drop, axis=1)
        df.drop(["model_name"], axis=1, inplace=True)
    except:
        pass
    df.reset_index(inplace=True)
    df = df.fillna("-")
    return df

def convert_duration_to_seconds(elapsed_time:str):
    _days = None
    _splits = None
    try:
        _splits = elapsed_time.split(" ")
        _days = np.float32(_splits[0])
    except:
        _days = 0.0
    
    _hours = np.float32(_splits[-1].split(":")[0])
    _minutes = np.float32(_splits[-1].split(":")[1])
    _seconds = np.float32(_splits[-1].split(":")[2])
    return _days*24*3600+_hours*3600+_minutes*60+_seconds
    

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

### The two functon below are used to format the reported results  
def get_best_training_config(
    df, metric_col_name = 'test_f1_weighted', 
    best_per_model=True, best_per_selection_algorithm=False, best_per_valuation_method=False, #groupby_use_fix_range=False,
    filter_by_num_selected_features=True, filter_by_num_desired_features=True, drop_unnecessay_cols=True, 
):
    _groupby = ['dataset_name']
    if best_per_model:
        _groupby += ['model_name']
    if best_per_selection_algorithm:
        _groupby += ['selection_algorithm']
    if best_per_valuation_method:
        _groupby += ['valuation_method']
    # if groupby_use_fix_range:
    #     _groupby += ['use_fix_range']
        
    _df = df.groupby(_groupby).agg({metric_col_name:max}).reset_index().merge(df, how="inner")

    if filter_by_num_selected_features:
        _df = _df.groupby(_groupby).agg({'num_selected_features':min}).reset_index().merge(_df, how="inner")
    if filter_by_num_desired_features:
        _df = _df.groupby(_groupby).agg({'num_desired_features':min}).reset_index().merge(_df, how="inner")

    scoring_name = metric_col_name.split("test_")[-1]
    if drop_unnecessay_cols:
        _df.drop(
            [
                'fit_time', 'fit_time_std', 'score_time', 'score_time_std',
                'train_{}'.format(scoring_name),
                'train_{}_std'.format(scoring_name), 
                # 'use_fix_range'
            ], axis=1, inplace=True
        )
    return _df

def upgrade_results_presentation(
    df, metric_col_name = 'test_f1_weighted', 
    num_selected_features_name = 'num_selected_features', num_desired_features_name = 'num_desired_features' 
):
    cols_to_drop = [metric_col_name+"_std", num_selected_features_name, num_desired_features_name]
    
    df[metric_col_name] = df[metric_col_name].round(2).astype(str).str[:5]
    df[metric_col_name+"_std"] = df[metric_col_name+"_std"].round(2).astype(str).str[:5]
    df[num_selected_features_name] = df[num_selected_features_name].astype(int).astype(str)
    
    if num_desired_features_name is None:
        cols_to_drop.remove(None)
        df[metric_col_name] = df[metric_col_name]+"±"+ df[metric_col_name+"_std"]+" ("+ df[num_selected_features_name]+")"
    else:
        df[num_desired_features_name] = df[num_desired_features_name].astype(int).astype(str)
        df[metric_col_name] = df[metric_col_name]+"±"+ df[metric_col_name+"_std"]+" ("+ df[num_selected_features_name]+"/"+ df[num_desired_features_name] +")"
    
    df.drop(cols_to_drop, axis=1, inplace=True)
    return df

def upgrade_aggregated_results_presentation(df_mean, df_std):
    _df_mean, _df_std = df_mean.copy(), df_std.copy()
    cols = list(df_mean.select_dtypes(exclude="object").columns)

    _df = df_mean.select_dtypes(include="object")

    for col in cols:
        _df_mean[col] = _df_mean[col].round(2).astype(str).str[:5]
        _df_std[col] = _df_std[col].round(2).astype(str).str[:5]
        _df[col] = _df_mean[col]+"±"+ _df_std[col]
    return _df

def extract_and_save_best_config_per_model(output_dir:str=None, metric_col_name:str = 'test_accuracy'):

    output_dfs_list = glob(pathjoin(output_dir,"*.csv"))
    
    valuation_method_list = set([filename.split("/")[-1].split("-")[0]  for filename in output_dfs_list])
    
    check_or_create_directory(pathjoin(output_dir,"best_configs"))
    
    for valuation_method in tqdm(valuation_method_list, colour="green"):
        output_dfs_list = glob(pathjoin(output_dir,"{}-*.csv".format(valuation_method)))
        selection_method_list = set([filename.split("/")[-1].split("-")[1]  for filename in output_dfs_list])
        
        for selection_method in selection_method_list:
            # print(valuation_method, selection_method)
            filename_list = glob(pathjoin(output_dir,"{}-{}-*.csv".format(valuation_method, selection_method)))
    
            dfs_list = []
            for filename in filename_list:
                 dfs_list += [pd.read_csv(filename)]
            
            df = pd.concat(dfs_list).reset_index(drop=True)
            for model_name in df["model_name"].unique():
                df_new = get_best_training_config(
                    df[df["model_name"]==model_name], 
                    metric_col_name = metric_col_name, 
                    best_per_model=False, # If commented, it means use default value
                    # best_per_selection_algorithm=True, 
                    # best_per_valuation_method=True,
                )

                if selection_method == "mRMR":
                    df_new["selection_algorithm"] = df_new["selection_algorithm"]+"_"+df_new["valuation_method"]
                
                df_new.to_csv(
                    pathjoin(output_dir,"best_configs")+"/{}-{}-{}.csv".format(valuation_method, selection_method, model_name),
                    index=False
                )
            # print(df_new.shape)
            # display(df_new.head())

############### Visualizations

# def plot_results(
#     results_dicts:dict=None, model_names:list=clf_names, metric_name:str="test_accuracy", aggfunc=np.mean,  colors:dict=None, 
#     show_error_bar=True, figure_title="Comparative analysis", filename=None, legend_position="top",
#     multiply_by=100, bars_width = .7,
#     # figsize=(4,6),
# ):
#     aggData_to_plot = {"model_name":model_names}
#     std_for_ci = dict()
    
#     for kk in results_dicts.keys():
#         aggData_to_plot[kk] = [multiply_by*aggfunc(results_dicts[kk][k][metric_name]) for k in  results_dicts[kk].keys()]
#         std_for_ci[kk] = [multiply_by*np.std(results_dicts[kk][k][metric_name]) for k in  results_dicts[kk].keys()]
        
#     fig, ax = plt.subplots(figsize=(4,6))
#     fig.tight_layout()
#     ax = pd.DataFrame(
#         aggData_to_plot
#     ).set_index("model_name").plot(
#         kind="barh", ax=ax, xerr=std_for_ci if show_error_bar else None,
#         color=None if colors is None else colors,
#         width=bars_width,
#     )
#     plt.legend(loc="center left")
#     ax.set_title(figure_title, fontdict={'size': 14}) 
#     ax.set_ylabel("", fontdict={'size': 14}) 
#     ax.set_xlabel("{}".format(metric_name) , fontdict={'size': 14})
#     # ax.get_legend().remove()
#     labels = list(results_dicts.keys())
#     handles = [plt.Rectangle((0,0),1,1, color=colors[label]) for label in labels]
#     if legend_position == "top":
#         plt.legend(handles, labels, bbox_to_anchor=(0.5, 0.95, 0.3, 0.3))
#     elif legend_position == "bottom":
#         plt.legend(handles, labels, bbox_to_anchor=(0.5, -0.4, 0.3, 0.3))
#     elif legend_position == "side":
#         plt.legend(handles, labels, bbox_to_anchor=(1.16, 0.42, 0.5, 0.2))
#     else:
#         plt.legend(handles, labels, bbox_to_anchor=(1.16, 0.42, 0.5, 0.2))
#     plt.savefig("./outputs/figures/{}-{}.png".format(metric_name, filename), dpi=200, bbox_inches='tight')
#     plt.show()