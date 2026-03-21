# KGroups_Bio

This repository contains the code files of a filter feature selection algorithm I designed and developped during my PhD studies at University College Dublin (UCD) in Ireland. 

My algorithm (KGroups) uses clustering for feature selection instead of sorting (in descending order) or an incremental/greedy algorithm such as sequential forward search.

## Structure
* Core functions are located in [utils/util_selector.py](utils/util_selector.py)

* Code files for the experiments on classification tabular datasets are  in [classification](./classification/)

* Datasets are located at [classification/inputs/datasets.mat](classification/inputs/datasets.mat)

* The outputs of KGroups are saved in this folder [classification/classification/outputs](outputs) and its subfolders.

* To analyse datasets histogram use [classification/Histograms.ipynb](classification/Histograms.ipynb)

* To play around with different binning technique (including ours), use [classification/binnings.ipynb](classification/binnings.ipynb)

* To generate experimental data for all the datasets, please use [classification/run_experiments_on_all_datasets.py](classification/run_experiments_on_all_datasets.py). You can run this Python file in the Terminal by typing `python3 run_experiments_on_all_datasets.py` or running this Jupyter notebook file [classification/run_python_file.ipynb](run_python_file.ipynb) (Need Python 3.8 at least). The results will be saved in csv files located at [classification/outputs/dataframes](outputs/dataframes)

* To have the same tables as the ones presented in the paper use [classification/FS_results_extraction.ipynb](classification/FS_results_extraction.ipynb). The resulting csv files will be stored at [classification/outputs/extracted_results](classification/outputs/extracted_results)


* To generate the experimental results all the biological datasets, please uncomment the line ``for mat_fname in tqdm(datasets_list, colour="green"): # Use this to run the experiment on all the datasets`` and comment out the line ``for mat_fname in tqdm([datasets_list[7]], colour="green"): # Use this to test the experiment on one dataset``. If you would like to test the code on a single dataset, comment the former and uncomment the latter.

* The file [Analysis_for_Chap7.ipynb](Analysis_for_Chap7.ipynb) is an outlier to the above. It analyses data extracted from a sample of 28 filter feature selection studies published between 1994 and 2025.
 

## Requirements
The required packages to be able use KGroups are defined in [requirements.txt](requirements.txt)

To install the requirements, please run the following command in the Terminal `pip3 install -r /path/to/requirements.txt`.

## Licence



