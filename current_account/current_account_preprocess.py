import pandas as pd
import numpy as np
from dask.distributed import Client

import warnings
warnings.filterwarnings('ignore')

import time
import sys
sys.path.append('/home/sandeep/Desktop/BankBuddy/Reco-usecases/new_reco/preprocessing_pipeline/')


from DataPreprocess import *

start_time = time.time()

currentaccount_file = './data/unprocessed/current_account.csv'

currentaccount_obj = DataPreprocess(currentaccount_file)
currentaccount_df = currentaccount_obj.get_df()

"""Setup Dask Client"""
workers = 1
thr_per_worker = 4
process = False
memory = '2GB'
client = Client(n_workers=workers, threads_per_worker=thr_per_worker, processes=process, memory_limit=memory)



"""Format Column Names"""
currentaccount_df.columns = currentaccount_obj.format_column_name(currentaccount_df)

"""Check dtypes"""
currentaccount_df_dtypes = currentaccount_obj.check_dtype(currentaccount_df)
print(currentaccount_df_dtypes)

""" Drop gender column"""
currentaccount_df = currentaccount_obj.drop_column(currentaccount_df,"gender")

"""Get list of obj/non-obj columns"""
currentaccount_obj_cols = currentaccount_obj.get_obj_col_list(currentaccount_df)
currentaccount_nonobj_cols = currentaccount_obj.get_nonobj_col_list(currentaccount_df)

"""Convert string (object datatypes) to lowercase"""
currentaccount_df = currentaccount_obj.convert_data_to_lowercase(currentaccount_df, currentaccount_obj_cols)

"""Get df with number of unique values column-wise"""
currentaccount_unique_col_values = currentaccount_obj.get_unique_col_values(currentaccount_df)
print(currentaccount_unique_col_values)
"""unnecessary_cols stores columns with number of unique values either equal to 1 or len(df)"""
currentaccount_unnecessary_cols = currentaccount_unique_col_values[currentaccount_unique_col_values['num_unique_values'].isin([1, len(currentaccount_df)])]
print("columns to be dropped - ", currentaccount_unnecessary_cols.column.unique())

target_df = pd.DataFrame()
target_df['current_acc_avl'] = currentaccount_df['current_acc_avl']
currentaccount_df.drop('current_acc_avl', axis=1, inplace=True)

"""Update list of obj/non-obj columns"""
currentaccount_obj_cols = currentaccount_obj.get_obj_col_list(currentaccount_df)
currentaccount_nonobj_cols = currentaccount_obj.get_nonobj_col_list(currentaccount_df)

"""CALCULATING DERIVED DATA"""

"""Age from DoB"""
currentaccount_df = currentaccount_obj.get_age_col(currentaccount_df, 'dob', '/', 1, 0, 2)
print("age added")

"""Customer Since"""
currentaccount_df = currentaccount_obj.get_customer_since(currentaccount_df, 'customer_to_bank', '/', 1, 0, 2)

"""Customer group - new/old"""
currentaccount_df = currentaccount_obj.group_customer(currentaccount_df, 'customer_since_months', 'customer_rel_dur_segment')
print("grouped")

"""Population and city tier"""
currentaccount_df = currentaccount_obj.get_population_column(currentaccount_df, 'city')
bins_dict = currentaccount_obj.get_population_bins(currentaccount_df, 'population')
bins = list(bins_dict.values())
bins = sorted(list(set(bins)))
print("Population bins - ", bins)

labels = currentaccount_obj.get_labels(bins)
print(labels)
currentaccount_df= currentaccount_obj.group_city_from_population(currentaccount_df, 'population', bins, labels)
print(currentaccount_df.head())



"""Since we have added some new columns - we have to update our cols_list"""
currentaccount_obj_cols = currentaccount_obj.get_obj_col_list(currentaccount_df)
currentaccount_nonobj_cols = currentaccount_obj.get_nonobj_col_list(currentaccount_df)

currentaccount_temp_df = currentaccount_df.copy()
currentaccount_temp_df.to_csv('./data/unprocessed/consolidated_for_comparison.csv', index=False)

print(currentaccount_df.head())

"""Handling Missing Data"""
currentaccount_missing_df = currentaccount_obj.get_missing_df(currentaccount_df)
currentaccount_missing_df = currentaccount_missing_df[currentaccount_missing_df.percent_missing > 0]
print(currentaccount_missing_df)

"""Handling Categorical Variables"""
currentaccount_cat_col_uniques_dict = currentaccount_obj.get_cat_cols_unique_val_dict(currentaccount_df, currentaccount_obj_cols)
print(currentaccount_cat_col_uniques_dict)

currentaccount_binary_cols_list = currentaccount_cat_col_uniques_dict.get(2).split(",")
currentaccount_onehot_col_list = currentaccount_cat_col_uniques_dict.get(4).split(",") + currentaccount_cat_col_uniques_dict.get(5).split(",")     #+ currentaccount_cat_col_uniques_dict[6].split(",")

currentaccount_df = currentaccount_obj.convert_cat_cols_to_binary(currentaccount_df, currentaccount_binary_cols_list)
currentaccount_df = currentaccount_obj.convert_cat_cols_to_onehot(currentaccount_df, currentaccount_onehot_col_list)
# # customer_df = customer_obj.group_less_occurring_cat_vars(customer_df, customer_obj_cols)
print(currentaccount_df.head())
print("Categorical data handling complete for Customer data")

"""OUTLIER DETECTION"""
currentaccount_outlier_df = currentaccount_obj.outlier_detection(currentaccount_df[currentaccount_nonobj_cols])
currentaccount_outlier_df = currentaccount_obj.outlier_details(currentaccount_outlier_df)
print("Outliers")
print(currentaccount_outlier_df)

"""" log scaling """
list_cols_scaled = ["income"]
currentaccount_df = currentaccount_obj.log_scaling(currentaccount_df,list_cols_scaled)

"""CORRELATION ANALYSIS"""

"""Top Absolute Correlation"""
print("Top Absolute Correlation")
#print(currentaccount_obj.get_top_abs_correlations(currentaccount_df.iloc[:, 1:], 10))

"""Highly correlated columns -- exceeding 0.75"""
print("Suggested Highly Correlated Columns to be dropped")
print(currentaccount_obj.drop_highly_corr_var(currentaccount_df, 0.75))

currentaccount_df['current_acc_avl'] = target_df['current_acc_avl']

print(currentaccount_df.head())
currentaccount_df.to_csv("./data/processed/currentaccount_processed.csv", index=False)
print("--- %s seconds ---" % (time.time() - start_time))