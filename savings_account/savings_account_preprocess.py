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

savings_file = './data/unprocessed/savings_account.csv'

savings_obj = DataPreprocess(savings_file)

savings_df = savings_obj.get_df()

"""Setup Dask Client"""
workers = 1
thr_per_worker = 4
process = False
memory = '2GB'
client = Client(n_workers=workers, threads_per_worker=thr_per_worker, processes=process, memory_limit=memory)



"""Format Column Names"""
savings_df.columns = savings_obj.format_column_name(savings_df)

"""Check dtypes"""
savings_df_dtypes = savings_obj.check_dtype(savings_df)
print(savings_df_dtypes)

""" Drop gender column"""
savings_df = savings_obj.drop_column(savings_df,"gender")

"""Get list of obj/non-obj columns"""
savings_obj_cols = savings_obj.get_obj_col_list(savings_df)
savings_nonobj_cols = savings_obj.get_nonobj_col_list(savings_df)

"""Convert string (object datatypes) to lowercase"""
savings_df = savings_obj.convert_data_to_lowercase(savings_df, savings_obj_cols)

"""Get df with number of unique values column-wise"""
savings_unique_col_values = savings_obj.get_unique_col_values(savings_df)
print(savings_unique_col_values)
"""unnecessary_cols stores columns with number of unique values either equal to 1 or len(df)"""
savings_unnecessary_cols = savings_unique_col_values[savings_unique_col_values['num_unique_values'].isin([1, len(savings_df)])]
print("columns to be dropped - ", savings_unnecessary_cols.column.unique())

target_df = pd.DataFrame()
target_df['savings_account_avl'] = savings_df['savings_account_avl']
savings_df.drop('savings_account_avl', axis=1, inplace=True)

"""Update list of obj/non-obj columns"""
savings_obj_cols = savings_obj.get_obj_col_list(savings_df)
savings_nonobj_cols = savings_obj.get_nonobj_col_list(savings_df)

"""CALCULATING DERIVED DATA"""

"""Age from DoB"""
savings_df = savings_obj.get_age_col(savings_df, 'dob', '/', 1, 0, 2)
print("age added")

"""Customer Since"""
savings_df = savings_obj.get_customer_since(savings_df, 'customer_to_bank', '/', 1, 0, 2)

"""Customer group - new/old"""
savings_df = savings_obj.group_customer(savings_df, 'customer_since_months', 'customer_rel_dur_segment')
print("grouped")

"""Population and city tier"""
savings_df = savings_obj.get_population_column(savings_df, 'city')
bins_dict = savings_obj.get_population_bins(savings_df, 'population')
bins = list(bins_dict.values())
bins = sorted(list(set(bins)))
print("Population bins - ", bins)

labels = savings_obj.get_labels(bins)
print(labels)
savings_df= savings_obj.group_city_from_population(savings_df, 'population', bins, labels)
print(savings_df.head())



"""Since we have added some new columns - we have to update our cols_list"""
savings_obj_cols = savings_obj.get_obj_col_list(savings_df)
savings_nonobj_cols = savings_obj.get_nonobj_col_list(savings_df)

savings_temp_df = savings_df.copy()
savings_temp_df.to_csv('./data/unprocessed/consolidated_for_comparison.csv', index=False)

print(savings_df.head())

"""Handling Missing Data"""
savings_missing_df = savings_obj.get_missing_df(savings_df)
savings_missing_df = savings_missing_df[savings_missing_df.percent_missing > 0]
print(savings_missing_df)

"""Handling Categorical Variables"""
savings_cat_col_uniques_dict = savings_obj.get_cat_cols_unique_val_dict(savings_df, savings_obj_cols)
print(savings_cat_col_uniques_dict)

savings_binary_cols_list = savings_cat_col_uniques_dict.get(2).split(",")
savings_onehot_col_list = savings_cat_col_uniques_dict.get(5).split(",")     #+ savings_cat_col_uniques_dict[6].split(",")

savings_df = savings_obj.convert_cat_cols_to_binary(savings_df, savings_binary_cols_list)
savings_df = savings_obj.convert_cat_cols_to_onehot(savings_df, savings_onehot_col_list)
# # customer_df = customer_obj.group_less_occurring_cat_vars(customer_df, customer_obj_cols)
print(savings_df.head())
print("Categorical data handling complete for Customer data")

"""OUTLIER DETECTION"""
savings_outlier_df = savings_obj.outlier_detection(savings_df[savings_nonobj_cols])
savings_outlier_df = savings_obj.outlier_details(savings_outlier_df)
print("Outliers")
print(savings_outlier_df)

"""" log scaling """
list_cols_scaled = ["income"]
savings_df = savings_obj.log_scaling(savings_df,list_cols_scaled)

"""CORRELATION ANALYSIS"""

"""Top Absolute Correlation"""
print("Top Absolute Correlation")
#print(savings_obj.get_top_abs_correlations(savings_df.iloc[:, 1:], 10))

"""Highly correlated columns -- exceeding 0.75"""
print("Suggested Highly Correlated Columns to be dropped")
print(savings_obj.drop_highly_corr_var(savings_df, 0.75))

savings_df['savings_account_avl'] = target_df['savings_account_avl']

print(savings_df.head())
savings_df.to_csv("./data/processed/savings_processed.csv", index=False)
print("--- %s seconds ---" % (time.time() - start_time))