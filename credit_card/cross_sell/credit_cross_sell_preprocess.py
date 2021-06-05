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

credit_cross_file = './data/unprocessed/credit_crossell.csv'
credit_obj = DataPreprocess(credit_cross_file)
credit_df = credit_obj.get_df()
print(len(credit_df))

"""Setup Dask Client"""
workers = 1
thr_per_worker = 4
process = False
memory = '2GB'

client = Client(n_workers=workers, threads_per_worker=thr_per_worker, processes=process, memory_limit=memory)

"""Format Column Names"""
credit_df.columns = credit_obj.format_column_name(credit_df)

"""Check dtypes"""
credit_df_dtypes = credit_obj.check_dtype(credit_df)
print(credit_df_dtypes)

""" Drop gender column"""
credit_df = credit_obj.drop_column(credit_df,"gender")

"""Get list of obj/non-obj columns"""
credit_obj_cols = credit_obj.get_obj_col_list(credit_df)
credit_nonobj_cols = credit_obj.get_nonobj_col_list(credit_df)

"""Convert string (object datatypes) to lowercase"""
credit_df = credit_obj.convert_data_to_lowercase(credit_df, credit_obj_cols)

"""Get df with number of unique values column-wise"""
credit_unique_col_values = credit_obj.get_unique_col_values(credit_df)
print(credit_unique_col_values)

"""Dropping rows with duplicate customer_ids"""
credit_df = credit_df.drop_duplicates(subset='customer_id', keep="last")
credit_df = credit_df.reset_index(drop=True)
print(len(credit_df))

"""unnecessary_cols stores columns with number of unique values either equal to 1 or len(df)"""
credit_unnecessary_cols = credit_unique_col_values[credit_unique_col_values['num_unique_values'].isin([1, len(credit_df)])]
print("columns to be dropped - ", credit_unnecessary_cols.column.unique())

target_df = pd.DataFrame()
target_df['card_type'] = credit_df['card_type']
credit_df.drop('card_type', axis=1, inplace=True)

"""Update list of obj/non-obj columns"""
credit_obj_cols = credit_obj.get_obj_col_list(credit_df)
credit_nonobj_cols = credit_obj.get_nonobj_col_list(credit_df)

"""Age from DoB"""
credit_df = credit_obj.get_age_col(credit_df, 'dob', '/', 1, 0, 2)
print("age added")

"""Population and city tier"""
credit_df = credit_obj.get_population_column(credit_df, 'city')
bins_dict = credit_obj.get_population_bins(credit_df, 'population')
bins = list(bins_dict.values())
bins = sorted(list(set(bins)))
print("Population bins - ", bins)

labels = credit_obj.get_labels(bins)
print(labels)
credit_df = credit_obj.group_city_from_population(credit_df, 'population', bins, labels)
print(credit_df.head())

"""Since we have added some new columns - we have to update our cols_list"""
credit_obj_cols = credit_obj.get_obj_col_list(credit_df)
credit_nonobj_cols = credit_obj.get_nonobj_col_list(credit_df)

temp_credit_df = credit_df.copy()
temp_credit_df.to_csv('./data/unprocessed/consolidated_for_comparison.csv', index=False)

"""Handling Missing Data"""
credit_missing_df = credit_obj.get_missing_df(credit_df)
credit_missing_df = credit_missing_df[credit_missing_df.percent_missing > 0]
print(credit_missing_df)

# credit_df = credit_obj.missing_data_handle(credit_df, credit_missing_df, credit_nonobj_cols)
print(credit_df.info())
print("Missing data handling complete for Credit Cross Sell data")

"""Handling Categorical Variables"""
credit_cat_col_uniques_dict = credit_obj.get_cat_cols_unique_val_dict(credit_df, credit_obj_cols)
print(credit_cat_col_uniques_dict)

credit_binary_cols_list = credit_cat_col_uniques_dict.get(2).split(",")
credit_onehot_col_list = credit_cat_col_uniques_dict.get(5).split(",")
# + customer_cat_col_uniques_dict[8].split(",") + customer_cat_col_uniques_dict[5].split(",")

credit_df = credit_obj.convert_cat_cols_to_binary(credit_df, credit_binary_cols_list)
credit_df = credit_obj.convert_cat_cols_to_onehot(credit_df, credit_onehot_col_list)
# customer_df = customer_obj.group_less_occurring_cat_vars(customer_df, customer_obj_cols)
print(credit_df.head())
print("Categorical data handling complete for Credit Cross Sell Data.")

"""OUTLIER DETECTION"""
credit_outlier_df = credit_obj.outlier_detection(credit_df[credit_nonobj_cols])
credit_outlier_df = credit_obj.outlier_details(credit_outlier_df)
print("Outliers")
print(credit_outlier_df)

"""" log scaling """
list_cols_scaled = ["income"]
credit_df = credit_obj.log_scaling(credit_df,list_cols_scaled)

"""CORRELATION ANALYSIS"""

"""Top Absolute Correlation"""
print("Top Absolute Correlation")
print(credit_obj.get_top_abs_correlations(credit_df.iloc[:, 1:], 10))

"""Highly correlated columns -- exceeding 0.75"""
print("Suggested Highly Correlated Columns to be dropped")
print(credit_obj.drop_highly_corr_var(credit_df, 0.75))

credit_df['card_type'] = target_df['card_type']
print(credit_df.head())

"""Storing Processed CSVs"""
credit_df.to_csv("./data/processed/credit_cross_sell_processed.csv", index=False)