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

insurance_file = './data/unprocessed/insurance_cross_sell.csv'

insurance_obj = DataPreprocess(insurance_file)
insurance_df = insurance_obj.get_df()

"""Setup Dask Client"""
workers = 1
thr_per_worker = 4
process = False
memory = '2GB'
client = Client(n_workers=workers, threads_per_worker=thr_per_worker, processes=process, memory_limit=memory)

"""Format Column Names"""
insurance_df.columns = insurance_obj.format_column_name(insurance_df)

"""Check dtypes"""
insurance_df_dtypes = insurance_obj.check_dtype(insurance_df)
print(insurance_df_dtypes)

""" Drop gender column"""
insurance_df = insurance_obj.drop_column(insurance_df,"gender")

"""Get list of obj/non-obj columns"""
insurance_obj_cols = insurance_obj.get_obj_col_list(insurance_df)
insurance_nonobj_cols = insurance_obj.get_nonobj_col_list(insurance_df)

"""Convert string (object datatypes) to lowercase"""
insurance_df = insurance_obj.convert_data_to_lowercase(insurance_df, insurance_obj_cols)

"""Get df with number of unique values column-wise"""
insurance_unique_col_values = insurance_obj.get_unique_col_values(insurance_df)
print(insurance_unique_col_values)
"""unnecessary_cols stores columns with number of unique values either equal to 1 or len(df)"""
insurance_unnecessary_cols = insurance_unique_col_values[insurance_unique_col_values['num_unique_values'].isin([1, len(insurance_df)])]
print("columns to be dropped - ", insurance_unnecessary_cols.column.unique())

target_df = pd.DataFrame()
target_df['insurance'] = insurance_df['insurance']
insurance_df.drop('insurance', axis=1, inplace=True)

"""Update list of obj/non-obj columns"""
insurance_obj_cols = insurance_obj.get_obj_col_list(insurance_df)
insurance_nonobj_cols = insurance_obj.get_nonobj_col_list(insurance_df)

"""CALCULATING DERIVED DATA"""

"""Age from DoB"""
insurance_df = insurance_obj.get_age_col(insurance_df, 'dob', '/', 1, 0, 2)
print("age added")

"""insurance Since"""
insurance_df = insurance_obj.get_customer_since(insurance_df, 'customer_to_bank', '/', 1, 0, 2)

"""insurance group - new/old"""
insurance_df = insurance_obj.group_customer(insurance_df, 'customer_since_months', 'customer_rel_dur_segment')
print("grouped")


"""Population and city tier"""
insurance_df = insurance_obj.get_population_column(insurance_df, 'city')
bins_dict = insurance_obj.get_population_bins(insurance_df, 'population')
bins = list(bins_dict.values())
bins = sorted(list(set(bins)))
print("Population bins - ", bins)

labels = insurance_obj.get_labels(bins)
print(labels)
insurance_df = insurance_obj.group_city_from_population(insurance_df, 'population', bins, labels)
print(insurance_df.head())


"""Since we have added some new columns - we have to update our cols_list"""
insurance_obj_cols = insurance_obj.get_obj_col_list(insurance_df)
insurance_nonobj_cols = insurance_obj.get_nonobj_col_list(insurance_df)

insurance_temp_df = insurance_df.copy()
insurance_temp_df.to_csv('./data/unprocessed/consolidated_for_comparison.csv', index=False)

print(insurance_df.head())

"""Handling Missing Data"""
insurance_missing_df = insurance_obj.get_missing_df(insurance_df)
insurance_missing_df = insurance_missing_df[insurance_missing_df.percent_missing > 0]
print(insurance_missing_df)

"""Handling Categorical Variables"""
insurance_cat_col_uniques_dict = insurance_obj.get_cat_cols_unique_val_dict(insurance_df, insurance_obj_cols)
print(insurance_cat_col_uniques_dict)

insurance_binary_cols_list = insurance_cat_col_uniques_dict.get(2).split(",")
insurance_onehot_col_list = insurance_cat_col_uniques_dict.get(14).split(",")     #+ insurance_cat_col_uniques_dict[6].split(",")

insurance_df = insurance_obj.convert_cat_cols_to_binary(insurance_df, insurance_binary_cols_list)
insurance_df = insurance_obj.convert_cat_cols_to_onehot(insurance_df, insurance_onehot_col_list)
# # insurance_df = insurance_obj.group_less_occurring_cat_vars(insurance_df, insurance_obj_cols)
print(insurance_df.head())
print("Categorical data handling complete for insurance data")

"""OUTLIER DETECTION"""
insurance_outlier_df = insurance_obj.outlier_detection(insurance_df[insurance_nonobj_cols])
insurance_outlier_df = insurance_obj.outlier_details(insurance_outlier_df)
print("Outliers")
print(insurance_outlier_df)

"""" log scaling """
list_cols_scaled = ["income"]
insurance_df = insurance_obj.log_scaling(insurance_df,list_cols_scaled)

"""CORRELATION ANALYSIS"""

"""Top Absolute Correlation"""
print("Top Absolute Correlation")
#print(insurance_obj.get_top_abs_correlations(insurance_df.iloc[:, 1:], 10))

"""Highly correlated columns -- exceeding 0.75"""
print("Suggested Highly Correlated Columns to be dropped")
print(insurance_obj.drop_highly_corr_var(insurance_df, 0.75))

insurance_df['insurance'] = target_df['insurance']
print(insurance_df.head())
insurance_df.to_csv("./data/processed/insurance_cross_sell_processed.csv", index=False)
print("--- %s seconds ---" % (time.time() - start_time))