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

fund_file = './data/unprocessed/funds.csv'

fund_obj = DataPreprocess(fund_file)
fund_df = fund_obj.get_df()

"""Setup Dask Client"""
workers = 1
thr_per_worker = 4
process = False
memory = '2GB'
client = Client(n_workers=workers, threads_per_worker=thr_per_worker, processes=process, memory_limit=memory)



"""Format Column Names"""
fund_df.columns = fund_obj.format_column_name(fund_df)

"""Check dtypes"""
fund_df_dtypes = fund_obj.check_dtype(fund_df)
print(fund_df_dtypes)

""" Drop gender column"""
fund_df = fund_obj.drop_column(fund_df,"gender")

"""Get list of obj/non-obj columns"""
fund_obj_cols = fund_obj.get_obj_col_list(fund_df)
fund_nonobj_cols = fund_obj.get_nonobj_col_list(fund_df)

"""Convert string (object datatypes) to lowercase"""
fund_df = fund_obj.convert_data_to_lowercase(fund_df, fund_obj_cols)

"""Get df with number of unique values column-wise"""
fund_unique_col_values = fund_obj.get_unique_col_values(fund_df)
print(fund_unique_col_values)
"""unnecessary_cols stores columns with number of unique values either equal to 1 or len(df)"""
fund_unnecessary_cols = fund_unique_col_values[fund_unique_col_values['num_unique_values'].isin([1, len(fund_df)])]
print("columns to be dropped - ", fund_unnecessary_cols.column.unique())

target_df = pd.DataFrame()
target_df['fund_type'] = fund_df['fund_type']
fund_df.drop('fund_type', axis=1, inplace=True)

"""Update list of obj/non-obj columns"""
fund_obj_cols = fund_obj.get_obj_col_list(fund_df)
fund_nonobj_cols = fund_obj.get_nonobj_col_list(fund_df)

"""CALCULATING DERIVED DATA"""

"""Age from DoB"""
fund_df = fund_obj.get_age_col(fund_df, 'dob', '/', 1, 0, 2)
print("age added")

"""Customer Since"""
fund_df = fund_obj.get_customer_since(fund_df, 'customer_to_bank', '/', 1, 0, 2)

"""Customer group - new/old"""
fund_df = fund_obj.group_customer(fund_df, 'customer_since_months', 'customer_rel_dur_segment')
print("grouped")


"""Population and city tier"""
fund_df = fund_obj.get_population_column(fund_df, 'city')
bins_dict = fund_obj.get_population_bins(fund_df, 'population')
bins = list(bins_dict.values())
bins = sorted(list(set(bins)))
print("Population bins - ", bins)

labels = fund_obj.get_labels(bins)
print(labels)
fund_df = fund_obj.group_city_from_population(fund_df, 'population', bins, labels)
print(fund_df.head())




"""Since we have added some new columns - we have to update our cols_list"""
fund_obj_cols = fund_obj.get_obj_col_list(fund_df)
fund_nonobj_cols = fund_obj.get_nonobj_col_list(fund_df)

fund_temp_df = fund_df.copy()
fund_temp_df.to_csv('./data/unprocessed/consolidated_for_comparison.csv', index=False)

print(fund_df.head())

"""Handling Missing Data"""
fund_missing_df = fund_obj.get_missing_df(fund_df)
fund_missing_df = fund_missing_df[fund_missing_df.percent_missing > 0]
print(fund_missing_df)

"""Handling Categorical Variables"""
fund_cat_col_uniques_dict = fund_obj.get_cat_cols_unique_val_dict(fund_df, fund_obj_cols)
print(fund_cat_col_uniques_dict)

fund_binary_cols_list = fund_cat_col_uniques_dict.get(2).split(",")
fund_onehot_col_list = fund_cat_col_uniques_dict[5].split(",")

fund_df = fund_obj.convert_cat_cols_to_binary(fund_df, fund_binary_cols_list)
fund_df = fund_obj.convert_cat_cols_to_onehot(fund_df, fund_onehot_col_list)
# # customer_df = customer_obj.group_less_occurring_cat_vars(customer_df, customer_obj_cols)
print(fund_df.head())
print("Categorical data handling complete for Customer data")

"""OUTLIER DETECTION"""
fund_outlier_df = fund_obj.outlier_detection(fund_df[fund_nonobj_cols])
fund_outlier_df = fund_obj.outlier_details(fund_outlier_df)
print("Outliers")
print(fund_outlier_df)

"""" log scaling """
list_cols_scaled = ["income"]
fund_df = fund_obj.log_scaling(fund_df,list_cols_scaled)

"""CORRELATION ANALYSIS"""

"""Top Absolute Correlation"""
print("Top Absolute Correlation")
#print(fund_obj.get_top_abs_correlations(fund_df.iloc[:, 1:], 10))

"""Highly correlated columns -- exceeding 0.75"""
print("Suggested Highly Correlated Columns to be dropped")
print(fund_obj.drop_highly_corr_var(fund_df, 0.75))

fund_df['fund_type'] = target_df['fund_type']
print(fund_df.head())
fund_df.to_csv("./data/processed/fund_processed.csv", index=False)
print("--- %s seconds ---" % (time.time() - start_time))