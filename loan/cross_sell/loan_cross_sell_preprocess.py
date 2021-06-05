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

loan_file = './data/unprocessed/loan_cross_sell.csv'

loan_obj = DataPreprocess(loan_file)
loan_df = loan_obj.get_df()

"""Setup Dask Client"""
workers = 1
thr_per_worker = 4
process = False
memory = '2GB'
client = Client(n_workers=workers, threads_per_worker=thr_per_worker, processes=process, memory_limit=memory)



"""Format Column Names"""
loan_df.columns = loan_obj.format_column_name(loan_df)

"""Check dtypes"""
loan_df_dtypes = loan_obj.check_dtype(loan_df)
print(loan_df_dtypes)

""" Drop gender column"""
loan_df = loan_obj.drop_column(loan_df,"gender")

"""Get list of obj/non-obj columns"""
loan_obj_cols = loan_obj.get_obj_col_list(loan_df)
loan_nonobj_cols = loan_obj.get_nonobj_col_list(loan_df)

"""Convert string (object datatypes) to lowercase"""
loan_df = loan_obj.convert_data_to_lowercase(loan_df, loan_obj_cols)

"""Get df with number of unique values column-wise"""
loan_unique_col_values = loan_obj.get_unique_col_values(loan_df)
print(loan_unique_col_values)
"""unnecessary_cols stores columns with number of unique values either equal to 1 or len(df)"""
loan_unnecessary_cols = loan_unique_col_values[loan_unique_col_values['num_unique_values'].isin([1, len(loan_df)])]
print("columns to be dropped - ", loan_unnecessary_cols.column.unique())

target_df = pd.DataFrame()
target_df['loan_availed'] = loan_df['loan_availed']
loan_df.drop('loan_availed', axis=1, inplace=True)

"""Update list of obj/non-obj columns"""
loan_obj_cols = loan_obj.get_obj_col_list(loan_df)
loan_nonobj_cols = loan_obj.get_nonobj_col_list(loan_df)

"""CALCULATING DERIVED DATA"""

"""Age from DoB"""
loan_df = loan_obj.get_age_col(loan_df, 'dob', '/', 1, 0, 2)
print("age added")

"""Customer Since"""
loan_df = loan_obj.get_customer_since(loan_df, 'customer_to_bank', '/', 1, 0, 2)

"""Customer group - new/old"""
loan_df = loan_obj.group_customer(loan_df, 'customer_since_months', 'customer_rel_dur_segment')
print("grouped")

""""Zip code retrieval and distance mapping"""
dict_add={}

address = list(loan_df['address'].unique())

for add in address:
    dict_add[add]=loan_obj.zipcode_distance_retrieval(add,"IN")


loan_df = loan_obj.address_distance(loan_df,dict_add,"address")


"""Since we have added some new columns - we have to update our cols_list"""
loan_obj_cols = loan_obj.get_obj_col_list(loan_df)
loan_nonobj_cols = loan_obj.get_nonobj_col_list(loan_df)

loan_temp_df = loan_df.copy()
loan_temp_df.to_csv('./data/unprocessed/consolidated_for_comparison.csv', index=False)

print(loan_df.head())

"""Handling Missing Data"""
loan_missing_df = loan_obj.get_missing_df(loan_df)
loan_missing_df = loan_missing_df[loan_missing_df.percent_missing > 0]
print(loan_missing_df)

"""Handling Categorical Variables"""
loan_cat_col_uniques_dict = loan_obj.get_cat_cols_unique_val_dict(loan_df, loan_obj_cols)
print(loan_cat_col_uniques_dict)

loan_binary_cols_list = loan_cat_col_uniques_dict.get(2).split(",")
loan_onehot_col_list = loan_cat_col_uniques_dict.get(4).split(",")     #+ loan_cat_col_uniques_dict[6].split(",")

loan_df = loan_obj.convert_cat_cols_to_binary(loan_df, loan_binary_cols_list)
loan_df = loan_obj.convert_cat_cols_to_onehot(loan_df, loan_onehot_col_list)
# # customer_df = customer_obj.group_less_occurring_cat_vars(customer_df, customer_obj_cols)
print(loan_df.head())
print("Categorical data handling complete for Customer data")

"""OUTLIER DETECTION"""
loan_outlier_df = loan_obj.outlier_detection(loan_df[loan_nonobj_cols])
loan_outlier_df = loan_obj.outlier_details(loan_outlier_df)
print("Outliers")
print(loan_outlier_df)

"""" log scaling """
list_cols_scaled = ["applicant_income","coapplicant_income","loan_amt"]
loan_df = loan_obj.log_scaling(loan_df,list_cols_scaled)

"""CORRELATION ANALYSIS"""

"""Top Absolute Correlation"""
print("Top Absolute Correlation")
#print(loan_obj.get_top_abs_correlations(loan_df.iloc[:, 1:], 10))

"""Highly correlated columns -- exceeding 0.75"""
print("Suggested Highly Correlated Columns to be dropped")
print(loan_obj.drop_highly_corr_var(loan_df, 0.75))

loan_df['loan_availed'] = target_df['loan_availed']
print(loan_df.head())
loan_df.to_csv("./data/processed/loan_cross_sell_processed.csv", index=False)
print("--- %s seconds ---" % (time.time() - start_time))