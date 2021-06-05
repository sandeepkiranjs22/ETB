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

termdep_file = './data/unprocessed/term_dep.csv'

termdep_obj = DataPreprocess(termdep_file)
termdep_df = termdep_obj.get_df()

"""Setup Dask Client"""
workers = 1
thr_per_worker = 4
process = False
memory = '2GB'
client = Client(n_workers=workers, threads_per_worker=thr_per_worker, processes=process, memory_limit=memory)



"""Format Column Names"""
termdep_df.columns = termdep_obj.format_column_name(termdep_df)

"""Check dtypes"""
termdep_df_dtypes = termdep_obj.check_dtype(termdep_df)
print(termdep_df_dtypes)

""" Drop gender column"""
termdep_df = termdep_obj.drop_column(termdep_df,"gender")

"""Get list of obj/non-obj columns"""
termdep_obj_cols = termdep_obj.get_obj_col_list(termdep_df)
termdep_nonobj_cols = termdep_obj.get_nonobj_col_list(termdep_df)

"""Convert string (object datatypes) to lowercase"""
termdep_df = termdep_obj.convert_data_to_lowercase(termdep_df, termdep_obj_cols)

"""Get df with number of unique values column-wise"""
termdep_unique_col_values = termdep_obj.get_unique_col_values(termdep_df)
print(termdep_unique_col_values)
"""unnecessary_cols stores columns with number of unique values either equal to 1 or len(df)"""
termdep_unnecessary_cols = termdep_unique_col_values[termdep_unique_col_values['num_unique_values'].isin([1, len(termdep_df)])]
print("columns to be dropped - ", termdep_unnecessary_cols.column.unique())

target_df = pd.DataFrame()
target_df['term_dep_avl'] = termdep_df['term_dep_avl']
termdep_df.drop('term_dep_avl', axis=1, inplace=True)

"""Update list of obj/non-obj columns"""
termdep_obj_cols = termdep_obj.get_obj_col_list(termdep_df)
termdep_nonobj_cols = termdep_obj.get_nonobj_col_list(termdep_df)

"""CALCULATING DERIVED DATA"""

"""Age from DoB"""
termdep_df = termdep_obj.get_age_col(termdep_df, 'dob', '/', 1, 0, 2)
print("age added")

"""Customer Since"""
termdep_df = termdep_obj.get_customer_since(termdep_df, 'customer_to_bank', '/', 1, 0, 2)

"""Customer group - new/old"""
termdep_df = termdep_obj.group_customer(termdep_df, 'customer_since_months', 'customer_rel_dur_segment')
print("grouped")

"""Population and city tier"""
termdep_df = termdep_obj.get_population_column(termdep_df, 'city')
bins_dict = termdep_obj.get_population_bins(termdep_df, 'population')
bins = list(bins_dict.values())
bins = sorted(list(set(bins)))
print("Population bins - ", bins)

labels = termdep_obj.get_labels(bins)
print(labels)
termdep_df= termdep_obj.group_city_from_population(termdep_df, 'population', bins, labels)
print(termdep_df.head())



"""Since we have added some new columns - we have to update our cols_list"""
termdep_obj_cols = termdep_obj.get_obj_col_list(termdep_df)
termdep_nonobj_cols = termdep_obj.get_nonobj_col_list(termdep_df)

termdep_temp_df = termdep_df.copy()
termdep_temp_df.to_csv('./data/unprocessed/consolidated_for_comparison.csv', index=False)

print(termdep_df.head())

"""Handling Missing Data"""
termdep_missing_df = termdep_obj.get_missing_df(termdep_df)
termdep_missing_df = termdep_missing_df[termdep_missing_df.percent_missing > 0]
print(termdep_missing_df)

"""Handling Categorical Variables"""
termdep_cat_col_uniques_dict = termdep_obj.get_cat_cols_unique_val_dict(termdep_df, termdep_obj_cols)
print(termdep_cat_col_uniques_dict)

termdep_binary_cols_list = termdep_cat_col_uniques_dict.get(2).split(",")
termdep_onehot_col_list = termdep_cat_col_uniques_dict.get(5).split(",")     #+ termdep_cat_col_uniques_dict[6].split(",")

termdep_df = termdep_obj.convert_cat_cols_to_binary(termdep_df, termdep_binary_cols_list)
termdep_df = termdep_obj.convert_cat_cols_to_onehot(termdep_df, termdep_onehot_col_list)
# # customer_df = customer_obj.group_less_occurring_cat_vars(customer_df, customer_obj_cols)
print(termdep_df.head())
print("Categorical data handling complete for Customer data")

"""OUTLIER DETECTION"""
termdep_outlier_df = termdep_obj.outlier_detection(termdep_df[termdep_nonobj_cols])
termdep_outlier_df = termdep_obj.outlier_details(termdep_outlier_df)
print("Outliers")
print(termdep_outlier_df)

"""" log scaling """
list_cols_scaled = ["income"]
termdep_df = termdep_obj.log_scaling(termdep_df,list_cols_scaled)

"""CORRELATION ANALYSIS"""

"""Top Absolute Correlation"""
print("Top Absolute Correlation")
#print(termdep_obj.get_top_abs_correlations(termdep_df.iloc[:, 1:], 10))

"""Highly correlated columns -- exceeding 0.75"""
print("Suggested Highly Correlated Columns to be dropped")
print(termdep_obj.drop_highly_corr_var(termdep_df, 0.75))

termdep_df['term_dep_avl'] = target_df['term_dep_avl']

print(termdep_df.head())
termdep_df.to_csv("./data/processed/termdep_processed.csv", index=False)
print("--- %s seconds ---" % (time.time() - start_time))