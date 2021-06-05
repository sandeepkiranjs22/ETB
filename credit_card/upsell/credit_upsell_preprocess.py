import pandas as pd
import numpy as np
from dask.distributed import Client

import warnings

warnings.filterwarnings('ignore')

import time
import sys

sys.path.append('/home/sandeep/Desktop/BankBuddy/Reco-usecases/new_reco/preprocessing_pipeline/')

from DataPreprocess import *


credit_upsell_file = './data/unprocessed/credit_upsell.csv'
credit_obj = DataPreprocess(credit_upsell_file)
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

"""Printing a Sample"""
print(credit_df[credit_df.customer_id == 'c5282'])
print()
print(credit_df.exp_type.unique())
credit_df.drop(['city_transaction_made', 'date_transaction_made'], axis=1, inplace=True)


# gender card_type occupation dob income customer_to_bank city marital_status
def group_customer_rows(data):
    #gender = data.gender.mode()[0]
    card_type = data.card_type.mode()[0]
    occupation = data.occupation.mode()[0]
    dob = data.dob.mode()[0]
    income = data.income.mode()[0]
    customer_to_bank = data.customer_to_bank.mode()[0]
    city = data.city.mode()[0]
    marital_status = data.marital_status.mode()[0]
    total_amt = data.amount.sum()
    total_trans = data.amount.count()
    bill_amt = (data.loc[data.exp_type == 'bills']['amount'].sum())
    bill_trans = (data.loc[data.exp_type == 'bills']['amount'].count())
    food_amt = (data.loc[data.exp_type == 'food']['amount'].sum()) + (data.loc[data.exp_type == 'grocery']['amount'].sum())
    food_trans = (data.loc[data.exp_type == 'food']['amount'].count()) + (data.loc[data.exp_type == 'grocery']['amount'].count())
    entertainment_amt = (data.loc[data.exp_type == 'entertainment']['amount'].sum())
    entertainment_trans = (data.loc[data.exp_type == 'entertainment']['amount'].count())
    fuel_amt = (data.loc[data.exp_type == 'fuel']['amount'].sum())
    fuel_trans = (data.loc[data.exp_type == 'fuel']['amount'].count())
    travel_amt = (data.loc[data.exp_type == 'travel']['amount'].sum())
    travel_trans = (data.loc[data.exp_type == 'travel']['amount'].count())

    return pd.Series([card_type, occupation, dob, income, customer_to_bank, city, marital_status, travel_amt,
                      travel_trans, fuel_amt, fuel_trans, entertainment_amt, entertainment_trans,
                      food_amt, food_trans, bill_amt, bill_trans, total_amt, total_trans])



credit_df2 = credit_df.groupby('customer_id').apply(group_customer_rows).reset_index()
credit_df2.columns = ['customer_id', 'card_type', 'occupation', 'dob', 'income', 'customer_to_bank', 'city',
                      'marital_status', 'travel_amt', 'travel_trans', 'fuel_amt', 'fuel_trans', 'entertainment_amt', 'entertainment_trans',
                      'food_amt', 'food_trans', 'bill_amt', 'bill_trans', 'total_amt', 'total_trans']
print(credit_df2.head())

"""Get list of obj/non-obj columns"""
credit_obj_cols = credit_obj.get_obj_col_list(credit_df2)
credit_nonobj_cols = credit_obj.get_nonobj_col_list(credit_df2)



"""Get df with number of unique values column-wise"""
credit_unique_col_values = credit_obj.get_unique_col_values(credit_df2)
print(credit_unique_col_values)

"""unnecessary_cols stores columns with number of unique values either equal to 1 or len(df)"""
credit_unnecessary_cols = credit_unique_col_values[credit_unique_col_values['num_unique_values'].isin([1, len(credit_df2)])]
print("columns to be dropped - ", credit_unnecessary_cols.column.unique())

target_df = pd.DataFrame()
target_df['card_type'] = credit_df2['card_type']
credit_df2.drop('card_type', axis=1, inplace=True)

"""Age from DoB"""
credit_df2 = credit_obj.get_age_col(credit_df2, 'dob', '/', 1, 0, 2)
print("age added")

"""Population and city tier"""
credit_df2 = credit_obj.get_population_column(credit_df2, 'city')
bins_dict = credit_obj.get_population_bins(credit_df2, 'population')
bins = list(bins_dict.values())
bins = sorted(list(set(bins)))
print("Population bins - ", bins)

labels = credit_obj.get_labels(bins)
print(labels)
credit_df2 = credit_obj.group_city_from_population(credit_df2, 'population', bins, labels)

"""Customer Since"""
credit_df2 = credit_obj.get_customer_since(credit_df2, 'customer_to_bank', '/', 1, 0, 2)

"""Customer group - new/old"""
credit_df2 = credit_obj.group_customer(credit_df2, 'customer_since_months', 'customer_rel_dur_segment')

credit_obj_cols = credit_obj.get_obj_col_list(credit_df2)
credit_nonobj_cols = credit_obj.get_nonobj_col_list(credit_df2)


temp_credit_df = credit_df2.copy()
temp_credit_df.to_csv('./data/unprocessed/consolidated_for_comparison.csv', index=False)

"""Handling Missing Data"""
credit_missing_df = credit_obj.get_missing_df(credit_df2)
credit_missing_df = credit_missing_df[credit_missing_df.percent_missing > 0]


"""Handling Categorical Variables"""
credit_cat_col_uniques_dict = credit_obj.get_cat_cols_unique_val_dict(credit_df2, credit_obj_cols)

credit_binary_cols_list = credit_cat_col_uniques_dict[2].split(",")
credit_onehot_col_list = credit_cat_col_uniques_dict[5].split(",")

credit_df2 = credit_obj.convert_cat_cols_to_binary(credit_df2, credit_binary_cols_list)
credit_df2 = credit_obj.convert_cat_cols_to_onehot(credit_df2, credit_onehot_col_list)
# customer_df = customer_obj.group_less_occurring_cat_vars(customer_df, customer_obj_cols)
print(credit_df2.head())
print(credit_df2.columns)

"""customer_rel_dur_segment - for this particular case - single value for all rows"""
credit_df2.drop('customer_rel_dur_segment', axis=1, inplace=True)
print("Categorical data handling complete for Credit Cross Sell Data.")

"""OUTLIER DETECTION"""

credit_outlier_df = credit_obj.outlier_detection(credit_df2[credit_nonobj_cols])
credit_outlier_df = credit_obj.outlier_details(credit_outlier_df)
print("Outliers")
print(credit_outlier_df)

"""" log scaling """
list_cols_scaled = ["income"]
credit_df2 = credit_obj.log_scaling(credit_df2,list_cols_scaled)

"""CORRELATION ANALYSIS"""

"""Top Absolute Correlation"""
print("Top Absolute Correlation")
print(credit_obj.get_top_abs_correlations(credit_df2.iloc[:, 1:], 10))

"""Highly correlated columns -- exceeding 0.75"""
print("Suggested Highly Correlated Columns to be dropped")
print(credit_obj.drop_highly_corr_var(credit_df2, 0.75))

credit_df2['card_type'] = target_df['card_type']
print(credit_df2.tail())

"""Storing Processed CSVs"""
credit_df2.to_csv("./data/processed/credit_upsell_processed.csv", index=False)