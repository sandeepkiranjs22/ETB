import pandas as pd
import numpy as np
from dask.distributed import Client

import warnings
warnings.filterwarnings('ignore')

import time
import sys
sys.path.append('/home/sandeep/Desktop/BankBuddy/Reco-usecases/new_reco/preprocessing_pipeline/')

from DataPreprocess import *

customer_file = './data/unprocessed/customer_ins.csv'
travel_file = './data/unprocessed/travel_ins.csv'
motor_file = './data/unprocessed/motor_ins.csv'
med_file = './data/unprocessed/med_ins.csv'
home_file = './data/unprocessed/home_ins.csv'
prop_file = './data/unprocessed/property_ins.csv'


start_time = time.time()

customer_obj = DataPreprocess(customer_file)
customer_df = customer_obj.get_df()

travel_obj = DataPreprocess(travel_file)
travel_df = travel_obj.get_df()

motor_obj = DataPreprocess(motor_file)
motor_df = motor_obj.get_df()

med_obj = DataPreprocess(med_file)
med_df = med_obj.get_df()

home_obj = DataPreprocess(home_file)
home_df = home_obj.get_df()

prop_obj = DataPreprocess(prop_file)
prop_df = prop_obj.get_df()


"""Setup Dask Client"""
workers = 1
thr_per_worker = 4
process = False
memory = '2GB'
client = Client(n_workers=workers, threads_per_worker=thr_per_worker, processes=process, memory_limit=memory)

"""Format Column Names"""
customer_df.columns = customer_obj.format_column_name(customer_df)
travel_df.columns = travel_obj.format_column_name(travel_df)
motor_df.columns = motor_obj.format_column_name(motor_df)
med_df.columns = med_obj.format_column_name(med_df)
home_df.columns = home_obj.format_column_name(home_df)
prop_df.columns = prop_obj.format_column_name(prop_df)



"""Get list of obj/non-obj columns"""
customer_obj_cols = customer_obj.get_obj_col_list(customer_df)
customer_nonobj_cols = customer_obj.get_nonobj_col_list(customer_df)

travel_obj_cols = travel_obj.get_obj_col_list(travel_df)
travel_nonobj_cols = travel_obj.get_nonobj_col_list(travel_df)

motor_obj_cols = motor_obj.get_obj_col_list(motor_df)
motor_nonobj_cols = motor_obj.get_nonobj_col_list(motor_df)

med_obj_cols = med_obj.get_obj_col_list(med_df)
med_nonobj_cols = med_obj.get_nonobj_col_list(med_df)

home_obj_cols = home_obj.get_obj_col_list(home_df)
home_nonobj_cols = home_obj.get_nonobj_col_list(home_df)

prop_obj_cols = prop_obj.get_obj_col_list(prop_df)
prop_nonobj_cols = prop_obj.get_nonobj_col_list(prop_df)



"""Convert string (object datatypes) to lowercase"""
customer_df = customer_obj.convert_data_to_lowercase(customer_df, customer_obj_cols)
travel_df = travel_obj.convert_data_to_lowercase(travel_df, travel_obj_cols)
motor_df = motor_obj.convert_data_to_lowercase(motor_df, motor_obj_cols)
med_df = med_obj.convert_data_to_lowercase(med_df, med_obj_cols)
home_df = home_obj.convert_data_to_lowercase(home_df, home_obj_cols)
prop_df = prop_obj.convert_data_to_lowercase(prop_df, prop_obj_cols)


"""Check dtypes"""
customer_df_dtypes = customer_obj.check_dtype(customer_df)
travel_df_dtypes = travel_obj.check_dtype(travel_df)
motor_df_dtypes = motor_obj.check_dtype(motor_df)
med_df_dtypes = med_obj.check_dtype(med_df)
home_df_dtypes = home_obj.check_dtype(home_df)
prop_df_dtypes = prop_obj.check_dtype(prop_df)


print(customer_df_dtypes)
print()
print(travel_df_dtypes)
print()
print(motor_df_dtypes)
print()
print(med_df_dtypes)
print()
print(home_df_dtypes)
print()
print(prop_df_dtypes)
print()


customer_temp_df = customer_df.copy()

"""Get df with number of unique values column-wise"""
customer_unique_col_values = customer_obj.get_unique_col_values(customer_df)
travel_unique_col_values = travel_obj.get_unique_col_values(travel_df)
motor_unique_col_values = motor_obj.get_unique_col_values(motor_df)
med_unique_col_values = med_obj.get_unique_col_values(med_df)
home_unique_col_values = home_obj.get_unique_col_values(home_df)
prop_unique_col_values = prop_obj.get_unique_col_values(prop_df)

"""unnecessary_cols stores columns with number of unique values either equal to 1 or len(df)"""
customer_unnecessary_cols = customer_unique_col_values[customer_unique_col_values['num_unique_values'].isin([1, len(customer_df)])]
print(customer_unique_col_values)
print("customer columns to be dropped - ", customer_unnecessary_cols.column.unique())
travel_unnecessary_cols = travel_unique_col_values[travel_unique_col_values['num_unique_values'].isin([1, len(travel_df)])]
print(travel_unique_col_values)
print("travel columns to be dropped - ", travel_unnecessary_cols.column.unique())
motor_unnecessary_cols = motor_unique_col_values[motor_unique_col_values['num_unique_values'].isin([1, len(motor_df)])]
print(motor_unique_col_values)
print("motor columns to be dropped - ", motor_unnecessary_cols.column.unique())
med_unnecessary_cols = med_unique_col_values[med_unique_col_values['num_unique_values'].isin([1, len(med_df)])]
print(med_unique_col_values)
print("med columns to be dropped - ", med_unnecessary_cols.column.unique())
home_unnecessary_cols = home_unique_col_values[home_unique_col_values['num_unique_values'].isin([1, len(home_df)])]
print(home_unique_col_values)
print("home columns to be dropped - ", home_unnecessary_cols.column.unique())
prop_unnecessary_cols = prop_unique_col_values[prop_unique_col_values['num_unique_values'].isin([1, len(prop_df)])]
print(prop_unique_col_values)
print("prop columns to be dropped - ", prop_unnecessary_cols.column.unique())



"""Dropping the target columns temporarily so that they do not undergo any preprocesing"""

target_df = pd.DataFrame()
target_df['travel_addon'] = travel_df['travel_addon']
target_df['motor_addon'] = motor_df['motor_addon']
target_df['medical_addon'] = med_df['medical_addon']
target_df['home_addon'] = home_df['addon']
target_df['prop_addon'] = prop_df['addon']

travel_df.drop('travel_addon', axis=1, inplace=True)
motor_df.drop('motor_addon', axis=1, inplace=True)
med_df.drop('medical_addon', axis=1, inplace=True)
home_df.drop('addon', axis=1, inplace=True)
prop_df.drop('addon', axis=1, inplace=True)

"""CALCULATING DERIVED DATA"""

"""Age from DoB"""
customer_df = customer_obj.get_age_col(customer_df, 'dob', '/', 1, 0, 2)
print("age column added")

"""Customer Since"""
customer_df = customer_obj.get_customer_since(customer_df, 'customer_to_bank', '/', 1, 0, 2)

"""Customer group - new/old"""
customer_df = customer_obj.group_customer(customer_df, 'customer_since_months', 'customer_rel_dur_segment')
print("customer_grouped")

"""Population and city tier"""
customer_df = customer_obj.get_population_column(customer_df, 'city')
bins_dict = customer_obj.get_population_bins(customer_df, 'population')
bins = list(bins_dict.values())
bins = sorted(list(set(bins)))
print("Population bins - ", bins)

labels = customer_obj.get_labels(bins)
print(labels)
customer_df = customer_obj.group_city_from_population(customer_df, 'population', bins, labels)
print(customer_df.head())

travel_df.drop('destination', axis=1, inplace=True)
motor_df.drop(['incident_city','incident_location', 'policy_bind_date', 'incident_state', 'policy_state', 'incident_city', 'auto_model'], axis=1, inplace=True)
# incident_date, incident_city, auto_make, auto_model, auto_year
motor_df = motor_obj.get_months(motor_df, 'incident_date', '-', 2, 1, 0)
#motor_df = motor_obj.get_vehicle_age(motor_df, 'auto_year')
motor_df = motor_df.replace('?', np.NaN)
# print(motor_df.head())

motor_df = motor_obj.get_years_old(motor_df,'auto_year')

motor_df = motor_obj.policy_csl_func(motor_df,"policy_csl","/")

motor_df = motor_obj.net_amt(motor_df,"capital_gains","capital_loss")


motor_df = motor_obj.severity_ordinal(motor_df,'incident_severity')

motor_df = motor_obj.hour_segment(motor_df,'incident_hour_of_the_day')


motor_df = motor_obj.witness_bucket_segment(motor_df,'witnesses')

motor_df = motor_obj.auto_make_to_ordinal(motor_df,"auto_make","years_auto_old")



home_df.drop(['quote_date','cover_start','mta_date','police'],axis=1,inplace=True)
prop_df.drop(['quote_date','cover_start','mta_date','police'],axis=1,inplace=True)


"""Since we modified columns - we have to update our cols_list"""
customer_obj_cols = customer_obj.get_obj_col_list(customer_df)
customer_nonobj_cols = customer_obj.get_nonobj_col_list(customer_df)
travel_obj_cols = travel_obj.get_obj_col_list(travel_df)
travel_nonobj_cols = travel_obj.get_nonobj_col_list(travel_df)
motor_obj_cols = motor_obj.get_obj_col_list(motor_df)
motor_nonobj_cols = motor_obj.get_nonobj_col_list(motor_df)
med_obj_cols = med_obj.get_obj_col_list(med_df)
med_nonobj_cols = med_obj.get_nonobj_col_list(med_df)

home_obj_cols = home_obj.get_obj_col_list(home_df)
home_nonobj_cols = home_obj.get_nonobj_col_list(home_df)
prop_obj_cols = prop_obj.get_obj_col_list(prop_df)
prop_nonobj_cols = prop_obj.get_nonobj_col_list(prop_df)


master_temp_df = pd.merge(travel_df, motor_df, on='customer_id', how='outer')
master_temp_df = pd.merge(master_temp_df, med_df, on='customer_id', how='outer')
master_temp_df = pd.merge(master_temp_df, home_df, on='customer_id', how='outer')
master_temp_df = pd.merge(master_temp_df, prop_df, on='customer_id', how='outer')
master_temp_df = pd.merge(master_temp_df, customer_temp_df, on='customer_id', how='outer')
master_temp_df.to_csv('./data/unprocessed/consolidated_for_comparison.csv', index=False)


"""Handling Missing Data"""
customer_missing_df = customer_obj.get_missing_df(customer_df)
customer_missing_df = customer_missing_df[customer_missing_df.percent_missing > 0]
# print(customer_missing_df)

travel_missing_df = travel_obj.get_missing_df(travel_df)
travel_missing_df = travel_missing_df[travel_missing_df.percent_missing > 0]
# print(customer_missing_df)

motor_missing_df = motor_obj.get_missing_df(motor_df)
motor_missing_df = motor_missing_df[motor_missing_df.percent_missing > 0]
# print(customer_missing_df)

med_missing_df = med_obj.get_missing_df(med_df)
med_missing_df = med_missing_df[med_missing_df.percent_missing > 0]
# print(customer_missing_df)

home_missing_df = home_obj.get_missing_df(home_df)
home_missing_df = home_missing_df[home_missing_df.percent_missing > 0]
# print(customer_missing_df)

prop_missing_df = prop_obj.get_missing_df(prop_df)
prop_missing_df = prop_missing_df[prop_missing_df.percent_missing > 0]
# print(customer_missing_df)

# customer_df = customer_obj.missing_data_handle(customer_df, customer_missing_df, customer_nonobj_cols)
# print(customer_df.info())

"""No missing values reported"""

"""Handling Categorical Variables"""
customer_cat_col_uniques_dict = customer_obj.get_cat_cols_unique_val_dict(customer_df, customer_obj_cols)
print(customer_cat_col_uniques_dict)

travel_cat_col_uniques_dict = travel_obj.get_cat_cols_unique_val_dict(travel_df, travel_obj_cols)
print(travel_cat_col_uniques_dict)

motor_cat_col_uniques_dict = motor_obj.get_cat_cols_unique_val_dict(motor_df, motor_obj_cols)
print(motor_cat_col_uniques_dict)

med_cat_col_uniques_dict = med_obj.get_cat_cols_unique_val_dict(med_df, med_obj_cols)
print(med_cat_col_uniques_dict)


home_cat_col_uniques_dict = home_obj.get_cat_cols_unique_val_dict(home_df, home_obj_cols)
print(home_cat_col_uniques_dict)

prop_cat_col_uniques_dict = prop_obj.get_cat_cols_unique_val_dict(prop_df, prop_obj_cols)
print(prop_cat_col_uniques_dict)



customer_binary_cols_list = customer_cat_col_uniques_dict[2].split(",")
customer_onehot_col_list = customer_cat_col_uniques_dict[14].split(",")
customer_df = customer_obj.convert_cat_cols_to_binary(customer_df, customer_binary_cols_list)
customer_df = customer_obj.convert_cat_cols_to_onehot(customer_df, customer_onehot_col_list)

travel_binary_cols_list = travel_cat_col_uniques_dict[2].split(",")
travel_onehot_col_list = travel_cat_col_uniques_dict[16].split(",")
travel_df = travel_obj.convert_cat_cols_to_binary(travel_df, travel_binary_cols_list)
travel_df = travel_obj.convert_cat_cols_to_onehot(travel_df, travel_onehot_col_list)

# motor_binary_cols_list = motor_cat_col_uniques_dict.get(3).split(",")
motor_onehot_col_list = motor_cat_col_uniques_dict[3].split(",") + motor_cat_col_uniques_dict[4].split(",") + \
                        motor_cat_col_uniques_dict[5].split(",")
# motor_df = motor_obj.convert_cat_cols_to_binary(motor_df, motor_binary_cols_list)
motor_df = motor_obj.convert_cat_cols_to_onehot(motor_df, motor_onehot_col_list)

med_binary_cols_list = med_cat_col_uniques_dict.get(2).split(",")
# med_onehot_col_list = med_cat_col_uniques_dict[3].split(",") + med_cat_col_uniques_dict[12].split(",")
med_df = med_obj.convert_cat_cols_to_binary(med_df, med_binary_cols_list)
# med_df = med_obj.convert_cat_cols_to_onehot(med_df, med_onehot_col_list)


# home_binary_cols_list = home_cat_col_uniques_dict.get(3).split(",")
home_onehot_col_list = home_cat_col_uniques_dict[2].split(",") + home_cat_col_uniques_dict[3].split(",") + \
                        home_cat_col_uniques_dict[4].split(",") + home_cat_col_uniques_dict[6].split(",")
# home_df = home_obj.convert_cat_cols_to_binary(home_df, home_binary_cols_list)
home_df = home_obj.convert_cat_cols_to_onehot(home_df, home_onehot_col_list)

prop_binary_cols_list = prop_cat_col_uniques_dict.get(2).split(",") + prop_cat_col_uniques_dict.get(3).split(",") +prop_cat_col_uniques_dict.get(4).split(",")
# prop_onehot_col_list = prop_cat_col_uniques_dict[3].split(",") + prop_cat_col_uniques_dict[12].split(",")
prop_df = prop_obj.convert_cat_cols_to_binary(prop_df, prop_binary_cols_list)
# prop_df = prop_obj.convert_cat_cols_to_onehot(prop_df, prop_onehot_col_list)


print("Missing and Categorical Handling done.")


"""OUTLIER DETECTION"""
customer_outlier_df = customer_obj.outlier_detection(customer_df[customer_nonobj_cols])
customer_outlier_df = customer_obj.outlier_details(customer_outlier_df)
print("Customer Outliers")
print(customer_outlier_df, end='\n\n')

travel_outlier_df = travel_obj.outlier_detection(travel_df[travel_nonobj_cols])
travel_outlier_df = travel_obj.outlier_details(travel_outlier_df)
print("Travel Outliers")
print(travel_outlier_df, end='\n\n')

motor_outlier_df = motor_obj.outlier_detection(motor_df[motor_nonobj_cols])
motor_outlier_df = motor_obj.outlier_details(motor_outlier_df)
print("Motor Outliers")
print(motor_outlier_df, end='\n\n')

med_outlier_df = med_obj.outlier_detection(med_df[med_nonobj_cols])
med_outlier_df = med_obj.outlier_details(med_outlier_df)
print("Med Outliers")
print(med_outlier_df, end='\n\n')

home_outlier_df = home_obj.outlier_detection(home_df[home_nonobj_cols])
home_outlier_df = home_obj.outlier_details(home_outlier_df)
print("home Outliers")
print(home_outlier_df, end='\n\n')

prop_outlier_df = prop_obj.outlier_detection(prop_df[prop_nonobj_cols])
prop_outlier_df = prop_obj.outlier_details(prop_outlier_df)
print("prop Outliers")
print(prop_outlier_df, end='\n\n')


"""CORRELATION ANALYSIS"""

"""Top Absolute Correlation"""
print("Top Absolute Correlation")
# print(customer_obj.get_top_abs_correlations(customer_df.iloc[:, 1:], 10), end='\n\n')
# print(travel_obj.get_top_abs_correlations(travel_df.iloc[:, 1:], 10), end='\n\n')
# print(motor_obj.get_top_abs_correlations(motor_df.iloc[:, 1:], 10), end='\n\n')
# print(med_obj.get_top_abs_correlations(med_df.iloc[:, 1:], 10), end='\n\n')
# print(home_obj.get_top_abs_correlations(home_df.iloc[:, 1:], 10), end='\n\n')
# print(prop_obj.get_top_abs_correlations(prop_df.iloc[:, 1:], 10), end='\n\n')


"""Highly correlated columns -- exceeding 0.75"""
print("Suggested Highly Correlated Columns to be dropped")
print(customer_obj.drop_highly_corr_var(customer_df, 0.75), end='\n\n')
print(travel_obj.drop_highly_corr_var(travel_df, 0.75), end='\n\n')
print(motor_obj.drop_highly_corr_var(motor_df, 0.75), end='\n\n')
print(med_obj.drop_highly_corr_var(med_df, 0.75), end='\n\n')
print(home_obj.drop_highly_corr_var(home_df, 0.75), end='\n\n')
print(prop_obj.drop_highly_corr_var(prop_df, 0.75), end='\n\n')


"""Merging dfs"""
travel_df = pd.merge(customer_df, travel_df, on="customer_id")
motor_df = pd.merge(customer_df, motor_df, on="customer_id")
med_df = pd.merge(customer_df, med_df, on="customer_id")
home_df = pd.merge(customer_df, home_df, on="customer_id")
prop_df = pd.merge(customer_df, prop_df, on="customer_id")

travel_df['travel_addon'] = target_df['travel_addon']
motor_df['motor_addon'] = target_df['motor_addon']
med_df['medical_addon'] = target_df['medical_addon']
home_df['home_addon'] = target_df['home_addon']
prop_df['prop_addon'] = target_df['prop_addon']



"""Storing Processed CSVs"""
customer_df.to_csv("./data/processed/customer_processed.csv", index=False)
travel_df.to_csv("./data/processed/travel_processed.csv", index=False)
motor_df.to_csv("./data/processed/motor_processed.csv", index=False)
med_df.to_csv("./data/processed/med_processed.csv", index=False)
home_df.to_csv("./data/processed/home_processed.csv", index=False)
prop_df.to_csv("./data/processed/prop_processed.csv", index=False)

print("--- %s seconds ---" % (time.time() - start_time))
