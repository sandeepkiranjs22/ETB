from dask.distributed import Client
from sklearn.metrics import accuracy_score
import xgboost as xgb

import pandas as pd
pd.set_option('display.max_rows',1000)
pd.set_option('display.max_columns',1000)

import sys
sys.path.append('/home/sandeep/Desktop/BankBuddy/Reco-usecases/new_reco/preprocessing_pipeline/')
sys.path.append('/home/sandeep/Desktop/BankBuddy/Reco-usecases/new_reco/model_training/')
from DataPreprocess import *
from ModelTraining import *


"""Setup Dask Client"""
workers = 1
thr_per_worker = 4
memory = '2GB'

if __name__ == '__main__':
	client = Client(n_workers=workers, threads_per_worker=thr_per_worker, memory_limit=memory)

	credit_cross_file = './data/processed/credit_cross_sell_processed.csv'
	credit_obj = ModelTraining(credit_cross_file)
	credit_df = credit_obj.get_df()
	credit_ddf = credit_obj.convert_df_to_ddf(credit_df)
	print(len(credit_df))

	customer_file = './data/unprocessed/consolidated_for_comparison.csv'
	customer_obj = ModelTraining(customer_file)
	customer_df = customer_obj.get_df()

	"""target column"""
	y_credit = credit_obj.label_encoding(credit_df['card_type'])
	y_credit_dask = credit_obj.convert_target_col_to_dask_df(y_credit)

	"""Saving the encoded column"""
	credit_dict = {}

	for i in range(len(y_credit)):
		credit_dict[y_credit[i]] = credit_df['card_type'][i]

	print(credit_dict)

	credit_ddf = credit_ddf.drop('card_type', axis=1)

	"""Train Test Split"""
	X_train_credit, X_test_credit, y_train_credit, y_test_credit = credit_obj.split_dataset(credit_ddf, y_credit_dask)

	"""MODEL TRAINING"""

	"""recommendation training"""
	credit_cols = credit_obj.get_nonobj_col_list(credit_df)
	params = {'objective': 'multi:softprob', 'nround': 1000, 'num_class': len(credit_dict), 'max_depth': 6}
	bst_credit = credit_obj.dxgb_train(client, params, X_train_credit[credit_cols], y_train_credit)
	print("Credit cross sell model trained")
	print()

	"""Predictions"""
	credit_pred = xgb.DMatrix(X_test_credit[credit_cols].head(len(X_test_credit)))
	credit_pred.feature_names = credit_cols
	credit_results = (bst_credit.predict(credit_pred))

	"""Get back recommendations"""
	credit_actual = list(y_test_credit[0])
	credit_top_recommendation = []
	credit_top3_recommendations = []

	for result in credit_results:
		credit_top_recommendation.append(credit_obj.decode_softmax_to_label(result, credit_dict, 1))
		credit_top3_recommendations.append(credit_obj.decode_softmax_to_label(result, credit_dict, 3))

	credit_model_accuracy = accuracy_score(credit_actual, credit_top_recommendation)
	credit_model_accuracy = round(credit_model_accuracy, 4) * 100
	print("Credit Cross Sell Model Accuracy - ", credit_model_accuracy)
	# print(loan_top_recommendation[0])
	print(credit_top3_recommendations[0])

	customer_df = customer_df[customer_df['customer_id'].isin(X_test_credit['customer_id'])]
	customer_df['credit_cross_sell_top3_reco'] = credit_top3_recommendations
	print(customer_df.head())
	customer_df.to_csv('./reco_results/credit_cross_sell_test_set_recommendations.csv', index=False)

	customer_df.to_csv('../model_outputs/reco_results/credit_cross_sell_test_set_recommendations.csv', index=False)
