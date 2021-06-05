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

    loan_file = './data/processed/loan_cross_sell_processed.csv'
    loan_obj = ModelTraining(loan_file)
    loan_df = loan_obj.get_df()
    loan_ddf = loan_obj.convert_df_to_ddf(loan_df)
    print(len(loan_df))

    customer_file = './data/unprocessed/consolidated_for_comparison.csv'
    customer_obj = ModelTraining(customer_file)
    customer_df = customer_obj.get_df()

    """target column"""
    y_loan = loan_obj.label_encoding(loan_df['loan_availed'])
    y_loan_dask = loan_obj.convert_target_col_to_dask_df(y_loan)

    """Saving the encoded column"""
    loan_dict = {}

    for i in range(len(y_loan)):
        loan_dict[y_loan[i]] = loan_df['loan_availed'][i]
    print(loan_dict)

    loan_ddf = loan_ddf.drop('loan_availed', axis=1)

    """Train Test Split"""
    X_train_loan, X_test_loan, y_train_loan, y_test_loan = loan_obj.split_dataset(loan_ddf, y_loan_dask)

    """MODEL TRAINING"""

    """loan recommendation training"""
    loan_cols = loan_obj.get_nonobj_col_list(loan_df)
    params = {'objective': 'multi:softprob', 'nround': 1000, 'num_class': len(loan_dict), 'max_depth': 6}
    bst_loan = loan_obj.dxgb_train(client, params, X_train_loan[loan_cols], y_train_loan)
    print("loan cross sell model trained")
    print()

    """Predictions"""
    loan_pred = xgb.DMatrix(X_test_loan[loan_cols])
    loan_pred.feature_names = loan_cols
    loan_results = (bst_loan.predict(loan_pred))

    """Get back recommendations"""
    loan_actual = list(y_test_loan[0].compute())
    loan_top_recommendation = []
    loan_top3_recommendations = []
    for result in loan_results:
        loan_top_recommendation.append(loan_obj.decode_softmax_to_label(result, loan_dict, 1))
        loan_top3_recommendations.append(loan_obj.decode_softmax_to_label(result, loan_dict, 3))

    # print(len(loan_actual))
    # print(len(loan_top_recommendation))
    # print(len(loan_top3_recommendations))

    loan_model_accuracy = accuracy_score(loan_actual, loan_top_recommendation)
    loan_model_accuracy = round(loan_model_accuracy, 4) * 100
    print("Loan Cross Sell Model Accuracy - ", loan_model_accuracy)
    # print(loan_top_recommendation[0])
    print(loan_top3_recommendations[0])

    customer_df = customer_df[customer_df['customer_id'].isin(X_test_loan['customer_id'])]
    customer_df['loan_cross_sell_top3_reco'] = loan_top3_recommendations
    print(customer_df.head())
    customer_df.to_csv('./reco_results/loan_cross_sell_test_set_recommendations.csv', index=False)