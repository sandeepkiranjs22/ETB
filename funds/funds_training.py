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

    fund_file = './data/processed/fund_processed.csv'
    fund_obj = ModelTraining(fund_file)
    fund_df = fund_obj.get_df()
    fund_ddf = fund_obj.convert_df_to_ddf(fund_df)
    print(len(fund_df))

    customer_file = './data/unprocessed/consolidated_for_comparison.csv'
    customer_obj = ModelTraining(customer_file)
    customer_df = customer_obj.get_df()

    """target column"""
    y_fund = fund_obj.label_encoding(fund_df['fund_type'])
    y_fund_dask = fund_obj.convert_target_col_to_dask_df(y_fund)

    """Saving the encoded column"""
    fund_dict = {}

    for i in range(len(y_fund)):
        fund_dict[y_fund[i]] = fund_df['fund_type'][i]
    print(fund_dict)

    fund_ddf = fund_ddf.drop('fund_type', axis=1)

    """Train Test Split"""
    X_train_fund, X_test_fund, y_train_fund, y_test_fund = fund_obj.split_dataset(fund_ddf, y_fund_dask)

    """MODEL TRAINING"""

    """fund recommendation training"""
    fund_cols = fund_obj.get_nonobj_col_list(fund_df)
    params = {'objective': 'multi:softprob', 'nround': 1000, 'num_class': len(fund_dict), 'max_depth': 6}
    bst_fund = fund_obj.dxgb_train(client, params, X_train_fund[fund_cols], y_train_fund)
    print("fund cross sell model trained")
    print()

    """Predictions"""
    fund_pred = xgb.DMatrix(X_test_fund[fund_cols])
    fund_pred.feature_names = fund_cols
    fund_results = (bst_fund.predict(fund_pred))

    """Get back recommendations"""
    fund_actual = list(y_test_fund[0].compute())
    fund_top_recommendation = []
    fund_top3_recommendations = []
    for result in fund_results:
        fund_top_recommendation.append(fund_obj.decode_softmax_to_label(result, fund_dict, 1))
        fund_top3_recommendations.append(fund_obj.decode_softmax_to_label(result, fund_dict, 3))

    # print(len(fund_actual))
    # print(len(fund_top_recommendation))
    # print(len(fund_top3_recommendations))

    fund_model_accuracy = accuracy_score(fund_actual, fund_top_recommendation)
    fund_model_accuracy = round(fund_model_accuracy, 4) * 100
    print("fund  Model Accuracy - ", fund_model_accuracy)
    # print(fund_top_recommendation[0])
    print(fund_top3_recommendations[0])

    customer_df = customer_df[customer_df['customer_id'].isin(X_test_fund['customer_id'])]
    customer_df['fund_top3_reco'] = fund_top3_recommendations
    print(customer_df.head())
    customer_df.to_csv('./reco_results/fund_test_set_recommendations.csv', index=False)