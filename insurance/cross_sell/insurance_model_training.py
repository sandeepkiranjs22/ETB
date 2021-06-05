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

    insurance_file = './data/processed/insurance_cross_sell_processed.csv'
    insurance_obj = ModelTraining(insurance_file)
    insurance_df = insurance_obj.get_df()
    insurance_ddf = insurance_obj.convert_df_to_ddf(insurance_df)
    print(len(insurance_df))

    customer_file = './data/unprocessed/consolidated_for_comparison.csv'
    customer_obj = ModelTraining(customer_file)
    customer_df = customer_obj.get_df()

    """target column"""
    y_insurance = insurance_obj.label_encoding(insurance_df['insurance'])
    y_insurance_dask = insurance_obj.convert_target_col_to_dask_df(y_insurance)

    """Saving the encoded column"""
    insurance_dict = {}

    for i in range(len(y_insurance)):
        insurance_dict[y_insurance[i]] = insurance_df['insurance'][i]
    print(insurance_dict)

    insurance_ddf = insurance_ddf.drop('insurance', axis=1)

    """Train Test Split"""
    X_train_insurance, X_test_insurance, y_train_insurance, y_test_insurance = insurance_obj.split_dataset(insurance_ddf, y_insurance_dask)

    """MODEL TRAINING"""

    """insurance recommendation training"""
    insurance_cols = insurance_obj.get_nonobj_col_list(insurance_df)
    params = {'objective': 'multi:softprob', 'nround': 1000, 'num_class': len(insurance_dict), 'max_depth': 6}
    bst_insurance = insurance_obj.dxgb_train(client, params, X_train_insurance[insurance_cols], y_train_insurance)
    print("insurance cross sell model trained")
    print()

    """Predictions"""
    insurance_pred = xgb.DMatrix(X_test_insurance[insurance_cols])
    insurance_pred.feature_names = insurance_cols
    insurance_results = (bst_insurance.predict(insurance_pred))

    """Get back recommendations"""
    insurance_actual = list(y_test_insurance[0].compute())
    insurance_top_recommendation = []
    insurance_top3_recommendations = []
    for result in insurance_results:
        insurance_top_recommendation.append(insurance_obj.decode_softmax_to_label(result, insurance_dict, 1))
        insurance_top3_recommendations.append(insurance_obj.decode_softmax_to_label(result, insurance_dict, 3))

    # print(len(insurance_actual))
    # print(len(insurance_top_recommendation))
    # print(len(insurance_top3_recommendations))

    insurance_model_accuracy = accuracy_score(insurance_actual, insurance_top_recommendation)
    insurance_model_accuracy = round(insurance_model_accuracy, 4) * 100
    print("insurance Cross Sell Model Accuracy - ", insurance_model_accuracy)
    # print(insurance_top_recommendation[0])
    print(insurance_top3_recommendations[0])

    customer_df = customer_df[customer_df['customer_id'].isin(X_test_insurance['customer_id'])]
    customer_df['insurance_cross_sell_top3_reco'] = insurance_top3_recommendations
    print(customer_df.head())
    customer_df.to_csv('./reco_results/insurance_cross_sell_test_set_recommendations.csv', index=False)
    customer_df.to_csv('../model_outputs/reco_results/insurance_cross_sell_test_set_recommendations.csv', index=False)