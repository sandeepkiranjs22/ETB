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

    savings_file = './data/processed/savings_processed.csv'
    savings_obj = ModelTraining(savings_file)
    savings_df = savings_obj.get_df()
    savings_ddf = savings_obj.convert_df_to_ddf(savings_df)
    print(len(savings_df))

    customer_file = './data/unprocessed/consolidated_for_comparison.csv'
    customer_obj = ModelTraining(customer_file)
    customer_df = customer_obj.get_df()

    """target column"""
    y_savings = savings_obj.label_encoding(savings_df['savings_account_tier'])
    y_savings_dask = savings_obj.convert_target_col_to_dask_df(y_savings)

    """Saving the encoded column"""
    savings_dict = {}

    for i in range(len(y_savings)):
        savings_dict[y_savings[i]] = savings_df['savings_account_tier'][i]
    print(savings_dict)

    savings_ddf = savings_ddf.drop('savings_account_tier', axis=1)

    """Train Test Split"""
    X_train_savings, X_test_savings, y_train_savings, y_test_savings = savings_obj.split_dataset(savings_ddf, y_savings_dask)

    """MODEL TRAINING"""

    """savings recommendation training"""
    savings_cols = savings_obj.get_nonobj_col_list(savings_df)
    params = {'objective': 'multi:softprob', 'nround': 1000, 'num_class': len(savings_dict), 'max_depth': 6}
    bst_savings = savings_obj.dxgb_train(client, params, X_train_savings[savings_cols], y_train_savings)
    print("savings cross sell model trained")
    print()

    """Predictions"""
    savings_pred = xgb.DMatrix(X_test_savings[savings_cols])
    savings_pred.feature_names = savings_cols
    savings_results = (bst_savings.predict(savings_pred))

    """Get back recommendations"""
    savings_actual = list(y_test_savings[0].compute())
    savings_top_recommendation = []
    savings_top3_recommendations = []
    for result in savings_results:
        savings_top_recommendation.append(savings_obj.decode_softmax_to_label(result, savings_dict, 1))
        savings_top3_recommendations.append(savings_obj.decode_softmax_to_label(result, savings_dict, 3))

    # print(len(savings_actual))
    # print(len(savings_top_recommendation))
    # print(len(savings_top3_recommendations))

    savings_model_accuracy = accuracy_score(savings_actual, savings_top_recommendation)
    savings_model_accuracy = round(savings_model_accuracy, 4) * 100
    print("savings  Model Accuracy - ", savings_model_accuracy)
    # print(savings_top_recommendation[0])
    print(savings_top3_recommendations[0])

    customer_df = customer_df[customer_df['customer_id'].isin(X_test_savings['customer_id'])]
    customer_df['savings_top3_reco'] = savings_top3_recommendations
    print(customer_df.head())
    customer_df.to_csv('./reco_results/savings_test_set_recommendations.csv', index=False)