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

    currentaccount_file = './data/processed/currentaccount_processed.csv'
    currentaccount_obj = ModelTraining(currentaccount_file)
    currentaccount_df = currentaccount_obj.get_df()


    currentaccount_df = currentaccount_obj.target_col(currentaccount_df,"current_acc_avl")

    currentaccount_ddf = currentaccount_obj.convert_df_to_ddf(currentaccount_df)


    customer_file = './data/unprocessed/consolidated_for_comparison.csv'
    customer_obj = ModelTraining(customer_file)
    customer_df = customer_obj.get_df()




    """target column"""
    y_currentaccount_dask = currentaccount_ddf['current_acc_avl']

    #print(currentaccount_df.columns)

    #currentaccount_df.drop('current_acc_avl',axis=1,inplace=True)

    currentaccount_ddf = currentaccount_ddf.drop('current_acc_avl', axis=1)

    """Train Test Split"""
    X_train_currentaccount, X_test_currentaccount, y_train_currentaccount, y_test_currentaccount = currentaccount_obj.split_dataset(currentaccount_ddf, y_currentaccount_dask)

    """MODEL TRAINING"""

    """currentaccount recommendation training"""
    currentaccount_cols = currentaccount_obj.get_nonobj_col_list(currentaccount_ddf.compute())
    params = {'nround': 1000, 'max_depth': 6 ,'objective': 'binary:logistic',
          'eta': 0.01, 'subsample': 0.5,
          'min_child_weight': 0.5}

    bst_currentaccount = currentaccount_obj.dxgb_train(client, params, X_train_currentaccount[currentaccount_cols], y_train_currentaccount)
    print("currentaccount model trained")
    print()

    """Predictions"""
    currentaccount_pred = xgb.DMatrix(X_test_currentaccount[currentaccount_cols])
    currentaccount_pred.feature_names = currentaccount_cols
    currentaccount_results = (bst_currentaccount.predict(currentaccount_pred))

    """Get back recommendations"""
    currentaccount_actual = list(y_test_currentaccount.compute())

    train_currentaccount = pd.DataFrame()
    train_currentaccount = X_test_currentaccount.compute()
    train_currentaccount['currentaccount_score']= currentaccount_results
    train_currentaccount = train_currentaccount.reset_index()
    train_currentaccount.drop('index',axis=1,inplace=True)
    
    customer_df = customer_df[customer_df['customer_id'].isin(train_currentaccount['customer_id'])]
    customer_df['currentaccount_reco'] = currentaccount_results
    
    customer_df['currentaccount_reco'] = currentaccount_results
    
    res = customer_obj.dict_mapper(currentaccount_results,"current_account",0.40)
  
    customer_df['currentaccount_reco'] = res
    
    customer_df.to_csv('./reco_results/currentaccount_test_set_recommendations.csv', index=False)
