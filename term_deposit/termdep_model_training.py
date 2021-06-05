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

    termdep_file = './data/processed/termdep_processed.csv'
    termdep_obj = ModelTraining(termdep_file)
    termdep_df = termdep_obj.get_df()


    termdep_df = termdep_obj.target_col(termdep_df,"term_dep_avl")

    termdep_ddf = termdep_obj.convert_df_to_ddf(termdep_df)


    customer_file = './data/unprocessed/consolidated_for_comparison.csv'
    customer_obj = ModelTraining(customer_file)
    customer_df = customer_obj.get_df()




    """target column"""
    y_termdep_dask = termdep_ddf['term_dep_avl']

    #print(termdep_df.columns)

    #termdep_df.drop('term_dep_avl',axis=1,inplace=True)

    termdep_ddf = termdep_ddf.drop('term_dep_avl', axis=1)

    """Train Test Split"""
    X_train_termdep, X_test_termdep, y_train_termdep, y_test_termdep = termdep_obj.split_dataset(termdep_ddf, y_termdep_dask)

    """MODEL TRAINING"""

    """termdep recommendation training"""
    termdep_cols = termdep_obj.get_nonobj_col_list(termdep_ddf.compute())
    params = {'nround': 1000, 'max_depth': 6 ,'objective': 'binary:logistic',
          'eta': 0.01, 'subsample': 0.5,
          'min_child_weight': 0.5}

    bst_termdep = termdep_obj.dxgb_train(client, params, X_train_termdep[termdep_cols], y_train_termdep)
    print("termdep model trained")
    print()

    """Predictions"""
    termdep_pred = xgb.DMatrix(X_test_termdep[termdep_cols])
    termdep_pred.feature_names = termdep_cols
    termdep_results = (bst_termdep.predict(termdep_pred))

    """Get back recommendations"""
    termdep_actual = list(y_test_termdep.compute())


    #print(termdep_actual)
    #print(termdep_results)

    train_termdep = pd.DataFrame()
    train_termdep = X_test_termdep.compute()
    train_termdep['termdep_score']= termdep_results
    train_termdep = train_termdep.reset_index()
    train_termdep.drop('index',axis=1,inplace=True)

  
    customer_df = customer_df[customer_df['customer_id'].isin(train_termdep['customer_id'])]
    
    customer_df['termdep_reco'] = termdep_results
    
    res = customer_obj.dict_mapper(termdep_results,"term_deposit",0.40)
  
    customer_df['termdep_reco'] = res
    
    customer_df.to_csv('./reco_results/termdep_test_set_recommendations.csv', index=False)