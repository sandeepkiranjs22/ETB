import pandas as pd
from joblib import Parallel, parallel_backend
import dask.dataframe as dd
from dask.distributed import Client
from dask_ml.preprocessing import DummyEncoder
import dask_ml.cluster
import matplotlib.pyplot as plt
import dask_xgboost as dxgb
from dask_ml.preprocessing import LabelEncoder
import dask.dataframe as dd
from dask_ml.model_selection import train_test_split
from dask_ml.model_selection import GridSearchCV
from dask_ml.xgboost import XGBClassifier
import xgboost as xgb

pd.set_option('display.max_rows', 1000)
pd.set_option('display.max_columns', 1000)

import warnings
warnings.filterwarnings('ignore')


class ModelTraining:

    def __init__(self, data_file):
        df = pd.read_csv(data_file)
        self.df = df

    def get_df(self):
        return self.df

    def convert_df_to_ddf(self, df):
        x = dd.from_pandas(df, chunksize=50000)
        return x
	

    def dict_mapper(self,listc,name,threshold):
              
        list1=[]
        
        for i in range(len(listc)):
            
            d=dict()
            
            val = listc[i]
        
            if val>threshold:
                d[name]=(listc[i])
                
            list1.append(d)
        
        return list1


    def target_col(self,df,y):
        """ target column"""
        for i in range(len(df)):
            if df[y][i]=="yes":
                df[y][i]=1
            else:
                df[y][i]=0
            
        return df	

	
    
    
    def label_encoding(self, y):
        """df should not contain the target column"""
        le = LabelEncoder()
        y = le.fit_transform(y)
        return y

    def convert_target_col_to_dask_df(self, y):
        darr = dd.from_array(y)
        y = darr.to_frame()
        return y

    def dxgb_train(self, client, params, x, y):
        bst = dxgb.train(client, params, x, y)
        return bst

    def get_nonobj_col_list(self, df):
        nonobj_cols = []
        for col in df.columns:
            if df[col].dtypes != "object":
                nonobj_cols.append(col)
        return nonobj_cols

    def addoffer_churn_customers(self,df,cats,col_name1,col_name2,threshold):
        
        list_reco=[]
        for i in range(len(df)):
            if df[col_name1][i] > threshold:   
                for j in cats:
                    if df[j][i]==1:
                        val=j.split("_")
                        list_reco.append(str(val[3]) +":" + str(df[col_name1][i])) 
                        
            else:
                list_reco.append("")

        return list_reco


    def split_dataset(self, x, y, test_size=0.20):
        X_train, X_test, y_train, y_test = train_test_split(x, y, random_state=123, test_size=test_size)
        return X_train, X_test, y_train, y_test

    def decode_softmax_to_label(self, prediction_array, reco_mapper, num):
        indexes = sorted(range(len(prediction_array)), key=lambda i: prediction_array[i], reverse=True)[:num]
        if num == 1:
            return indexes[0]

        reco_dict = {}
        for i in indexes:
            reco_dict[reco_mapper[i]] = prediction_array[i]
        return reco_dict
