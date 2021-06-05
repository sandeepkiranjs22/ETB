import pandas as pd
import numpy as np
from dask.distributed import Client

import warnings
warnings.filterwarnings('ignore')

import time
import sys
sys.path.append('/home/sandeep/Desktop/BankBuddy/Reco-usecases/new_reco/preprocessing_pipeline/')


data = pd.read_csv("./data/processed/credit_upsell_processed.csv")


def find_max_spent_category(data,cols):
    
    list_cat=[]
    
    for i in range(len(data)):
        max_val=0
        for j in cols:
            max_val=max(max_val,data[j][i])
            
        for j in cols:
            if max_val == data[j][i]:
                list_cat.append(j.split("_")[0])
                break
        
    data['max_spent_category']=list_cat
    
    return data


def find_max_trans_category(data,cols):
    
    list_cat=[]
    
    for i in range(len(data)):
        max_val=0
        for j in cols:
            max_val=max(max_val,data[j][i])
            
        for j in cols:
            if max_val == data[j][i]:
                list_cat.append(j.split("_")[0])
                break
        
    data['max_trans_category']=list_cat
    
    return data


def offer_upsell_reco(data,col1,col2):
    
    list_reco=[[] for i in range(len(data))]

    for i in range(len(data)):
        
        if data[col1][i]==data[col2][i]:
                
            if data['card_type'][i]=="silver":
                
                list_reco[i].append("gold "+str(data[col1][i]))
                list_reco[i].append("platinum "+str(data[col1][i]))
                
            
            if data['card_type'][i] == "gold":
                
                list_reco[i].append("platinum "+str(data[col1][i]))
                list_reco[i].append("signature "+str(data[col1][i]))
            
            
            if data['card_type'][i] == "platinum":
                
                list_reco[i].append("platinum "+str(data[col1][i]))
                list_reco[i].append("signature "+str(data[col1][i]))
                
            
            if data['card_type'][i] == "signature":

                list_reco[i].append("signature "+str(data[col1][i]))
            
        else:
            
                
            if data['card_type'][i]=="silver":
                
                list_reco[i].append("gold "+str(data[col1][i]))
                list_reco[i].append("gold "+str(data[col2][i]))
            
            if data['card_type'][i] == "gold":
                
                list_reco[i].append("platinum "+str(data[col1][i]))
                list_reco[i].append("platinum "+str(data[col2][i]))
            
            if data['card_type'][i] == "platinum":
                
                list_reco[i].append("signature "+str(data[col1][i]))
                list_reco[i].append("signature "+str(data[col2][i]))
                
            
            if data['card_type'][i] == "signature":

                list_reco[i].append("signature "+str(data[col1][i]))
                list_reco[i].append("signature "+str(data[col2][i]))
            
    return list_reco          


data = find_max_spent_category(data,['travel_amt','fuel_amt', 'entertainment_amt','food_amt', 'bill_amt'])

data = find_max_trans_category(data,['travel_trans','fuel_trans','entertainment_trans', 'food_trans','bill_trans'])

l=offer_upsell_reco(data,'max_spent_category','max_trans_category')

data['reco']=""

for i in range(len(data)):
    dict_reco={}
    val=0.75
    for j in range(len(l[i])):
        dict_reco[l[i][j]]=val
        val=1-val 
        
    data['reco'][i] = dict_reco


data2 = pd.read_csv('./data/unprocessed/consolidated_for_comparison.csv')
data2= data2[data2['customer_id'].isin(data['customer_id'])]
data2['top_reco'] = data['reco']
data2.to_csv('./reco_results/credit_upsell_test_set_recommendations.csv', index=False)
data2.to_csv('../model_outputs/reco_results/credit_upsell_test_set_recommendations.csv', index=False)
