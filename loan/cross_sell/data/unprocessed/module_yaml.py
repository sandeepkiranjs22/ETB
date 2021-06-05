import yaml
import pandas as pd
import json
import numpy as np

import sys
sys.path.append('/home/sandeep/Desktop/BankBuddy/Reco-usecases/yaml_files_validation')

from yaml_pipeline import *

obj = yaml_pipeline()

travel_yaml   = obj.read_yaml_filename("travel_ins.yml","travel_insurance")

motor_yaml    = obj.read_yaml_filename("motor_ins.yml","motor_insurance")

medical_yaml  = obj.read_yaml_filename("medical_ins.yml","medical_insurance")


data = pd.read_csv("test_set_recommendation_results.csv")

travel_list_final_reco = obj.yaml_parser(data,"travel_top3_reco",travel_yaml)
motor_list_final_reco = obj.yaml_parser(data,"motor_top3_reco",motor_yaml)
medical_list_final_reco = obj.yaml_parser(data,"med_top3_reco",medical_yaml)


data['medical_final_reco']=""
data['travel_final_reco']=""
data['motor_final_reco']=""


data['medical_final_reco']=medical_list_final_reco
data['travel_final_reco']=travel_list_final_reco
data['motor_final_reco']=motor_list_final_reco

data.to_csv("yaml_validation.csv",index=False)
