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

    travel_file = './data/processed/travel_processed.csv'
    motor_file = './data/processed/motor_processed.csv'
    med_file = './data/processed/med_processed.csv'
    home_file = './data/processed/home_processed.csv'
    prop_file = './data/processed/prop_processed.csv'
    
    customer_file = './data/unprocessed/consolidated_for_comparison.csv'

    customer_obj = ModelTraining(customer_file)
    customer_df = customer_obj.get_df()

    travel_obj = ModelTraining(travel_file)
    travel_df = travel_obj.get_df()
    travel_ddf = travel_obj.convert_df_to_ddf(travel_df)
    print(len(travel_df))

    motor_obj = ModelTraining(motor_file)
    motor_df = motor_obj.get_df()
    motor_ddf = motor_obj.convert_df_to_ddf(motor_df)
    print(len(motor_df))

    med_obj = ModelTraining(med_file)
    med_df = med_obj.get_df()
    med_ddf = med_obj.convert_df_to_ddf(med_df)
    print(len(med_df))


    home_obj = ModelTraining(home_file)
    home_df = home_obj.get_df()
    home_ddf = home_obj.convert_df_to_ddf(home_df)
    print(len(home_df))

    prop_obj = ModelTraining(prop_file)
    prop_df = prop_obj.get_df()
    prop_ddf = prop_obj.convert_df_to_ddf(prop_df)
    print(len(prop_df))


    y_travel = travel_obj.label_encoding(travel_df['travel_addon'])
    # y_travel = pd.DataFrame(y_travel)
    y_motor = motor_obj.label_encoding(motor_df['motor_addon'])
    y_med = med_obj.label_encoding(med_df['medical_addon'])


    y_home = home_obj.label_encoding(home_df['home_addon'])
    # y_home = pd.DataFrame(y_home)
    y_prop = prop_obj.label_encoding(prop_df['prop_addon'])


    # y_travel_dask = travel_obj.convert_df_to_ddf(y_travel)
    y_travel_dask = travel_obj.convert_target_col_to_dask_df(y_travel)
    y_motor_dask = motor_obj.convert_target_col_to_dask_df(y_motor)
    y_med_dask = med_obj.convert_target_col_to_dask_df(y_med)

    y_home_dask = home_obj.convert_target_col_to_dask_df(y_home)
    y_prop_dask = prop_obj.convert_target_col_to_dask_df(y_prop)

    """Saving the encoded column"""
    travel_dict = {}
    motor_dict = {}
    med_dict = {}
    home_dict = {}
    prop_dict = {}


    for i in range(len(y_travel)):
        travel_dict[y_travel[i]] = travel_df['travel_addon'][i]

    for i in range(len(y_motor)):
        motor_dict[y_motor[i]] = motor_df['motor_addon'][i]

    for i in range(len(y_med)):
        med_dict[y_med[i]] = med_df['medical_addon'][i]

    for i in range(len(y_home)):
        home_dict[y_home[i]] = home_df['home_addon'][i]

    for i in range(len(y_prop)):
        prop_dict[y_prop[i]] = prop_df['prop_addon'][i]
    


    print(travel_dict)
    print(motor_dict)
    print(med_dict)
    print(home_dict)
    print(prop_dict)
    
    
    print(travel_dict.values())
    print(motor_dict.values())
    print(med_dict.values())
    print(home_dict.values())
    print(prop_dict.values())
    


    travel_ddf = travel_ddf.drop('travel_addon', axis=1)
    motor_ddf = motor_ddf.drop('motor_addon', axis=1)
    med_ddf = med_ddf.drop('medical_addon', axis=1)
    home_ddf = home_ddf.drop('home_addon', axis=1)
    prop_ddf = prop_ddf.drop('prop_addon', axis=1)


    X_train_travel, X_test_travel, y_train_travel, y_test_travel = travel_obj.split_dataset(travel_ddf, y_travel_dask)
    X_train_motor, X_test_motor, y_train_motor, y_test_motor = motor_obj.split_dataset(motor_ddf, y_motor_dask)
    X_train_med, X_test_med, y_train_med, y_test_med = med_obj.split_dataset(med_ddf, y_med_dask)
    X_train_home, X_test_home, y_train_home, y_test_home = home_obj.split_dataset(home_ddf, y_home_dask)    
    X_train_prop, X_test_prop, y_train_prop, y_test_prop = prop_obj.split_dataset(prop_ddf, y_prop_dask)


    """MODEL TRAINING"""

    """travel recommendation training"""
    travel_cols = travel_obj.get_nonobj_col_list(travel_df)
    params = {'objective': 'multi:softprob', 'nround': 1000, 'num_class': len(travel_dict), 'max_depth': 6}
    bst_travel = travel_obj.dxgb_train(client, params, X_train_travel[travel_cols], y_train_travel)
    print("travel model trained")
    print()

    """motor recommendation training"""
    motor_cols = motor_obj.get_nonobj_col_list(motor_df)
    params = {'objective': 'multi:softprob', 'nround': 1000, 'num_class': len(motor_dict), 'max_depth': 6}
    bst_motor = motor_obj.dxgb_train(client, params, X_train_motor[motor_cols], y_train_motor)
    print("motor model trained")
    # bst_motor.save_model('./models/motor_model.json')

    """medical recommendation training"""
    med_cols = med_obj.get_nonobj_col_list(med_df)
    params = {'objective': 'multi:softprob', 'nround': 1000, 'num_class': len(med_dict), 'max_depth': 6}
    bst_med = med_obj.dxgb_train(client, params, X_train_med[med_cols], y_train_med)
    print("med model trained")
    # bst_med.save_model('./models/med_model.json')


    """home recommendation training"""
    home_cols = home_obj.get_nonobj_col_list(home_df)
    params = {'objective': 'multi:softprob', 'nround': 1000, 'num_class': len(home_dict), 'max_depth': 6}
    bst_home = home_obj.dxgb_train(client, params, X_train_home[home_cols], y_train_home)
    print("home model trained")
    print()

    """prop recommendation training"""
    prop_cols = prop_obj.get_nonobj_col_list(prop_df)
    params = {'objective': 'multi:softprob', 'nround': 1000, 'num_class': len(prop_dict), 'max_depth': 6}
    bst_prop = prop_obj.dxgb_train(client, params, X_train_prop[prop_cols], y_train_prop)
    print("prop model trained")

    """Predictions"""
    # print(len(travel_cols))
    # print(travel_cols)
    # travel_pred = xgb.DMatrix(X_test_travel[travel_cols].head(len(X_test_travel)))
    travel_pred = xgb.DMatrix(X_test_travel[travel_cols])
    # print("X_test_travel", len(X_test_travel))
    # print(travel_pred.num_row())
    # print(travel_pred.num_col())
    travel_pred.feature_names = travel_cols
    # print(travel_pred.feature_names)
    travel_results = (bst_travel.predict(travel_pred))
    # print(len(travel_results))

    motor_pred = xgb.DMatrix(X_test_motor[motor_cols].head(len(X_test_motor)))
    motor_results = (bst_motor.predict(motor_pred))

    med_pred = xgb.DMatrix(X_test_med[med_cols].head(len(X_test_med)))
    med_results = (bst_med.predict(med_pred))

    home_pred = xgb.DMatrix(X_test_home[home_cols])

    home_pred.feature_names = home_cols

    home_results = (bst_home.predict(home_pred))

    prop_pred = xgb.DMatrix(X_test_prop[prop_cols].head(len(X_test_prop)))
    prop_results = (bst_prop.predict(prop_pred))


    """Get back recommendations"""
    travel_actual = list(y_test_travel[0].compute())
    travel_top_recommendation = []
    travel_top3_recommendations = []
    for result in travel_results:
        travel_top_recommendation.append(travel_obj.decode_softmax_to_label(result, travel_dict, 1))
        travel_top3_recommendations.append(travel_obj.decode_softmax_to_label(result, travel_dict, 3))

    # print(len(travel_actual))
    # print(len(travel_top_recommendation))
    # print(len(travel_top3_recommendations))
    travel_model_accuracy = accuracy_score(travel_actual, travel_top_recommendation)
    travel_model_accuracy = round(travel_model_accuracy, 4) * 100
    print("Travel Model Accuracy - ", travel_model_accuracy)
    print(travel_top_recommendation[0])
    print(travel_top3_recommendations[0])

    motor_actual = list(y_test_motor[0].compute())
    motor_top_recommendation = []
    motor_top3_recommendations = []
    for result in motor_results:
        motor_top_recommendation.append(motor_obj.decode_softmax_to_label(result, motor_dict, 1))
        motor_top3_recommendations.append(motor_obj.decode_softmax_to_label(result, motor_dict, 3))

    motor_model_accuracy = accuracy_score(motor_actual, motor_top_recommendation)
    motor_model_accuracy = round(motor_model_accuracy, 4) * 100
    print("Motor Model Accuracy - ", motor_model_accuracy)
    print(motor_top_recommendation[0])
    print(motor_top3_recommendations[0])

    med_actual = list(y_test_med[0].compute())
    med_top_recommendation = []
    med_top3_recommendations = []
    for result in med_results:
        med_top_recommendation.append(med_obj.decode_softmax_to_label(result, med_dict, 1))
        med_top3_recommendations.append(med_obj.decode_softmax_to_label(result, med_dict, 3))

    med_model_accuracy = accuracy_score(med_actual, med_top_recommendation)
    med_model_accuracy = round(med_model_accuracy, 4) * 100
    print("Medical Model Accuracy - ", med_model_accuracy)
    print(med_top_recommendation[0])
    print(med_top3_recommendations[0])


    home_actual = list(y_test_home[0].compute())
    home_top_recommendation = []
    home_top3_recommendations = []
    for result in home_results:
        home_top_recommendation.append(home_obj.decode_softmax_to_label(result, home_dict, 1))
        home_top3_recommendations.append(home_obj.decode_softmax_to_label(result, home_dict, 3))


    home_model_accuracy = accuracy_score(home_actual, home_top_recommendation)
    home_model_accuracy = round(home_model_accuracy, 4) * 100
    print("home Model Accuracy - ", home_model_accuracy)
    print(home_top_recommendation[0])
    print(home_top3_recommendations[0])
    
    

    prop_actual = list(y_test_prop[0].compute())
    prop_top_recommendation = []
    prop_top3_recommendations = []
    for result in prop_results:
        prop_top_recommendation.append(prop_obj.decode_softmax_to_label(result, prop_dict, 1))
        prop_top3_recommendations.append(prop_obj.decode_softmax_to_label(result, prop_dict, 3))

    prop_model_accuracy = accuracy_score(prop_actual, prop_top_recommendation)
    prop_model_accuracy = round(prop_model_accuracy, 4) * 100
    print("prop Model Accuracy - ", prop_model_accuracy)
    print(prop_top_recommendation[0])
    print(prop_top3_recommendations[0])


    """Storing results in csv for now"""
    travel_recommendations_df = pd.DataFrame()
    motor_recommendations_df = pd.DataFrame()
    med_recommendations_df = pd.DataFrame()
    home_recommendations_df = pd.DataFrame()
    prop_recommendations_df = pd.DataFrame()


    travel_recommendations_df['customer_id'] = X_test_travel['customer_id']
    travel_recommendations_df['travel_top3_reco'] = travel_top3_recommendations
    print(travel_recommendations_df.head())

    motor_recommendations_df['customer_id'] = X_test_motor['customer_id']
    motor_recommendations_df['motor_top3_reco'] = motor_top3_recommendations
    print(motor_recommendations_df.head())

    med_recommendations_df['customer_id'] = X_test_med['customer_id']
    med_recommendations_df['med_top3_reco'] = med_top3_recommendations
    print(med_recommendations_df.head())


    home_recommendations_df['customer_id'] = X_test_home['customer_id']
    home_recommendations_df['home_top3_reco'] = home_top3_recommendations
    print(home_recommendations_df.head())

    prop_recommendations_df['customer_id'] = X_test_prop['customer_id']
    prop_recommendations_df['prop_top3_reco'] = prop_top3_recommendations
    print(prop_recommendations_df.head())


    unique_test_ids_list = list(travel_recommendations_df.customer_id.unique()) + list(motor_recommendations_df.customer_id.unique()) + list(med_recommendations_df.customer_id.unique())+list(home_recommendations_df.customer_id.unique()) + list(prop_recommendations_df.customer_id.unique())
    unique_test_ids_list = list(set(unique_test_ids_list))
    print("total test ids - ", len(unique_test_ids_list))

    master_reco_df = pd.merge(travel_recommendations_df, motor_recommendations_df, on='customer_id', how='outer')
    master_reco_df = pd.merge(master_reco_df, med_recommendations_df, on='customer_id', how='outer')
    master_reco_df = pd.merge(master_reco_df, home_recommendations_df, on='customer_id', how='outer')
    master_reco_df = pd.merge(master_reco_df, prop_recommendations_df, on='customer_id', how='outer')

    print(customer_df.columns)
    master_reco_df = pd.merge(master_reco_df, customer_df, on='customer_id', how='outer')
    master_reco_df = master_reco_df[master_reco_df['customer_id'].isin(unique_test_ids_list)]

    print("Total Columns in TestDf - ", len(master_reco_df.columns))
    print("Total Rows in TestDf - ", len(master_reco_df))
    master_reco_df.to_csv('./reco_results/insurance_upsell_test_set_recommendation_results.csv', index=False)
    master_reco_df.to_csv('../model_outputs/reco_results/insurance_upsell_test_set_recommendations.csv', index=False)
