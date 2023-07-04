import os
import pandas as pd
import numpy as np
import json
import io
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import cross_validate
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_selection import SelectKBest, f_classif, VarianceThreshold
import matplotlib.pyplot as plt
import seaborn as sns
import dvc.api
import datetime
import mlflow

def get_text_only(df):
    df_text_only = df.filter(regex='^reviewText_|^overall|^summary_')
    df_text_only.name = 'text_only'
    return df_text_only

def get_not_text(df):
    df_not_text = df.filter(regex='^origin_|^overall|^vote|^numbers_amount|^verified|^reviewTextLength|^unixReviewTime')
    df_not_text.name = 'not_text'
    return df_not_text

def get_model(model_name):
    if model_name == 'rf':
        return RandomForestClassifier(random_state=1, min_samples_leaf= 5)
    elif model_name == 'svm':
        return LinearSVC(random_state=1, max_iter=10000, tol=0.001)
    elif model_name == 'dummy':
        return DummyClassifier(random_state=1)
    else:
        raise ValueError("Nie podano modelu")

def drop_inappropriate(df):
    return df.drop(['reviewText', 'summary'], axis=1)

def get_x_y(df, class_column_name):
    return df.drop(class_column_name, axis=1), df[class_column_name] 

def selection(x_train, x_test, y_train, selection_method, reduction_method):
    print("start selection")
    if selection_method == "VarianceThreshold":
        selector = VarianceThreshold(threshold=0.01)
        x_train_2 = selector.fit_transform(x_train)
        print(x_train.columns[selector.get_support()])
        x_train = x_train_2
        x_test = selector.transform(x_test)
    elif selection_method == 'SelectKBest':
        selector = SelectKBest(score_func=f_classif, k=1000)
        x_train = selector.fit_transform(x_train, y_train)
        x_test = selector.transform(x_test)
    print("start reduction")
    if reduction_method == "PCA":
        reductor = PCA(n_components=50)
        x_train = reductor.fit_transform(x_train)
        x_test = reductor.transform(x_test)
    elif reduction_method == 'TruncatedSVD':
        reductor = TruncatedSVD(n_components=100)
        x_train = reductor.fit_transform(x_train)
        x_test = reductor.transform(x_test)
    return x_train, x_test
    
params = dvc.api.params_show()
os.environ['MLFLOW_TRACKING_URI'] = 'http://0.0.0.0:5000'

start_file_train = params['extraction']['target_file_train']
start_file_test = params['extraction']['target_file_test']
class_column_name = params['class_column_name']
target_file = params['learn_and_evaluation']['target_file']
method = params['learn_and_evaluation']['method']
model_name = params['learn_and_evaluation']['model']
selection_method = params['learn_and_evaluation']['selection']
reduction_method = params['learn_and_evaluation']['reduction']

if method == 'cross':
    
    df_train = pd.read_json(start_file_train, lines=True)
    df_text_only = get_text_only(df_train)
    df_not_text = get_not_text(df_train)
    df_train = df_train.drop(['reviewText', 'summary'], axis=1)
    dummy = DummyClassifier(random_state=1)
    RF_model = RandomForestClassifier(random_state=1, min_samples_leaf= 5)
    SVM_model = LinearSVC(random_state=1, max_iter=10000, tol=0.001)
    results = []
    df_names = ['all_data', 'text_only', 'not_text']
    for model_idx, model in enumerate([dummy, RF_model, SVM_model]):
        print('#'*50)
        mlflow.start_run()
        mlflow.set_tag('model', model)
        mlflow.log_param("Method", method)
        mlflow.log_param("Model", model)
        for df_name, df_cross in zip(df_names, [df_train, df_text_only, df_not_text]):
            result = cross_validate(model, df_cross.drop(class_column_name, axis=1), df_cross[class_column_name], cv=5, 
                             scoring=['f1_micro'])   

            mlflow.log_metric(df_name + '_test_f1_micro', np.mean(result['test_f1_micro']))
            results.append({type(model).__name__ + "_" + df_name + '_test_f1_micro': np.mean(result['test_f1_micro'])})
            print(np.mean(result['test_f1_micro']))
        mlflow.end_run()
    with open(target_file, 'w') as f:
        json.dump(results, f)
        
elif method == 'fit':
    df_train = pd.read_json(start_file_train, lines=True)
    df_train = drop_inappropriate(df_train)
    x_train, y_train = get_x_y(df_train, class_column_name)
    
    df_test = pd.read_json(start_file_test, lines=True)
    df_test = drop_inappropriate(df_test)
    x_test, y_test = get_x_y(df_test, class_column_name)
    

    x_train, x_test = selection(x_train, x_test, y_train, selection_method, reduction_method)
    
    model = get_model(model_name)
    print("start fit")
    model.fit(x_train, y_train)
    print("start predict")
    
    y_pred = model.predict(x_test)
    result = f1_score(y_test, y_pred, average='micro')
    
    now = datetime.datetime.now()
    timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")
    mlflow_name = type(model).__name__ + ' '+ selection_method + ' '+ reduction_method + ' '+ timestamp
    mlflow.start_run(run_name= mlflow_name)
    mlflow.set_tag('model', model)
    mlflow.log_param("Method", method)
    mlflow.log_param("Model", model)
    mlflow.log_param("Selection", selection_method)
    mlflow.log_param("Reduction", reduction_method)
    mlflow.log_metric('f1_micro', result)
    mlflow.end_run()
    print(result)
    with open(target_file, 'w') as f:
        json.dump(result, f)
    

