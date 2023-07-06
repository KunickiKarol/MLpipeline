import json
import os
import pandas as pd
from sklearn.model_selection import train_test_split
import dvc.api

params = dvc.api.params_show()

start_file = params['preprocessing']['target_file']
target_file_train = params['split']['target_file_train']
target_file_test = params['split']['target_file_test']
class_column_name = params['class_column_name']

df = pd.read_json(start_file, lines=True)

train, test = train_test_split(df, test_size=0.2, stratify=df[class_column_name])

train.to_json(target_file_train, orient='records', lines=True)
test.to_json(target_file_test, orient='records', lines=True)

