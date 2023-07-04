import json
import pandas as pd
import dvc.api

params = dvc.api.params_show()

start_file = params['processing']['start_file']
target_file = params['processing']['target_file']
file_names = params['processing']['file_names']

merged_data = {}
df = pd.DataFrame()

for file_name in file_names:
    data = pd.read_json(start_file + '/' +file_name, lines=True)
    data['data_origin'] = file_name
    df = pd.concat([df, data], ignore_index=True)
    
df = df.drop_duplicates()
df = df.reset_index(drop=True)
df.to_json(target_file, orient='records', lines=True)
