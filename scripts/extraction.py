import os
import pandas as pd
import dvc.api
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import CountVectorizer

def count_unique_words(text):
    words = text.split()
    return len(set(words))

def count_words(text):
    text = text.lower()
    words_to_count = ['good', 'well', 'great']
    count = sum(text.count(word) for word in words_to_count)
    return count

def handle_nan(df):
    df = df.drop('style', axis=1)
    df = df.drop('image', axis=1)
    df = df.drop('reviewerName', axis=1)
    df['reviewText'].fillna('', inplace=True)
    df['vote'].fillna(0, inplace=True)
    df['summary'].fillna('', inplace=True)
    return df
    
def add_features(df):
    df['reviewTextLength'] = df['reviewText'].astype(str).apply(len)
    df['numbers_amount'] = df['reviewText'].str.count(r'\d+')
    df["uniqWords"] = df["reviewText"].apply(count_unique_words)
    df["goodWords"] = df["reviewText"].apply(count_words)
    return df
    
def to_categorical(df):
    one_hot_data_origin = pd.get_dummies(df['data_origin'], prefix='origin_')
    df = pd.concat([df, one_hot_data_origin], axis=1)
    df = df.drop('data_origin', axis=1)
    
    one_hot_data_origin = pd.get_dummies(df['verified'], prefix='verified_')
    df = pd.concat([df, one_hot_data_origin], axis=1)
    df = df.drop('verified', axis=1)
    return df
    
def drop_useless(df):
    df = df.drop('asin', axis=1)
    df = df.drop('reviewTime', axis=1)
    df = df.drop('reviewerID', axis=1)
    return df
    
def standarize(df_train, df_test):
    columns_to_std = ['vote', 'numbers_amount', 'unixReviewTime', 'reviewTextLength', 'uniqWords', 'goodWords']
    
    X_train = df_train[columns_to_std].values
    X_test = df_test[columns_to_std].values
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    df_train[columns_to_std] = X_train
    df_test[columns_to_std] = X_test

    return df_train, df_test

def normalize_text(df, text_column_name):
    nltk.download('wordnet')
    nltk.download('stopwords')
    nltk.download('omw-1.4')
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    column_name = 'tmp'
    df[column_name] = df[text_column_name].apply(lambda x: re.sub(r'[^\w\s]', '', str(x)))
    df[column_name] = df[column_name].apply(lambda x: word_tokenize(x.lower()))
    df[column_name] = df[column_name].apply(lambda x: [word for word in x if word not in stop_words])
    lemmatizer = WordNetLemmatizer()
    df[column_name] = df[column_name].apply(lambda x: [lemmatizer.lemmatize(word) for word in x])
    df[column_name] = df[column_name].apply(lambda x: [word for word in x if word.isalnum()])
    df[column_name] = df[column_name].apply(lambda x: ' '.join(x))
    vectorizer = CountVectorizer(binary=True, max_df= 0.75, min_df = 2, max_features=1000)
    count_matrix = vectorizer.fit_transform(df[column_name])
    count_array = count_matrix.toarray()
    df_words = pd.DataFrame(data=count_array, columns = vectorizer.get_feature_names_out())
    df_words = df_words.add_prefix(text_column_name+'_')
    df = pd.concat([df, df_words], axis=1)
    df = df.drop(column_name, axis=1)
    return df

def clean_text(df):
    df = normalize_text(df, 'reviewText')
    df = normalize_text(df, 'summary')
    return df
    
def to_float(df):
    df['vote'] = df['vote'].apply(lambda x: x.replace(',', '.') if isinstance(x, str) else x)
    return df

def extract(df_train, df_test):
    df_train = to_float(df_train)
    df_test = to_float(df_test)
    
    df_train = drop_useless(df_train)
    df_test = drop_useless(df_test)
    
    df_train = handle_nan(df_train)
    df_test = handle_nan(df_test)
    
    df_train = add_features(df_train)
    df_test = add_features(df_test)
    
    df_train = to_categorical(df_train)
    df_test = to_categorical(df_test)
    
    df_train, df_test = standarize(df_train, df_test)
    
    df_train = clean_text(df_train)
    df_test = clean_text(df_test)
    
    common_columns = list(set(df_train.columns) & set(df_test.columns))
    df_train = df_train[common_columns]
    df_test = df_test[common_columns]
    return df_train, df_test

params = dvc.api.params_show()

start_file_train = params['split']['target_file_train']
start_file_test = params['split']['target_file_test']
target_file_train = params['extraction']['target_file_train']
target_file_test = params['extraction']['target_file_test']

df_train = pd.read_json(start_file_train, lines=True)
df_test = pd.read_json(start_file_test, lines=True)
df_train, df_test = extract(df_train, df_test)

df_train.to_json(target_file_train, orient='records', lines=True)
df_test.to_json(target_file_test, orient='records', lines=True)
