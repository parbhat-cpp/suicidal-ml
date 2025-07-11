import os
import sys
import numpy as np
import pickle
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score
from gensim.utils import simple_preprocess
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from dotenv import load_dotenv

from src.exception import CustomException

load_dotenv()

def preprocess_text(text):
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    
    tokens = simple_preprocess(text, deacc=True, min_len=3)
    clean_tokens = [
        lemmatizer.lemmatize(word)
        for word in tokens
        if word not in stop_words
    ]
    return clean_tokens

def avg_word2vec(model, doc):
    vecs = [model.wv[word] for word in doc if word in model.wv.index_to_key]
    if len(vecs) == 0:
        return np.zeros(model.vector_size)
    return np.mean(vecs, axis=0)

def save_numpy_arr(file_path: str, arr: np.array):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        
        with open(file_path, "wb") as file_obj:
            np.save(file_obj, arr)
    except Exception as e:
        raise CustomException(e, sys)

def load_numpy_array_data(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return np.load(file_obj)
    except Exception as e:
        raise CustomException(e,sys)

def evaluate_model(x_train, y_train, x_test, y_test, models: dict, params: dict):
    try:
        report = {}
        
        for model_name, model in models.items():
            param = params[model_name]
            
            grid_search = GridSearchCV(model, param, cv=3, n_jobs=-1)
            grid_search.fit(x_train, y_train)
            
            model.set_params(**grid_search.best_params_)
            model.fit(x_train, y_train)
            
            y_test_pred = model.predict(x_test)
            
            test_model_score = r2_score(y_test, y_test_pred)
            
            report[model_name] = test_model_score
        
        return report
    except Exception as e:
        raise CustomException(e, sys)

def save_object(file_path, obj):
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
    except Exception as e:
        raise CustomException(e, sys)

def load_object(file_path: str):
    try:
        if not os.path.exists(file_path):
            raise Exception(f"The file: {file_path} does not exists")
        
        with open(file_path, "rb") as file_obj:
            obj = pickle.load(file_obj)
        
        return obj
    except Exception as e:
        raise CustomException(e, sys)
