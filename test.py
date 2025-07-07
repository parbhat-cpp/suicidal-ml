import os
import pickle
import numpy as np
from src.utils.main_utils import preprocess_text,avg_word2vec

if __name__ == '__main__':
    artifacts_dir = [f"./artifacts/{dir}" for dir in os.listdir('./artifacts') if os.path.isdir(f"./artifacts/{dir}")]
    latest_subdir = max(artifacts_dir, key=os.path.getmtime)

    word2vec_model_path = f"{latest_subdir}/data_transformation/transformer/word2vec.pkl"
    classifier_model_path = f"{latest_subdir}/model_trainer/trained_model/model.pkl"
    
    input = str(input('Enter text: '))

    with open(word2vec_model_path, "rb") as file_obj:
        word2vec_model = pickle.load(file_obj)
    
    with open(classifier_model_path, "rb") as file_obj:
        model = pickle.load(file_obj)
    
    sentence = preprocess_text(input)
    sentence = avg_word2vec(word2vec_model, sentence)
    
    prediction = model.predict(np.array(sentence).reshape(1,-1))

    print(prediction)
