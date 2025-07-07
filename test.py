import pickle
from src.utils.main_utils import preprocess_text,avg_word2vec

if __name__ == '__main__':
    input = str(input('Enter text: '))
    with open('./artifacts/07_06_2025_19_35_10/model_trainer/trained_model/model.pkl', "rb") as file_obj:
        model = pickle.load(file_obj)
    
    sentence = preprocess_text(input)
    # sentence = avg_word2vec(sentence)
    
    model.predict()
    