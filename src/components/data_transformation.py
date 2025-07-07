import os
import sys
import pandas as pd
import numpy as np
import gensim
from joblib import Parallel, delayed

from src.entity.config_entity import DataTransformationConfig
from src.entity.artifact_entity import DataTransformationArtifact,DataIngestionArtifact
from src.exception import CustomException
from src.logger import logging
from src.utils.main_utils import save_numpy_arr,preprocess_text,avg_word2vec,save_object

class DataTransformation:
    def __init__(self, data_ingestion_artifact: DataIngestionArtifact, data_transformation_config: DataTransformationConfig) -> None:
        self.data_ingestion_artifact = data_ingestion_artifact
        self.data_transformation_config = data_transformation_config
    
    def initiate_data_transformation(self):
        try:
            logging.info('Data transformation started')
            # get file path
            trained_file_path = self.data_ingestion_artifact.trained_file_path
            test_file_path = self.data_ingestion_artifact.test_file_path
            
            logging.info('Loading train and test files')
            # load train and test files
            train_df = pd.read_csv(trained_file_path)
            test_df = pd.read_csv(test_file_path)
            
            logging.info('Preprocessing input and target features')
            # update target value with 1 and 0 for suicide and non-suicide respectively
            train_df['class'] = train_df['class'].apply(lambda x: 1 if x == 'suicide' else 0)
            test_df['class'] = test_df['class'].apply(lambda x: 1 if x == 'suicide' else 0)
            
            processed_train_text = train_df['text'].apply(preprocess_text)
            processed_test_text = test_df['text'].apply(preprocess_text)
            
            logging.info('Applying average word 2 vec')
            model = gensim.models.Word2Vec(processed_train_text, workers=4)
            
            save_object(
                self.data_transformation_config.word_transformer_path,
                model,
            )
            
            # Parallely average words to vector
            train_vectors = np.array(
                Parallel(n_jobs=-1)(delayed(avg_word2vec)(model, word) for word in processed_train_text)
            )
            test_vectors = np.array(
                Parallel(n_jobs=-1)(delayed(avg_word2vec)(model, word) for word in processed_test_text)
            )
            
            logging.info('Saving train and test numpy array')
            save_numpy_arr(self.data_transformation_config.transformed_train_file_path, np.c_[train_vectors, train_df['class']])
            save_numpy_arr(self.data_transformation_config.transformed_test_file_path, np.c_[test_vectors, test_df['class']])
            
            return DataTransformationArtifact(
                transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path=self.data_transformation_config.transformed_test_file_path,
                transformer_word2vec_file_path=self.data_transformation_config.word_transformer_path,
            )
        except Exception as e:
            raise CustomException(e, sys)
