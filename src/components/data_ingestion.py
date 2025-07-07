import os
import sys
import kaggle
import zipfile
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.entity.config_entity import DataIngestionConfig
from src.entity.artifact_entity import DataIngestionArtifact
from src.exception import CustomException
from src.logger import logging
import src.constants as training_constants

class DataIngestion:
    def __init__(self, data_ingestion_config: DataIngestionConfig) -> None:
        try:
            self.data_ingestion_config = data_ingestion_config
            self.kaggle_api = kaggle.api
        except Exception as e:
            raise CustomException(e, sys)
    
    def download_dataset_from_kaggle(self):
        try:
            """
            Function to download dataset from kaggle
            """
            download_dataset_path = os.path.join(
                self.data_ingestion_config.data_ingestion_dir,
                training_constants.DATA_INGESTION_FEATURE_STORE_DIR,
            )
            
            logging.info('Downloading dataset')
            self.kaggle_api.dataset_download_files(
                self.data_ingestion_config.dataset,
                path=download_dataset_path
            )
            logging.info('Dataset downloaded')
            with zipfile.ZipFile(self.data_ingestion_config.feature_zip_file_path, "r") as zip_file:
                zip_file.extractall(download_dataset_path)
            
            os.remove(self.data_ingestion_config.feature_zip_file_path)
            logging.info('Unziping and removal of datset.zip file complete')
            
            return self.data_ingestion_config.feature_store_file_path
        except Exception as e:
            raise CustomException(e, sys)
    
    def split_train_test_data(self, feature_dataset_path: str):
        try:
            """
            This function access feature dataset and create train.csv and test.csv file
            """
            df = pd.read_csv(feature_dataset_path)
            
            # remove Unnamed column from dataset
            df.drop('Unnamed: 0', inplace=True, axis=1)
            
            logging.info('Splitting dataset')
            train_dataset,test_dataset = train_test_split(
                df,test_size=self.data_ingestion_config.train_test_split_ratio
            )
            
            dir_path = os.path.dirname(self.data_ingestion_config.training_file_path)
            os.makedirs(dir_path, exist_ok=True)
            
            train_dataset.to_csv(
                self.data_ingestion_config.training_file_path,
                index=False,
            )
            test_dataset.to_csv(
                self.data_ingestion_config.testing_file_path,
                index=False,
            )
            logging.info('Dataset splitting done')
        except Exception as e:
            raise CustomException(e, sys)
    
    def initiate_data_ingestion(self) -> DataIngestionArtifact:
        try:
            logging.info('DataIngestion initiated')
            feature_dataset_path = self.download_dataset_from_kaggle()
            self.split_train_test_data(feature_dataset_path)
            
            return DataIngestionArtifact(
                trained_file_path=self.data_ingestion_config.training_file_path,
                test_file_path=self.data_ingestion_config.testing_file_path,
            )
        except Exception as e:
            raise CustomException(e, sys)
