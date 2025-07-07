import os
import sys

from src.exception import CustomException
from src.logger import logging
from src.entity.config_entity import TrainingPipelineConfig
from src.entity.config_entity import DataIngestionConfig
from src.entity.config_entity import DataTransformationConfig
from src.entity.config_entity import ModelTrainerConfig
from src.entity.artifact_entity import DataIngestionArtifact
from src.entity.artifact_entity import DataTransformationArtifact
from src.entity.artifact_entity import ModelTrainerArtifact
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

class TrainingPipeline:
    def __init__(self):
        self.training_pipeline_config = TrainingPipelineConfig()
    
    def start_data_ingestion(self):
        try:
            data_ingestion_config = DataIngestionConfig(self.training_pipeline_config)
            data_ingestion = DataIngestion(data_ingestion_config)
            data_ingestion_artifact = data_ingestion.initiate_data_ingestion()

            return data_ingestion_artifact
        except Exception as e:
            raise CustomException(e, sys)
    
    def start_data_transformation(self, data_ingestion_artifact: DataIngestionArtifact):
        try:
            data_transformation_config = DataTransformationConfig(self.training_pipeline_config)
            data_transformation = DataTransformation(data_ingestion_artifact,data_transformation_config)
            data_transformation_artifact = data_transformation.initiate_data_transformation()

            return data_transformation_artifact
        except Exception as e:
            raise CustomException(e, sys)

    def start_model_trainer(self, data_transformation_artifact: DataTransformationArtifact):
        try:
            model_trainer_config = ModelTrainerConfig(self.training_pipeline_config)
            model_trainer = ModelTrainer(model_trainer_config, data_transformation_artifact)
            model_trainer_artifact = model_trainer.initiate_model_trainer()

            return model_trainer_artifact
        except Exception as e:
            raise CustomException(e, sys)

    def run_pipeline(self):
        try:
            data_ingestion_artifact = self.start_data_ingestion()
            data_transformation_artifact = self.start_data_transformation(data_ingestion_artifact)
            model_trainer_artifact = self.start_model_trainer(data_transformation_artifact)

            return model_trainer_artifact
        except Exception as e:
            raise CustomException(e, sys)
