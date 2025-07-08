import os
from dotenv import load_dotenv

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.entity.config_entity import TrainingPipelineConfig
from src.entity.config_entity import DataIngestionConfig
from src.entity.config_entity import DataTransformationConfig
from src.entity.config_entity import ModelTrainerConfig

load_dotenv()

os.environ['KAGGLE_USERNAME'] = str(os.getenv('KAGGLE_USERNAME'))
os.environ['KAGGLE_KEY'] = str(os.getenv('KAGGLE_KEY'))

if __name__ == '__main__':
    training_pipeline_config = TrainingPipelineConfig()
    
    data_ingestion_config = DataIngestionConfig(training_pipeline_config)
    data_ingestion = DataIngestion(data_ingestion_config)
    data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
    print(data_ingestion_artifact)
    
    data_transformation_config = DataTransformationConfig(training_pipeline_config)
    data_transformation = DataTransformation(data_ingestion_artifact,data_transformation_config)
    data_transformation_artifact = data_transformation.initiate_data_transformation()
    print(data_transformation_artifact)
    
    model_trainer_config = ModelTrainerConfig(training_pipeline_config)
    model_trainer = ModelTrainer(model_trainer_config, data_transformation_artifact)
    model_trainer_artifact = model_trainer.initiate_model_trainer()
    print(model_trainer_artifact)
