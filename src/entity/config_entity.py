import os
from datetime import datetime

import src.constants as constants

class TrainingPipelineConfig:
    def __init__(self, timestamp = datetime.now()) -> None:
        timestamp = timestamp.strftime('%m_%d_%Y_%H_%M_%S')
        self.artifact_name = constants.ARTIFACT_DIR
        self.artifact_dir = os.path.join(self.artifact_name, timestamp)
        self.timestamp = timestamp

class DataIngestionConfig:
    def __init__(self, training_pipeline_config: TrainingPipelineConfig) -> None:
        self.data_ingestion_dir = os.path.join(
            training_pipeline_config.artifact_dir,
            constants.DATA_INGESTION_DIR_NAME,
        )
        self.feature_store_file_path = os.path.join(
            self.data_ingestion_dir,
            constants.DATA_INGESTION_FEATURE_STORE_DIR,
            constants.FILE_NAME,
        )
        self.feature_zip_file_path = os.path.join(
            self.data_ingestion_dir,
            constants.DATA_INGESTION_FEATURE_STORE_DIR,
            constants.DATASET_NAME,
        )
        self.training_file_path = os.path.join(
            self.data_ingestion_dir,
            constants.DATA_INGESTION_INGESTED_DIR,
            constants.TRAIN_FILE_NAME,
        )
        self.testing_file_path = os.path.join(
            self.data_ingestion_dir,
            constants.DATA_INGESTION_INGESTED_DIR,
            constants.TEST_FILE_NAME,
        )
        self.train_test_split_ratio = constants.DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO
        self.dataset = constants.DATASET

class DataTransformationConfig:
    def __init__(self, training_pipeline_config: TrainingPipelineConfig) -> None:
        self.data_transformation_dir = os.path.join(
            training_pipeline_config.artifact_dir,
            constants.DATA_TRANSFORMATION_DIR_NAME,
        )
        self.transformed_train_file_path = os.path.join(
            self.data_transformation_dir,
            constants.DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR,
            constants.DATA_TRANSFORMATION_TRAIN_FILE_PATH,
        )
        self.transformed_test_file_path = os.path.join(
            self.data_transformation_dir,
            constants.DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR,
            constants.DATA_TRANSFORMATION_TEST_FILE_PATH,
        )
        self.word_transformer_path = os.path.join(
            self.data_transformation_dir,
            constants.DATA_TRANSFORMATION_WORD_2_VEC_MODEL_DIR,
            constants.DATA_TRANSFORMATION_WORD_2_VEC_PATH
        )

class ModelTrainerConfig:
    def __init__(self, training_pipeline_config: TrainingPipelineConfig) -> None:
        self.model_trainer_dir = os.path.join(
            training_pipeline_config.artifact_dir, constants.MODEL_TRAINER_DIR_NAME
        )
        self.trained_model_file_path = os.path.join(
            self.model_trainer_dir, constants.MODEL_TRAINER_TRAINED_MODEL_DIR, constants.MODEL_TRAINER_TRAINED_MODEL_NAME
        )
        self.expected_accuracy = constants.MODEL_TRAINER_EXPECTED_SCORE
        self.overfitting_underfitting_threshold = constants.MODEL_TRAINER_OVER_FITTING_UNDER_FITTING_THRESHOLD
