import os
import sys

import mlflow.sklearn

from src.entity.config_entity import ModelTrainerConfig
from src.entity.artifact_entity import ModelTrainerArtifact,DataTransformationArtifact,ClassificationMetricArtifact
from src.exception import CustomException
from src.logger import logging
from src.utils.main_utils import load_numpy_array_data,evaluate_model,save_object
from src.utils.ml_utils.classification_metrics import get_classfication_score

import mlflow
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB

class ModelTrainer:
    def __init__(self, model_trainer_config: ModelTrainerConfig, data_transformation_artifact: DataTransformationArtifact) -> None:
        self.model_trainer_config = model_trainer_config
        self.data_transformation_artifact = data_transformation_artifact
    
    def track_mlflow(self, best_model, classification_metric: ClassificationMetricArtifact):
        with mlflow.start_run():
            f1_score = classification_metric.f1_score
            precision_score = classification_metric.precision_score
            recall_score = classification_metric.recall_score
            
            mlflow.log_metric("f1_score", f1_score)
            mlflow.log_metric("precision_score", precision_score)
            mlflow.log_metric("recall_score", recall_score)
            mlflow.sklearn.log_model(best_model, 'best_model')
    
    def model_train(self, x_train, y_train, x_test, y_test):
        try:
            models = {
                "RandomForestClassifier": RandomForestClassifier(verbose=1),
                "LogisticRegression": LogisticRegression(verbose=1),
                "GaussianNB": GaussianNB(),
            }
            
            params = {
                "RandomForestClassifier": {
                    "max_features": ["sqrt", "log2"],
                    "criterion": ['gini', 'entropy', 'log_loss'],
                },
                "LogisticRegression": {
                    "penalty": ['elasticnet', 'l1', 'l2'],
                    "solver": ['lbfgs', 'liblinear', 'newton-cg', 'newton-cholesky', 'sag', 'saga']
                },
                "GaussianNB": {},
            }
            
            logging.info('Evaluating classification models')
            model_report: dict = evaluate_model(x_train, y_train, x_test, y_test, models, params)
            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]
            
            logging.info('Model evaluation done')
            
            y_train_pred = best_model.predict(x_train)
            train_classification_metric = get_classfication_score(y_train, y_train_pred)
            
            self.track_mlflow(best_model, train_classification_metric)
            
            y_test_pred = best_model.predict(x_test)
            test_classification_metric = get_classfication_score(y_test, y_test_pred)
            
            self.track_mlflow(best_model, test_classification_metric)
            
            model_dir_path = os.path.dirname(self.model_trainer_config.trained_model_file_path)
            os.makedirs(model_dir_path)
            
            save_object(self.model_trainer_config.trained_model_file_path, best_model)

            logging.info('Model saved')
            model_trainer_artifact = ModelTrainerArtifact(
                trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                train_metric_artifact=train_classification_metric,
                test_metric_artifact=test_classification_metric,
            )
            
            return model_trainer_artifact
        except Exception as e:
            raise CustomException(e, sys)
    
    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        try:
            logging.info('ModelTrainer initiated')
            train_file_path = self.data_transformation_artifact.transformed_train_file_path
            test_file_path = self.data_transformation_artifact.transformed_test_file_path
            
            logging.info('Loading train & test numpy array')
            train_arr = load_numpy_array_data(train_file_path)
            test_arr = load_numpy_array_data(test_file_path)
            
            x_train,y_train,x_test,y_test = (
                train_arr[:,:-1],
                train_arr[:,-1],
                test_arr[:,:-1],
                test_arr[:,-1],
            )
            model_trainer_artifact = self.model_train(x_train,y_train,x_test,y_test)
            
            return model_trainer_artifact
        except Exception as e:
            raise CustomException(e, sys)
