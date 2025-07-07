import sys
from sklearn.metrics import f1_score,precision_score,recall_score

from src.exception import CustomException
from src.entity.artifact_entity import ClassificationMetricArtifact

def get_classfication_score(y_true, y_pred) -> ClassificationMetricArtifact:
    try:
        model_f1_score = f1_score(y_true, y_pred)
        model_precision_score = precision_score(y_true, y_pred)
        model_recall_score = recall_score(y_pred, y_true)
        
        return ClassificationMetricArtifact(
            f1_score=float(model_f1_score),
            precision_score=float(model_precision_score),
            recall_score=float(model_recall_score),
        )
    except Exception as e:
        raise CustomException(e, sys)
