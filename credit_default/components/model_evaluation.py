from credit_default.entity import artifact_entity, config_entity
from credit_default.exception import CustomException
from credit_default.logger import logging
from credit_default.utils import load_object, load_numpy_array_data
from credit_default.predictor import ModelResolver
import sys
import pandas as pd
import numpy as np
from credit_default.config import TARGET_COLUMN
from sklearn.metrics import f1_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import RobustScaler

class ModelEvaluation:
    def __init__(self,
                 model_eval_config: config_entity.ModelEvaluationConfig,
                 data_ingestion_artifact: artifact_entity.DataIngestionArtifact,
                 data_transformation_artifact: artifact_entity.DataTransformationArtifact,
                 model_trainer_artifact: artifact_entity.ModelTrainerArtifact):
        try:
            logging.info(f"{'>>' * 20}  Model Evaluation {'<<' * 20}")
            self.model_eval_config = model_eval_config
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_transformation_artifact = data_transformation_artifact
            self.model_trainer_artifact = model_trainer_artifact
            self.model_resolver = ModelResolver()
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_model_evaluation(self, test_arr: np.ndarray) -> artifact_entity.ModelEvaluationArtifact:
     try:
        logging.info("Checking if the saved model folder has a model and comparing the trained models")
        latest_dir_path = self.model_resolver.get_latest_dir_path()
        if latest_dir_path is None:
            model_eval_artifact = artifact_entity.ModelEvaluationArtifact(
                is_model_accepted=True,
                improved_accuracy=None
            )
            logging.info(f"Model evaluation artifact: {model_eval_artifact}")
            return model_eval_artifact

        logging.info("Finding the locations of the transformer model and model")
        transformer_path = self.model_resolver.get_latest_transformer_path()
        model_path = self.model_resolver.get_latest_model_path()

        logging.info("Loading the previously trained transformer and model")
        transformer = load_object(file_path=transformer_path)
        model = load_object(file_path=model_path)

        logging.info("Loading the currently trained transformer and model")
        current_transformer = load_object(file_path=self.data_transformation_artifact.transform_object_path)
        current_model = load_object(file_path=self.model_trainer_artifact.model_path)

        logging.info("Calculating F1 score using the current trained model")
        y_pred_current = current_model.predict(test_arr[:, :-1])  # Exclude the target column
        print(f"Prediction using the trained model: {y_pred_current[:5]}")
        current_model_score = f1_score(y_true=test_arr[:, -1], y_pred=y_pred_current)
        logging.info(f"F1 score using the current trained model: {current_model_score}")

        if latest_dir_path:
            logging.info("Calculating F1 score using the previous trained model")
            y_pred_previous = model.predict(test_arr[:, :-1])  # Exclude the target column
            print(f"Prediction using previous model: {y_pred_previous[:5]}")
            previous_model_score = f1_score(y_true=test_arr[:, -1], y_pred=y_pred_previous)
            logging.info(f"F1 score using previous trained model: {previous_model_score}")

            if current_model_score < previous_model_score:
                logging.info("The current trained model is not better than the previous model")
                raise Exception("The current trained model is not better than the previous model")

        model_eval_artifact = artifact_entity.ModelEvaluationArtifact(
            is_model_accepted=True,
            improved_accuracy=current_model_score - previous_model_score
        )
        logging.info(f"Model evaluation artifact: {model_eval_artifact}")
        return model_eval_artifact

     except Exception as e:
        raise CustomException(e, sys)


