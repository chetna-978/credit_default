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

            logging.info("Transforming the test data using the current transformer")
        
               # Transforming the test data using the current transformer
            transformed_test_arr = current_transformer.named_steps['RobustScaler'].transform(test_arr[:, :-1])

               # Ensure the number of features in the test data matches the expected number
            if transformed_test_arr.shape[1] != current_transformer.named_steps['RobustScaler'].n_features_in_:
             raise ValueError("Number of features in the test data is not compatible with the transformer")

            # Add the target column back to the transformed test data
            transformed_test_arr_with_target = np.hstack((transformed_test_arr, test_arr[:, -1].reshape(-1, 1)))

            logging.info("Calculating accuracy using the current trained model")
            y_pred = current_model.predict(transformed_test_arr_with_target[:, :-1])  # Predict using the current model
            print(f"Prediction using the trained model: {y_pred[:5]}")
            current_model_score = f1_score(y_true=test_arr[:, -1], y_pred=y_pred)
            logging.info(f"Accuracy using the current trained model: {current_model_score}")

            if latest_dir_path:
                logging.info("Calculating accuracy using the previous trained model")
                print(f"Number of features in test data: {test_arr.shape[1]}")
                print(f"Number of features in transformed test data: {transformed_test_arr.shape[1]}")
                
                input_arr = transformer.transform(test_arr[:, :-1])  # Exclude the target column during transformation
                y_pred = model.predict(input_arr)
                print(f"Prediction using previous model: {y_pred[:5]}")
                previous_model_score = f1_score(y_true=test_arr[:, -1], y_pred=y_pred)
                logging.info(f"Accuracy using previous trained model: {previous_model_score}")

                if float(current_model_score) <= float(previous_model_score):
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

