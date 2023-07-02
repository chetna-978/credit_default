import os
import sys
from datetime import datetime
import pandas as pd
from credit_default.logger import logging
from credit_default.exception import CustomException
from credit_default.predictor import ModelResolver
from credit_default.utils import load_object
from sklearn.preprocessing import RobustScaler
from credit_default.components.data_transformation import DataTransformation

PREDICTION_DIR = "prediction"

def apply_transformations(df, data_transformation, robust_scaler):
    transformed_df = data_transformation.data_modified(df)
    transformed_df = robust_scaler.transform(transformed_df)
    return transformed_df

def start_batch_prediction(input_file_path):
    try:
        os.makedirs(PREDICTION_DIR, exist_ok=True)
        logging.info("Creating model resolver object")
        model_resolver = ModelResolver(model_registry="saved_models")
        logging.info(f"Reading file: {input_file_path}")
        df = pd.read_csv(input_file_path)

        logging.info("Loading data transformation object")
        transformation_path = model_resolver.get_latest_transformer_path()
        data_transformation = DataTransformation.load_object(file_path=transformation_path)

        logging.info("Loading robust scaler object")
        robust_scaler = load_object(file_path=data_transformation.data_transformation_config.transform_object_path)

        logging.info("Applying transformations")
        transformed_df = apply_transformations(df, data_transformation, robust_scaler)

        logging.info("Loading model to make prediction")
        model = load_object(file_path=model_resolver.get_latest_model_path())

        prediction = model.predict(transformed_df)

        df["prediction"] = prediction

        prediction_file_name = os.path.basename(input_file_path).replace(".csv", f"{datetime.now().strftime('%m%d%Y__%H%M%S')}.csv")
        prediction_file_path = os.path.join(PREDICTION_DIR, prediction_file_name)
        df.to_csv(prediction_file_path, index=False, header=True)

        return prediction_file_path
    except Exception as e:
        raise CustomException(e, sys)
