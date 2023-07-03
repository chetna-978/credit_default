import os
import sys
from datetime import datetime
import pandas as pd
from credit_default.exception import CustomException
from credit_default.predictor import ModelResolver
from credit_default.utils import load_object
from sklearn.preprocessing import RobustScaler
from credit_default.logger import logging
from credit_default.config import TARGET_COLUMN

class BatchPredictionPipeline:
    def __init__(self):
        self.PREDICTION_DIR = "prediction"
        self.model_resolver = ModelResolver(model_registry="saved_models")
        self.robust_scaler = RobustScaler()

    def data_modified(self, df):
        try:
            # Perform data modifications
            df['grad_school'] = (df['EDUCATION'] == 1).astype(int)
            df['university'] = (df['EDUCATION'] == 2).astype(int)
            df['high_school'] = (df['EDUCATION'] == 3).astype(int)
            df['others'] = ((df['EDUCATION'] == 4) | (df['EDUCATION'] == 5) | (df['EDUCATION'] == 6) | (df['EDUCATION'] == 0)).astype(int)

            df.drop('EDUCATION', axis=1, inplace=True)

            df['male'] = (df['SEX'] == 'M').astype(int)
            df.drop('SEX', axis=1, inplace=True)

            df['married'] = (df['MARRIAGE'] == 1).astype(int)
            df['single'] = (df['MARRIAGE'] == 2).astype(int)
            df['na'] = ((df['MARRIAGE'] == 3) | (df['MARRIAGE'] == 0)).astype(int)

            df.drop(['MARRIAGE'], axis=1, inplace=True)

            columns_to_drop = ['ID', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5']
            df.drop(columns_to_drop, axis=1, inplace=True)
            return df
        except Exception as e:
            raise CustomException(e, sys)

    def apply_transformations(self, df):
        transformed_df = self.robust_scaler.transform(df)
        return transformed_df

    def start_batch_prediction(self, input_file):
        try:
            os.makedirs(self.PREDICTION_DIR, exist_ok=True)
            df = pd.read_csv(input_file)
            logging.info(f"Reading file: {input_file}")
            
            df.drop(TARGET_COLUMN, axis=1, inplace=True)
            logging.info(f"target column removed")

            logging.info("Creating robust scaler object")
            self.robust_scaler.fit(self.data_modified(df))
            logging.info("Applying transformations")
            transformed_df = self.apply_transformations(df)
            logging.info("Applying transformations")
            model = load_object(file_path=self.model_resolver.get_latest_model_path())
    
            prediction = model.predict(transformed_df)
            
            df["prediction"] = prediction
            
            prediction_file_name = os.path.basename(input_file).replace(".csv", f"{datetime.now().strftime('%m%d%Y__%H%M%S')}.csv")
            prediction_file_path = os.path.join(self.PREDICTION_DIR, prediction_file_name)
            df.to_csv(prediction_file_path, index=False, header=True)
            
            return prediction_file_path
        except Exception as e:
            raise CustomException(e, sys)
