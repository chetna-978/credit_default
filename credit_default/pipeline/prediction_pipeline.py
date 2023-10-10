import os
import sys
from datetime import datetime
import pandas as pd
from credit_default.exception import CustomException
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from credit_default.predictor import ModelResolver
from credit_default.utils import load_object
from sklearn.preprocessing import RobustScaler
from credit_default.logger import logging
from credit_default.config import TARGET_COLUMN

class BatchPredictionPipeline:
    def __init__(self):
        self.PREDICTION_DIR = "prediction"
        self.model_resolver = ModelResolver(model_registry="saved_models")

    def handle_unknown_categories(self, test_data: pd.DataFrame):
       # Define a predefined category for unknown values
       predefined_category = 'Unknown'  # You can change this to any desired category name

       # Map unknown categories in the test data to the predefined category
       for column in ['REPAY_1_Category', 'REPAY_2_Category', 'REPAY_3_Category', 'REPAY_4_Category', 'REPAY_5_Category', 'REPAY_6_Category']:
        test_data[column] = test_data[column].apply(lambda x: predefined_category if x not in ['Category1', 'Category2', 'Category3'] else x)

    def get_data_transformer_object(self):
        try:
            columns_to_encode = ['REPAY_1_Category', 'REPAY_2_Category','REPAY_3_Category', 'REPAY_4_Category', 'REPAY_5_Category','REPAY_6_Category']
           
            #robust_scaler = RobustScaler()

            encoder_pipeline = Pipeline(steps=[
             ("one_hot_encoder",OneHotEncoder(handle_unknown='ignore'))
             ])

            columns_for_scaling = ['LIMIT_BAL','BILL_AMT1','BILL_AMT2','BILL_AMT3','BILL_AMT4','PAY_AMT1','PAY_AMT2','PAY_AMT3','PAY_AMT4','PAY_AMT6']

            pipeline = Pipeline(steps=[
                ('RobustScaler', RobustScaler())
            ])
            

            transformation_pipeline=ColumnTransformer(
                [
                ("encoder_pipeline",encoder_pipeline,columns_to_encode),
                ("pipeline",pipeline,columns_for_scaling)

                ]
                  )

            return transformation_pipeline 
 
        except Exception as e:
            raise CustomException(e, sys)

    def data_modified(self, df: pd.DataFrame):
        try:

            df['grad_school'] = (df['EDUCATION'] == 1).astype(int)
            df['university'] = (df['EDUCATION'] == 2).astype(int)
            df['high_school'] = (df['EDUCATION'] == 3).astype(int)
            df['others'] = ((df['EDUCATION'] == 4) | (df['EDUCATION'] == 5) | (df['EDUCATION'] == 6) | (df['EDUCATION'] == 0)).astype(int)
            df.drop('EDUCATION', axis=1, inplace=True)

            df['married'] = (df['MARRIAGE'] == 1).astype(int)
            df['single'] = (df['MARRIAGE'] == 2).astype(int)
            df['na'] = ((df['MARRIAGE'] == 3) | (df['MARRIAGE'] == 0)).astype(int)
            df.drop(['MARRIAGE'], axis=1, inplace=True)
            
            # Define mapping for PAY values to categories
            category_mapping = {-2: 'No Dues', -1: 'No Dues', 0: 'Duly Paid', 1: 'One Month Delay', 2: 'Two Months Delay', 3: 'Three Months Delay'}
            # Create a category for values >= 4
            df['REPAY_1_Category'] = df['PAY_0'].apply(lambda x: 'Four or More Months Delay' if x >= 4 else category_mapping.get(x, 'Unknown'))
            df['REPAY_2_Category'] = df['PAY_2'].apply(lambda x: 'Four or More Months Delay' if x >= 4 else category_mapping.get(x, 'Unknown'))
            df['REPAY_3_Category'] = df['PAY_3'].apply(lambda x: 'Four or More Months Delay' if x >= 4 else category_mapping.get(x, 'Unknown'))
            df['REPAY_4_Category'] = df['PAY_4'].apply(lambda x: 'Four or More Months Delay' if x >= 4 else category_mapping.get(x, 'Unknown'))
            df['REPAY_5_Category'] = df['PAY_5'].apply(lambda x: 'Four or More Months Delay' if x >= 4 else category_mapping.get(x, 'Unknown'))
            df['REPAY_6_Category'] = df['PAY_6'].apply(lambda x: 'Four or More Months Delay' if x >= 4 else category_mapping.get(x, 'Unknown'))
            columns_to_drop = ['PAY_0', 'PAY_2','PAY_3', 'PAY_4', 'PAY_5', 'PAY_6', 'BILL_AMT5', 'BILL_AMT6', 'PAY_AMT5','SEX','single']
            df.drop(columns=columns_to_drop, inplace=True)
            return df
        except Exception as e:
            raise CustomException(e, sys)

    def start_batch_prediction(self, input_file):
        try:
            os.makedirs(self.PREDICTION_DIR, exist_ok=True)
            df = pd.read_csv(input_file)
            logging.info(f"Reading file: {input_file}")
            
             # Drop id column
            df.drop(['ID'],axis=1,inplace=True)

            # Drop target column if it exists
            if TARGET_COLUMN in df.columns:
                df.drop(TARGET_COLUMN, axis=1, inplace=True)
                logging.info(f"Target column removed")
  
            # Data modification
            data = self.data_modified(df)

            # Load the latest model
            logging.info("Loading the latest model")
            model = load_object(file_path=self.model_resolver.get_latest_model_path())
            transformer = load_object(file_path=self.model_resolver.get_latest_transformer_path())
            
            data_scaled = transformer.transform(df)
            # Perform predictions
            logging.info("Performing predictions")
            prediction = model.predict(data_scaled)
            
            # Add prediction column to the DataFrame
            df["prediction"] = prediction

            prediction_file_name = os.path.basename(input_file).replace(".csv", f"{datetime.now().strftime('%m%d%Y__%H%M%S')}.csv")
            prediction_file_path = os.path.join(self.PREDICTION_DIR, prediction_file_name)
            
            df.to_csv(prediction_file_path, index=False, header=True)
            
            return prediction_file_path
        except Exception as e:
            raise CustomException(e, sys)

    def predict(self,  input_data:pd.DataFrame ):
        try:
            data = pd.DataFrame(input_data)
            # Drop id column
            data.drop(['ID'],axis=1,inplace=True)

            # Data modification
            data = self.data_modified(data)

            # Load the latest model
            logging.info("Loading the latest model")
            model = load_object(file_path=self.model_resolver.get_latest_model_path())
            transformer = load_object(file_path=self.model_resolver.get_latest_transformer_path())
            
            data_scaled = transformer.transform(data)

            # Perform predictions
            logging.info("Performing predictions")
            prediction = model.predict(data_scaled)
            
            return prediction
        except Exception as e:
            raise CustomException(e, sys)
