from credit_default.entity import artifact_entity, config_entity
from credit_default.exception import CustomException
from credit_default.logger import logging
from typing import Optional, Tuple
import os
from sklearn.pipeline import Pipeline
import pandas as pd
from credit_default import utils
from sklearn.preprocessing import OneHotEncoder,RobustScaler
from sklearn.compose import ColumnTransformer
from imblearn.combine import SMOTETomek
import numpy as np
#from sklearn.preprocessing import RobustScaler
from credit_default.config import TARGET_COLUMN
import sys

class DataTransformation:
    def __init__(self, data_transformation_config: config_entity.DataTransformationConfig,
                 data_ingestion_artifact: artifact_entity.DataIngestionArtifact):
        try:
            logging.info(f"{'>>'*20} Data Transformation {'<<'*20}")
            self.data_transformation_config = data_transformation_config
            self.data_ingestion_artifact = data_ingestion_artifact
        except Exception as e:
            raise CustomException(e, sys)
         

    def get_data_transformer_object(self):
        try:
            columns_to_encode = ['REPAY_2_Category','REPAY_3_Category', 'REPAY_4_Category', 'REPAY_5_Category','REPAY_6_Category']
           
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
        
    def handle_unknown_categories(self, train_data: pd.DataFrame, test_data: pd.DataFrame):
        # Identify unique categories in the training and test datasets
        train_categories = set(train_data['REPAY_3_Category'].unique())
        test_categories = set(test_data['REPAY_3_Category'].unique())

        # Find unknown categories in the test dataset
        unknown_categories = test_categories - train_categories

        # Define a predefined category for unknown values
        predefined_category = 'Unknown'  # You can change this to any desired category name

        # Map unknown categories in the test data to the predefined category
        for column in ['REPAY_1_Category', 'REPAY_2_Category', 'REPAY_3_Category', 'REPAY_4_Category', 'REPAY_5_Category', 'REPAY_6_Category']:
            test_data[column] = test_data[column].apply(lambda x: predefined_category if x in unknown_categories else x)


    def initiate_data_transformation(self) -> Tuple[artifact_entity.DataTransformationArtifact, np.ndarray, np.ndarray]:
        try: 

            # reading training and testing file
            train_df = pd.read_csv(self.data_ingestion_artifact.train_file_path)
            test_df = pd.read_csv(self.data_ingestion_artifact.test_file_path)
           
            # modifying data
            self.data_modified(train_df)
            self.data_modified(test_df)

            self.handle_unknown_categories(train_df, test_df)


            # selecting input feature for train and test dataframe
            input_feature_train_df = train_df.drop(TARGET_COLUMN, axis=1)
            input_feature_test_df = test_df.drop(TARGET_COLUMN, axis=1)

            # selecting target feature for train and test dataframe
            target_feature_train_df = train_df[TARGET_COLUMN]
            target_feature_test_df = test_df[TARGET_COLUMN]
            
            logging.info("Obtaining preprocessing object")

            preprocessing_obj=self.get_data_transformer_object()
            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)

            smt = SMOTETomek(random_state=42)
            logging.info(f'Before resampling in training set Input: {input_feature_train_arr.shape} Target:{target_feature_train_df.shape}')
            input_feature_train_arr, target_feature_train_arr = smt.fit_resample(input_feature_train_arr, target_feature_train_df)
            logging.info(f'After resampling in training set Input: {input_feature_train_arr.shape} Target:{target_feature_train_df.shape}')

            logging.info(f'Before resampling in testing set Input: {input_feature_test_arr.shape} Target:{target_feature_test_df.shape}')
            input_feature_test_arr, target_feature_test_arr = smt.fit_resample(input_feature_test_arr,target_feature_test_df)
            logging.info(f'After resampling in testing set Input: {input_feature_test_arr.shape} Target:{target_feature_test_df.shape}')

            train_arr = np.c_[input_feature_train_arr, target_feature_train_arr]
            test_arr = np.c_[input_feature_test_arr, target_feature_test_arr]

            utils.save_object(file_path=self.data_transformation_config.transform_object_path,
                              obj=preprocessing_obj)
        
            data_transformation_artifact = artifact_entity.DataTransformationArtifact(
                transform_object_path=self.data_transformation_config.transform_object_path
            )
            logging.info(f'Data transformation object {data_transformation_artifact}')
            return data_transformation_artifact, train_arr, test_arr
        except Exception as e:
            raise CustomException(e, sys)
