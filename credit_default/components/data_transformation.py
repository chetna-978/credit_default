from credit_default.entity import artifact_entity, config_entity
from credit_default.exception import CustomException
from credit_default.logger import logging
from typing import Optional
import os, sys 
from sklearn.pipeline import Pipeline
import pandas as pd
from credit_default import utils
import numpy as np
from sklearn.preprocessing import RobustScaler
from credit_default.config import TARGET_COLUMN


class DataTransformation:
    def __init__(self, data_transformation_config: config_entity.DataTransformationConfig,
                 data_ingestion_artifact: artifact_entity.DataIngestionArtifact):
        try:
            logging.info(f"{'>>'*20} Data Transformation {'<<'*20}")
            self.data_transformation_config = data_transformation_config
            self.data_ingestion_artifact = data_ingestion_artifact
        except Exception as e:
            raise CustomException(e, sys)


    @classmethod
    def get_data_transformer_object(cls) -> Pipeline:
        try:
            robust_scaler = RobustScaler()
            pipeline = Pipeline(steps=[
                ('RobustScaler', robust_scaler)
            ])
            return pipeline
        except Exception as e:
            raise CustomException(e, sys)

    def data_modified(self, df: pd.DataFrame):
        try:
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

    def initiate_data_transformation(self) -> artifact_entity.DataTransformationArtifact:
        try:
            # reading training and testing file
            train_df = pd.read_csv(self.data_ingestion_artifact.train_file_path)
            test_df = pd.read_csv(self.data_ingestion_artifact.test_file_path)

            # modifying data
            train_df = self.data_modified(df=train_df)
            test_df = self.data_modified(df=test_df)

            # selecting input feature for train and test dataframe
            input_feature_train_df = train_df.drop(TARGET_COLUMN, axis=1)
            input_feature_test_df = test_df.drop(TARGET_COLUMN, axis=1)

            # selecting target feature for train and test dataframe
            target_feature_train_df = train_df[TARGET_COLUMN]
            target_feature_test_df = test_df[TARGET_COLUMN]

            transformation_pipeline = DataTransformation.get_data_transformer_object()
            transformation_pipeline.fit(input_feature_train_df)

            # transforming input features
            input_feature_train_arr = transformation_pipeline.transform(input_feature_train_df)
            input_feature_test_arr = transformation_pipeline.transform(input_feature_test_df)

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            utils.save_object(file_path=self.data_transformation_config.transform_object_path,
                              obj=transformation_pipeline)
        
            data_transformation_artifact = artifact_entity.DataTransformationArtifact(
                transform_object_path=self.data_transformation_config.transform_object_path
            )
            logging.info(f"Data transformation object {data_transformation_artifact}")
            return data_transformation_artifact, train_arr, test_arr
        except Exception as e:
            raise CustomException(e, sys)
