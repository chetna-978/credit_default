from credit_default.entity import artifact_entity, config_entity
from credit_default.exception import CustomException
from credit_default.logger import logging
from typing import Optional
import os, sys 
import numpy as np
from credit_default import utils
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, f1_score, roc_auc_score, roc_curve


class ModelTrainer:
    models = {
        "AdaBoost Classifier": AdaBoostClassifier(),
        "Random-forest Classifier": RandomForestClassifier()   
    }

    param = {
    
       'AdaBoost Classifier': { 'n_estimators': [10,50, 100,200,500],
                                'learning_rate': [0.0001, 0.001,0.01, 0.1,1.0]
                             },
        

       'Random-forest Classifier':{'n_estimators' : [20,60,100,120],
                                    'max_features' : [0.2,0.6,1.0,'sqrt','log2'],
                                    'max_samples' : [0.5,0.75,1.0]
                                    }               

    }

    def __init__(self, model_trainer_config: config_entity.ModelTrainerConfig,
                 data_transformation_artifact: artifact_entity.DataTransformationArtifact):
        try:
            logging.info(f"{'>>'*20} Model Trainer {'<<'*20}")
            self.model_trainer_config = model_trainer_config
            self.data_transformation_artifact = data_transformation_artifact
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_model_trainer(self, train_array, test_array) -> artifact_entity.ModelTrainerArtifact:
     try:
        logging.info(f"Splitting input and target feature from both train and test arr.")
        X_train, y_train = train_array[:, :-1], train_array[:, -1]
        X_test, y_test = test_array[:, :-1], test_array[:, -1]

        model_report = utils.evaluate_models(
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            models=ModelTrainer.models,
            param=ModelTrainer.param
        )
        
        # To get the best model score from the dictionary
        best_model_name = max(model_report, key=model_report.get)
        best_model_score = model_report[best_model_name]
        
        if best_model_score[0] < 0.70:
            raise Exception("No best model found")
        
        # Create and fit the best model
        best_model = ModelTrainer.models[best_model_name]
        best_model.fit(X_train, y_train)
        
        # Save the trained model
        logging.info(f"Saving model object")
        utils.save_object(file_path=self.model_trainer_config.model_path, obj=best_model)

        # Prepare the artifact
        logging.info(f"Prepare the artifact")
        model_trainer_artifact = artifact_entity.ModelTrainerArtifact(model_path=self.model_trainer_config.model_path)
        logging.info(f"Model trainer artifact: {model_trainer_artifact}")
        return model_trainer_artifact
     except Exception as e:
        raise CustomException(e, sys)
