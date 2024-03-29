import pandas as pd
from credit_default.logger import logging
from credit_default.exception  import  CustomException
from credit_default.config import mongo_client
import os,sys
import yaml
from scipy import stats
from sklearn.ensemble import IsolationForest
import numpy as np
from sklearn.model_selection import RandomizedSearchCV
import dill
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report,precision_score, recall_score, f1_score, roc_auc_score,roc_curve


def get_collection_as_dataframe(database_name:str,collection_name:str)->pd.DataFrame:
    """
    Description: This function return collection as dataframe
    =========================================================
    Params:
    database_name:  database name
    collection_name:  collection name
    =========================================================
    return Pandas dataframe of a collection
    """
    try:
        logging.info(f"Reading data from database: {database_name} and collection: {collection_name}")
        df = pd.DataFrame(list(mongo_client[database_name][collection_name].find()))
        logging.info(f"Found columns: {df.columns}")
        if "_id" in df.columns:
            logging.info(f"Dropping column: _id ")
            df = df.drop("_id",axis=1)
        logging.info(f"Row and columns in df: {df.shape}")
        return df
    except Exception as e:
        raise CustomException(e, sys)
    

def write_yaml_file(file_path,data:dict):
    try:
        file_dir = os.path.dirname(file_path)
        os.makedirs(file_dir,exist_ok=True)
        with open(file_path,"w") as file_writer:
            yaml.dump(data,file_writer)
    except Exception as e:
        raise CustomException(e, sys)

def save_object(file_path: str, obj: object) -> None:
    try:
        logging.info("Entered the save_object method of utils")
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)
        logging.info("Exited the save_object method of utils")
    except Exception as e:
        raise CustomException(e, sys) from e


def load_object(file_path: str, ) -> object:
    try:
        if not os.path.exists(file_path):
            raise Exception(f"The file: {file_path} does not exists")
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)
    except Exception as e:
        raise CustomException(e, sys) from e

def save_numpy_array_data(file_path: str, array: np.array):
    """
    Save numpy array data to file
    file_path: str location of file to save
    array: np.array data to save
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            np.save(file_obj, array)
    except Exception as e:
        raise CustomException(e, sys) from e

def load_numpy_array_data(file_path: str,allow_pickle=True) -> np.array:
    """
    load numpy array data from file
    file_path: str location of file to load
    return: np.array data loaded
    """
    try:
        with open(file_path, "rb") as file_obj:
            return np.load(file_obj)
    except Exception as e:
        raise CustomException(e, sys) from e
    
def percentile_capping(df, cols, from_low_end, from_high_end):
    for col in cols:
      stats.mstats.winsorize(a=df[col], limits=(from_low_end, from_high_end), inplace=True)

def evaluate_clf(true, predicted):
    '''
    This function takes in true values and predicted values
    Returns: Accuracy, F1-Score, Precision, Recall, Roc-auc Score
    '''
    acc = accuracy_score(true, predicted) # Calculate Accuracy
    f1 = f1_score(true, predicted) # Calculate F1-score
    precision = precision_score(true, predicted) # Calculate Precision
    recall = recall_score(true, predicted)  # Calculate Recall
    roc_auc = roc_auc_score(true, predicted) #Calculate Roc
    return acc, f1 , precision, recall, roc_auc


def evaluate_models(X_train, X_test, y_train, y_test, models, param):
    '''
    This function takes in X and y and models dictionary as input
    It splits the data into Train Test split
    Iterates through the given model dictionary and evaluates the metrics
    Returns: Dictionary which contains the model names as keys and test model scores as values
    '''

    report = {}

    for model_name, model in models.items():
        param_grid = param.get(model_name, {})
        
        # Replace GridSearchCV with RandomizedSearchCV
        randomized_search = RandomizedSearchCV(estimator=model,
                                               param_distributions=param_grid,
                                               n_iter=9,  # Specify the number of iterations
                                               cv=3
                                               )
        randomized_search.fit(X_train, y_train)

        best_model = randomized_search.best_estimator_
        best_model.fit(X_train, y_train)

        # Make predictions
        y_train_pred = best_model.predict(X_train)
        y_test_pred = best_model.predict(X_test)

        # Test set performance
        test_model_score = evaluate_clf(y_test, y_test_pred)
        report[model_name] =  test_model_score

    return report


def anomaly_detection(self,df):
     try: 
    # Create an instance of the Isolation Forest algorithm
      isolation_forest = IsolationForest()

    # Fit the model on the training data
      isolation_forest.fit(df)

    # Predict anomalies (-1) and normal instances (1) on the testing data
      predictions = isolation_forest.predict(df)
  
      predictions = isolation_forest.predict(df)  # Use your test data
    
      # Add the predictions to your DataFrame
      df_test = df.copy()
      # Convert input_feature_test_arr to a DataFrame
      #df_test = pd.DataFrame(input_feature_test_df, columns=input_feature_train_df.columns.tolist())

      df_test['isolation_forest_prediction'] = predictions

    # Now you can analyze the predictions, for example, by counting anomalies and normals
      anomaly_count = (df_test['isolation_forest_prediction'] == -1).sum()
      normal_count = (df_test['isolation_forest_prediction'] == 1).sum()

    # Remove rows predicted as anomalies (-1)
      df_cleaned_test = df_test[df_test['isolation_forest_prediction'] == 1].copy()

    # Drop the 'isolation_forest_prediction' column as it's no longer needed
      df_cleaned_test.drop(columns = ['isolation_forest_prediction'], inplace=True)
     except Exception as e:
            raise CustomException(e, sys)
