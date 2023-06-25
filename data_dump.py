import pymongo
import pandas as pd
import  json
client = pymongo.MongoClient("mongodb://localhost:27017/neurolabDB")


DATA_FILE_PATH = "/config/workspace/UCI_Credit_Card.csv"
DATABASE_NAME = "credit_card_default_prediction"
COLLECTION_NAME = "credit_default"


if __name__=="__main__":
    df = pd.read_csv(DATA_FILE_PATH)
    print(f"Rows and Columns: {df.shape}")

    #Convert dataframe to json so that we can dump these records in mongodb  
    df.reset_index(drop=True,inplace=True)

    json_record = list(json.loads(df.T.to_json()).values())
    print(json_record[0])

    #insert converted json record to mongodb
    client[DATABASE_NAME][COLLECTION_NAME].insert_many(json_record)
