from credit_default.pipeline.training_pipeline import start_training_pipeline
from credit_default.pipeline.prediction_pipeline import BatchPredictionPipeline

file_path= "/config/workspace/UCI_Credit_Card.csv"
print(__name__)
if __name__=="__main__":
     try:
          #start_training_pipeline()
          pipeline = BatchPredictionPipeline()
          prediction_file = pipeline.start_batch_prediction(input_file=file_path)
          print(prediction_file)
     except Exception as e:
          print(e) 