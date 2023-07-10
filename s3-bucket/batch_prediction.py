import json
from textwrap import dedent
import pendulum
import os
import datetime
import sys
from credit_default.components.data_transformation import DataTransformation


def download_files(**kwargs):
    bucket_name = os.getenv("BUCKET_NAME")
    input_dir = "/app/input_files"
    # creating directory
    os.makedirs(input_dir, exist_ok=True)
    os.system(f"aws s3 sync s3://{bucket_name}/input_files /app/input_files")


def batch_prediction(**kwargs):
    from credit_default.pipeline.prediction_pipeline import BatchPredictionPipeline
    input_dir = "/app/input_files"
    for file_name in os.listdir(input_dir):
        # make prediction
        start_batch_prediction(input_file=os.path.join(input_dir, file_name))


def sync_prediction_dir_to_s3_bucket(**kwargs):
    bucket_name = os.getenv("BUCKET_NAME")
    # upload prediction folder to predictionfiles folder in s3 bucket
    os.system(f"aws s3 sync /app/prediction s3://{bucket_name}/prediction_files")


if __name__ == "__main__":
    download_files()
    batch_prediction()
    sync_prediction_dir_to_s3_bucket()
