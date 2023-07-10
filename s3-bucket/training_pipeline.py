import json
from textwrap import dedent
import pendulum
import os
import sys


def training():
    from credit_default.pipeline.training_pipeline import start_training_pipeline
    start_training_pipeline()


def sync_artifact_to_s3_bucket():
    bucket_name = os.getenv("BUCKET_NAME")
    os.system(f"aws s3 sync /app/artifact s3://{bucket_name}/artifacts")
    os.system(f"aws s3 sync /app/saved_models s3://{bucket_name}/saved_models")


if __name__ == "__main__":
    training()
    sync_artifact_to_s3_bucket()
