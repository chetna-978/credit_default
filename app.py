from flask import Flask, render_template, request, jsonify
import pandas as pd
from credit_default.logger import logging
from credit_default.exception import CustomException
from credit_default.utils import load_object
from credit_default.predictor import ModelResolver
from datetime import datetime
from credit_default.components.data_transformation import DataTransformation
import os

app = Flask(__name__)

PREDICTION_DIR = "prediction"

def apply_transformations(df, data_transformation, robust_scaler):
    transformed_df = data_transformation.data_modified(df)
    transformed_df = robust_scaler.transform(transformed_df)
    return transformed_df

def start_batch_prediction(input_file_path):
    try:
        os.makedirs(PREDICTION_DIR, exist_ok=True)
        logging.info("Creating model resolver object")
        model_resolver = ModelResolver(model_registry="saved_models")
        logging.info(f"Reading file: {input_file_path}")
        df = pd.read_csv(input_file_path)

        logging.info("Loading data transformation object")
        transformation_path = model_resolver.get_latest_transformer_path()
        data_transformation = DataTransformation.load_object(file_path=transformation_path)

        logging.info("Loading robust scaler object")
        robust_scaler = load_object(file_path=data_transformation.data_transformation_config.transform_object_path)

        logging.info("Applying transformations")
        transformed_df = apply_transformations(df, data_transformation, robust_scaler)

        logging.info("Loading model to make prediction")
        model = load_object(file_path=model_resolver.get_latest_model_path())

        prediction = model.predict(transformed_df)

        df["prediction"] = prediction

        prediction_file_name = os.path.basename(input_file_path).replace(".csv", f"{datetime.now().strftime('%m%d%Y__%H%M%S')}.csv")
        prediction_file_path = os.path.join(PREDICTION_DIR, prediction_file_name)
        df.to_csv(prediction_file_path, index=False, header=True)

        return prediction_file_path
    except Exception as e:
        raise CustomException(e, sys)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'GET':
        return render_template('index.html')
    elif request.method == 'POST':
        try:
            file = request.files['file']
            file_path = f"uploads/{file.filename}"
            file.save(file_path)

            data = pd.read_csv(file_path)

            prediction = start_batch_prediction(data)

            return render_template('index.html', prediction=round(prediction, 2))
        except Exception as e:
            return jsonify({'error': str(e)})

@app.route('/predictAPI', methods=['POST'])
def predict_api():
    if request.method == 'POST':
        try:
            file = request.files['file']
            file_path = f"uploads/{file.filename}"
            file.save(file_path)

            data = pd.read_csv(file_path)

            prediction = start_batch_prediction(data)

            return jsonify({'prediction': round(prediction, 2)})
        except Exception as e:
            return jsonify({'error': str(e)})

if __name__ == '__main__':
    port = os.environ.get('PORT', '8080')
    if not port:
        port = 8080
    else:
        port = int(port)
    logging.info(f"Flask application running on port {port}")
    app.run(host="0.0.0.0", port=port, debug=False)