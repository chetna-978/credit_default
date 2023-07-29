from flask import Flask, render_template, request, jsonify
import pandas as pd
from credit_default.logger import logging
from credit_default.exception import CustomException
from credit_default.utils import load_object
from credit_default.pipeline.prediction_pipeline import BatchPredictionPipeline
from datetime import datetime
import os
import sys

app = Flask(__name__)

# Create an instance of BatchPredictionPipeline
predict_pipeline = BatchPredictionPipeline()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'GET':
        return render_template('index.html')
    else:
        data = {
            'limit_bal': float(request.form.get('LIMIT_BAL')),
            'sex': int(request.form.get('SEX')),
            'education': int(request.form.get('EDUCATION')),
            'marriage': int(request.form.get('MARRIAGE')),
            'age': int(request.form.get('AGE')),
            'pay_0': int(request.form.get('PAY_0')),
            'pay_2': int(request.form.get('PAY_2')),
            'pay_3': int(request.form.get('PAY_3')),
            'pay_4': int(request.form.get('PAY_4')),
            'pay_5': int(request.form.get('PAY_5')),
            'pay_6': int(request.form.get('PAY_6')),
            'BILL_AMT1': int(request.form.get('BILL_AMT1')),
            'BILL_AMT2': int(request.form.get('BILL_AMT2')),
            'BILL_AMT3': int(request.form.get('BILL_AMT3')),
            'BILL_AMT4': int(request.form.get('BILL_AMT4')),
            'BILL_AMT5': int(request.form.get('BILL_AMT5')),
            'BILL_AMT6': int(request.form.get('BILL_AMT6')),
            'PAY_AMT1': int(request.form.get('PAY_AMT1')),
            'PAY_AMT2': int(request.form.get('PAY_AMT2')),
            'PAY_AMT3': int(request.form.get('PAY_AMT3')),
            'PAY_AMT4': int(request.form.get('PAY_AMT4')),
            'PAY_AMT5': int(request.form.get('PAY_AMT5')),
            'PAY_AMT6': int(request.form.get('PAY_AMT6')),
        }

        try:
            prediction = predict_pipeline.predict(data)
            # Format the prediction result to display
            result = "Default" if prediction[0] == 1 else "Not Default"
            return render_template('index.html', results=result)
        except Exception as e:
            return render_template('index.html', error_message=str(e))

if __name__ == '__main__':
    port = os.environ.get('PORT', '8080')
    if not port:
        port = 8080
    else:
        port = int(port)
    logging.info(f"Flask application running on port {port}")
    app.run(host="0.0.0.0", port=port, debug=False)
