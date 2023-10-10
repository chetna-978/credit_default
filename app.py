from flask import Flask, render_template, request, jsonify, Response, make_response
from flask_cors import CORS
import pandas as pd
from credit_default.logger import logging
from credit_default.exception import CustomException
from credit_default.utils import load_object
from credit_default.predictor import ModelResolver
from datetime import datetime
from credit_default.components.data_transformation import DataTransformation
import os

from credit_default.pipeline.prediction_pipeline import BatchPredictionPipeline

app = Flask(__name__)
CORS(app) 
app.static_folder = 'static'

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST','GET'])
def predict():
    if request.method == 'GET':
        return render_template('index.html')
    
    elif request.method == 'POST':
        try:
            # Extract form data
            id = int(request.form['ID'])
            limit_bal = float(request.form['LIMIT_BAL'])
            sex = int(request.form['SEX'])
            education = int(request.form['EDUCATION'])
            marriage = int(request.form['MARRIAGE'])
            age = int(request.form['AGE'])
            pay_0 = int(request.form['PAY_0'])
            pay_2 = int(request.form['PAY_2'])
            pay_3 = int(request.form['PAY_3'])
            pay_4 = int(request.form['PAY_4'])
            pay_5 = int(request.form['PAY_5'])
            pay_6 = int(request.form['PAY_6'])
            bill_amt1 = float(request.form['BILL_AMT1'])
            bill_amt2 = float(request.form['BILL_AMT2'])
            bill_amt3 = float(request.form['BILL_AMT3'])
            bill_amt4 = float(request.form['BILL_AMT4'])
            bill_amt5 = float(request.form['BILL_AMT5'])
            bill_amt6 = float(request.form['BILL_AMT6'])
            pay_amt1 = float(request.form['PAY_AMT1'])
            pay_amt2 = float(request.form['PAY_AMT2'])
            pay_amt3 = float(request.form['PAY_AMT3'])
            pay_amt4 = float(request.form['PAY_AMT4'])
            pay_amt5 = float(request.form['PAY_AMT5'])
            pay_amt6 = float(request.form['PAY_AMT6'])

            # Prepare the input data for prediction (create a DataFrame)
            input_data = pd.DataFrame({
                'ID': [id],
                'LIMIT_BAL': [limit_bal],
                'SEX': [sex],
                'EDUCATION': [education],
                'MARRIAGE': [marriage],
                'AGE': [age],
                'PAY_0': [pay_0],
                'PAY_2': [pay_2],
                'PAY_3': [pay_3],
                'PAY_4': [pay_4],
                'PAY_5': [pay_5],
                'PAY_6': [pay_6],
                'BILL_AMT1': [bill_amt1],
                'BILL_AMT2': [bill_amt2],
                'BILL_AMT3': [bill_amt3],
                'BILL_AMT4': [bill_amt4],
                'BILL_AMT5': [bill_amt5],
                'BILL_AMT6': [bill_amt6],
                'PAY_AMT1': [pay_amt1],
                'PAY_AMT2': [pay_amt2],
                'PAY_AMT3': [pay_amt3],
                'PAY_AMT4': [pay_amt4],
                'PAY_AMT5': [pay_amt5],
                'PAY_AMT6': [pay_amt6]
            })
       
            pipeline = BatchPredictionPipeline()
            prediction = pipeline.predict(input_data)
            results = "default" if prediction == 1 else "not default"


            if 'application/json' in request.accept_mimetypes:
                return jsonify({'prediction': results})  # Return results in JSON
            else:
                return render_template('result.html', results=results)

        except Exception as e:
            return render_template('error.html', error=str(e))

    return jsonify({'error': 'Invalid request'}), 400

if __name__ == '__main__':
    port = os.environ.get('PORT', '8080')
    if not port:
        port = 8080
    else:
        port = int(port)
    app.run(host="0.0.0.0", port=port, debug=False)
