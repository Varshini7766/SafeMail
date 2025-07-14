from flask import Flask, request, render_template, redirect, url_for, flash, send_file, make_response, jsonify
import pandas as pd
import os
import io
import json
import base64
from model.model import load_model, predict_emails
from collections import Counter
import numpy as np

app = Flask(__name__)
app.secret_key = 'fraud-email-detector-key'

MODEL_PATH = 'model/rf_model.pkl'
VECTORIZER_PATH = 'model/vectorizer.pkl'

# Global variable to store recent predictions for dashboard
recent_predictions = []

@app.route('/', methods=['GET', 'POST'])
def index():
    global recent_predictions
    results = None
    error = None
    uploaded_filename = None
    results_json = 'null'
    results_b64 = ''
    if not (os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH)):
        error = 'Model or vectorizer not found. Please train the model first.'
        return render_template('index.html', results=results, error=error, uploaded_filename=uploaded_filename, results_json=results_json, results_b64=results_b64)
    if request.method == 'POST':
        try:
            model, vectorizer = load_model(MODEL_PATH, VECTORIZER_PATH)
            if 'file' in request.files and request.files['file'].filename != '':
                file = request.files['file']
                uploaded_filename = file.filename
                file.seek(0)
                try:
                    df = pd.read_csv(file.stream)
                except Exception as e:
                    error = f"Could not read CSV file: {e}"
                    return render_template('index.html', results=None, error=error, uploaded_filename=uploaded_filename, results_json=results_json, results_b64=results_b64)
                if 'text' not in df.columns:
                    error = "CSV must have a 'text' column."
                    return render_template('index.html', results=None, error=error, uploaded_filename=uploaded_filename, results_json=results_json, results_b64=results_b64)
                email_list = [str(x) for x in df['text'].tolist()]
                predictions, confidences = predict_emails(model, vectorizer, email_list)
                results = []
                for text, pred, conf in zip(email_list, predictions, confidences):
                    label = 'FRAUD' if pred == 1 else 'NORMAL'
                    confidence = f"{max(conf):.2f}"
                    results.append({'text': text, 'prediction': label, 'confidence': confidence})
                results_json = json.dumps(results)
                results_b64 = base64.b64encode(results_json.encode('utf-8')).decode('utf-8')
                
                # Store results for dashboard
                recent_predictions.extend(results)
                if len(recent_predictions) > 100:  # Keep only last 100 predictions
                    recent_predictions = recent_predictions[-100:]
                    
            elif 'single_email' in request.form and request.form['single_email'].strip() != '':
                email = request.form['single_email'].strip()
                predictions, confidences = predict_emails(model, vectorizer, [email])
                label = 'FRAUD' if predictions[0] == 1 else 'NORMAL'
                confidence = f"{max(confidences[0]):.2f}"
                results = [{'text': email, 'prediction': label, 'confidence': confidence}]
                results_json = json.dumps(results)
                results_b64 = base64.b64encode(results_json.encode('utf-8')).decode('utf-8')
                
                # Store results for dashboard
                recent_predictions.extend(results)
                if len(recent_predictions) > 100:  # Keep only last 100 predictions
                    recent_predictions = recent_predictions[-100:]
            else:
                error = 'Please upload a CSV or enter an email.'
        except Exception as e:
            error = f'Error processing input: {e}'
    return render_template('index.html', results=results, error=error, uploaded_filename=uploaded_filename, results_json=results_json, results_b64=results_b64)

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

@app.route('/api/dashboard-data')
def dashboard_data():
    global recent_predictions
    
    if not recent_predictions:
        return jsonify({
            'prediction_distribution': {'FRAUD': 0, 'NORMAL': 0},
            'confidence_distribution': {'High': 0, 'Medium': 0, 'Low': 0},
            'recent_predictions': [],
            'total_predictions': 0,
            'fraud_percentage': 0,
            'avg_confidence': 0
        })
    
    # Prediction distribution
    predictions = [r['prediction'] for r in recent_predictions]
    prediction_counts = Counter(predictions)
    
    # Confidence distribution
    confidences = [float(r['confidence']) for r in recent_predictions]
    high_conf = sum(1 for c in confidences if c >= 0.8)
    medium_conf = sum(1 for c in confidences if 0.5 <= c < 0.8)
    low_conf = sum(1 for c in confidences if c < 0.5)
    
    # Recent predictions (last 10)
    recent = recent_predictions[-10:][::-1]  # Reverse to show newest first
    
    # Statistics
    total = len(recent_predictions)
    fraud_count = prediction_counts.get('FRAUD', 0)
    fraud_percentage = (fraud_count / total * 100) if total > 0 else 0
    avg_confidence = np.mean(confidences) if confidences else 0
    
    return jsonify({
        'prediction_distribution': dict(prediction_counts),
        'confidence_distribution': {'High': high_conf, 'Medium': medium_conf, 'Low': low_conf},
        'recent_predictions': recent,
        'total_predictions': total,
        'fraud_percentage': round(fraud_percentage, 1),
        'avg_confidence': round(avg_confidence, 2)
    })

@app.route('/download', methods=['POST'])
def download():
    try:
        results_b64 = request.form.get('download_data')
        if results_b64 is None or results_b64 == '':
            return "No results to download.", 400
        results_json = base64.b64decode(results_b64).decode('utf-8')
        results = json.loads(results_json)
        df = pd.DataFrame(results)
        output = io.StringIO()
        df.to_csv(output, index=False)
        output.seek(0)
        response = make_response(output.getvalue())
        response.headers['Content-Disposition'] = 'attachment; filename=fraud_detection_results.csv'
        response.headers['Content-type'] = 'text/csv'
        return response
    except Exception as e:
        return f"Error generating download: {e}", 500

if __name__ == '__main__':
    app.run(debug=True) 