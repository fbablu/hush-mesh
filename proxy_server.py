#!/usr/bin/env python3
from flask import Flask, send_from_directory, request, jsonify
from flask_cors import CORS
import requests
import logging
import os

app = Flask(__name__)
CORS(app)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ML_SERVER_URL = "http://localhost:9000"

@app.route('/')
def index():
    return """
    <html>
    <head><title>ACPS Maritime Demos</title></head>
    <body style="font-family: Arial; padding: 20px;">
        <h1>🚢 ACPS Maritime Threat Detection Demos</h1>
        <ul>
            <li><a href="/enhanced_multi_route.html">Enhanced Multi-Route Planning Demo</a></li>
            <li><a href="/test_ml.html">ML Model Test Page</a></li>
            <li><a href="/ml_monitor.html">ML Model Real-Time Monitor</a></li>
        </ul>
    </body>
    </html>
    """

@app.route('/<path:filename>')
def serve_file(filename):
    try:
        demo_dir = os.path.join(os.path.dirname(__file__), 'demo')
        if os.path.exists(os.path.join(demo_dir, filename)):
            return send_from_directory(demo_dir, filename)
        return send_from_directory('.', filename)
    except Exception as e:
        logger.error(f"Error serving file {filename}: {e}")
        return f"File not found: {filename}", 404

@app.route('/api/health', methods=['GET'])
def health_proxy():
    try:
        response = requests.get(f"{ML_SERVER_URL}/health", timeout=5)
        return jsonify(response.json()), response.status_code
    except requests.exceptions.RequestException as e:
        logger.error(f"Health check failed: {e}")
        return jsonify({'status': 'error', 'message': 'ML server not responding'}), 503

@app.route('/api/predict', methods=['POST'])
def predict_proxy():
    try:
        response = requests.post(f"{ML_SERVER_URL}/predict", json=request.json, timeout=10)
        return jsonify(response.json()), response.status_code
    except requests.exceptions.RequestException as e:
        logger.error(f"Prediction request failed: {e}")
        return jsonify({'error': 'ML server error'}), 503

@app.route('/api/reset', methods=['POST'])
def reset_proxy():
    try:
        response = requests.post(f"{ML_SERVER_URL}/reset", timeout=5)
        return jsonify(response.json()), response.status_code
    except requests.exceptions.RequestException as e:
        return jsonify({'error': str(e)}), 503

if __name__ == '__main__':
    logger.info("Starting proxy server on port 8082...")
    app.run(host='0.0.0.0', port=8082, debug=False)
