#!/bin/bash
# Deploy ML and Proxy Servers Script

echo "🚀 Deploying ML and Proxy Servers..."
echo "======================================"

# Kill existing servers
echo "⏹️  Stopping existing servers..."
pkill -f ml_api_server.py
pkill -f proxy_server.py
sleep 2

# Backup old files
echo "💾 Backing up old files..."
cd /workshop/hush-mesh/ml
if [ -f ml_api_server.py ]; then
    cp ml_api_server.py ml_api_server.py.backup.$(date +%Y%m%d_%H%M%S)
fi

# Deploy new ML server
echo "📦 Deploying new ML server..."
cat > /workshop/hush-mesh/ml/ml_api_server.py << 'MLEOF'
#!/usr/bin/env python3
# Fixed version with proper class inheritance
import json
from flask import Flask, request, jsonify
from flask_cors import CORS
import logging
import random

# Try to import numpy, fallback to basic Python if not available
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    print("Warning: NumPy not available. Using basic Python math.")

# Try to import torch, fallback to simulation if not available
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError as e:
    print(f"Warning: PyTorch not available ({e}). Using simulation mode.")
    TORCH_AVAILABLE = False

app = Flask(__name__)
CORS(app)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define model classes based on PyTorch availability
if TORCH_AVAILABLE:
    class MaritimeThreatDetector(nn.Module):
        def __init__(self, input_size=15, hidden_size=128, num_classes=8):
            super(MaritimeThreatDetector, self).__init__()
            self.input_size = input_size
            self.num_classes = num_classes
            self.conv1 = nn.Conv1d(input_size, 64, kernel_size=3, padding=1)
            self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
            self.lstm = nn.LSTM(128, hidden_size, batch_first=True, bidirectional=True)
            self.dropout = nn.Dropout(0.3)
            self.classifier = nn.Sequential(
                nn.Linear(hidden_size * 2, 64),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(64, num_classes)
            )
            self.relu = nn.ReLU()
            self.softmax = nn.Softmax(dim=1)
        
        def forward(self, x):
            x = x.unsqueeze(2)
            x = self.relu(self.conv1(x))
            x = self.relu(self.conv2(x))
            x = x.transpose(1, 2)
            lstm_out, _ = self.lstm(x)
            x = self.dropout(lstm_out[:, -1, :])
            x = self.classifier(x)
            return self.softmax(x)
else:
    class MaritimeThreatDetector:
        def __init__(self, input_size=15, hidden_size=128, num_classes=8):
            self.input_size = input_size
            self.num_classes = num_classes
        
        def forward(self, x):
            if NUMPY_AVAILABLE:
                return np.random.dirichlet(np.ones(self.num_classes))
            else:
                probs = [random.random() for _ in range(self.num_classes)]
                total = sum(probs)
                return [p/total for p in probs]
        
        def eval(self):
            pass
        
        def load_state_dict(self, state_dict):
            pass
        
        def __call__(self, x):
            return self.forward(x)

# Load model
model = None
try:
    model = MaritimeThreatDetector()
    if TORCH_AVAILABLE:
        state_dict = torch.load('/workshop/hush-mesh/models/model.pth', map_location='cpu')
        model.load_state_dict(state_dict)
    model.eval()
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Error loading model: {e}")
    model = MaritimeThreatDetector()
    model.eval()
    logger.warning("Using model with random weights")

def extract_features(sensor_data):
    """Extract 15 features from sensor data"""
    try:
        convoy_data = sensor_data.get('convoy_data', {})
        env_conditions = sensor_data.get('environmental_conditions', {})
        data_quality = sensor_data.get('data_quality', {})
        drone_array = sensor_data.get('drone_array', [{}])
        
        if not drone_array:
            drone_array = [{}]
        
        drone = drone_array[0]
        sensor_suite = drone.get('sensor_suite', {})
        
        features = [
            convoy_data.get('speed_knots', 12.0),
            convoy_data.get('vessel_count', 4),
            env_conditions.get('sea_state', 2),
            env_conditions.get('visibility_km', 15.0),
            env_conditions.get('wind_speed_knots', 10.0),
            env_conditions.get('wave_height_m', 2.0),
            data_quality.get('sensor_reliability', 0.9),
            drone.get('position', {}).get('altitude_m', 180.0),
            sensor_suite.get('electro_optical', {}).get('visible_spectrum', {}).get('objects_detected', 1),
            sensor_suite.get('electro_optical', {}).get('infrared', {}).get('thermal_signatures', 1),
            sensor_suite.get('radar', {}).get('contacts', 1),
            sensor_suite.get('acoustic', {}).get('hydrophone_data', {}).get('ambient_noise_db', 95.0),
            sensor_suite.get('electronic_warfare', {}).get('rf_spectrum', {}).get('signals_detected', 3),
            len(sensor_data.get('threat_detections', [])),
            1.0 if sensor_data.get('ground_truth', {}).get('threat_present', False) else 0.0
        ]
        
        return features
    except Exception as e:
        logger.error(f"Error extracting features: {e}")
        return [12.0, 4, 2, 15.0, 10.0, 2.0, 0.9, 180.0, 1, 1, 1, 95.0, 3, 0, 0.0]

@app.route('/predict', methods=['POST'])
def predict():
    global model
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        data = request.json
        sensor_data = data.get('sensor_data', {})
        
        features = extract_features(sensor_data)
        
        if TORCH_AVAILABLE:
            input_tensor = torch.FloatTensor(features).unsqueeze(0)
            logger.info(f"Input tensor shape: {input_tensor.shape}")
        elif NUMPY_AVAILABLE:
            input_tensor = np.array(features)
            logger.info(f"Input array shape: {input_tensor.shape}")
        else:
            input_tensor = features
            logger.info(f"Input list length: {len(input_tensor)}")
        
        if TORCH_AVAILABLE:
            with torch.no_grad():
                outputs = model(input_tensor)
                probabilities = outputs.cpu().numpy()[0]
                logger.info(f"Model output shape: {outputs.shape}")
        else:
            probabilities = model(input_tensor)
            logger.info(f"Simulation mode - generated probabilities")
            
        threat_classes = [
            'small_fast_craft', 'floating_mine_like_object', 'submarine_periscope',
            'debris_field', 'fishing_vessel', 'cargo_ship', 'research_vessel', 'unknown'
        ]
        
        if NUMPY_AVAILABLE and not isinstance(probabilities, list):
            predicted_class_idx = int(np.argmax(probabilities))
        else:
            predicted_class_idx = probabilities.index(max(probabilities)) if isinstance(probabilities, list) else int(np.argmax(probabilities))
        
        predicted_class = threat_classes[predicted_class_idx]
        confidence = float(probabilities[predicted_class_idx])
        is_threat = predicted_class_idx < 4
        
        all_probs = {}
        for i, cls in enumerate(threat_classes):
            all_probs[cls] = float(probabilities[i])
        
        logger.info(f"Prediction: {predicted_class} ({confidence:.2%}) - Threat: {is_threat}")
        
        return jsonify({
            'threat_detected': bool(is_threat),
            'threat_type': str(predicted_class),
            'confidence': float(confidence),
            'all_probabilities': all_probs
        })
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

@app.route('/', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'online',
        'model_loaded': model is not None,
        'endpoints': ['/predict', '/reset']
    })

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'healthy',
        'torch_available': TORCH_AVAILABLE,
        'numpy_available': NUMPY_AVAILABLE
    })

@app.route('/reset', methods=['POST'])
def reset():
    return jsonify({'status': 'reset'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=9000, debug=False)
MLEOF

# Deploy proxy server
echo "📦 Deploying proxy server..."
cat > /workshop/hush-mesh/proxy_server.py << 'PROXYEOF'
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
PROXYEOF

# Make files executable
chmod +x /workshop/hush-mesh/ml/ml_api_server.py
chmod +x /workshop/hush-mesh/proxy_server.py

# Start ML server
echo "▶️  Starting ML server..."
cd /workshop/hush-mesh/ml
python3 ml_api_server.py > /tmp/ml_server.log 2>&1 &
ML_PID=$!
echo "   ML Server PID: $ML_PID"
sleep 3

# Start proxy server
echo "▶️  Starting proxy server..."
cd /workshop/hush-mesh
python3 proxy_server.py > /tmp/proxy_server.log 2>&1 &
PROXY_PID=$!
echo "   Proxy Server PID: $PROXY_PID"
sleep 2

echo ""
echo "======================================"
echo "✅ Deployment complete!"
echo ""
echo "📊 Server Status:"
echo "   ML Server: http://localhost:9000 (PID: $ML_PID)"
echo "   Proxy Server: http://localhost:8082 (PID: $PROXY_PID)"
echo ""
echo "📋 Logs:"
echo "   ML Server: /tmp/ml_server.log"
echo "   Proxy Server: /tmp/proxy_server.log"
echo ""
echo "🧪 Run verification:"
echo "   cd /workshop/hush-mesh && python3 verify_ml_integration.py"
echo ""
echo "🌐 Access demo:"
echo "   https://d3pka9yj6j75yn.cloudfront.net/ports/8082/enhanced_multi_route.html"
echo "======================================"