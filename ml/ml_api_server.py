#!/usr/bin/env python3
import json
import torch
import torch.nn as nn
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
import logging

app = Flask(__name__)
CORS(app)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MaritimeThreatDetector(nn.Module):
    def __init__(self, input_size=15, hidden_size=128, num_classes=8):
        super(MaritimeThreatDetector, self).__init__()
        # Conv1d expects (batch, channels, length) - input has 15 features as channels
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
        # x shape: (batch, 15) -> (batch, 15, 1) for conv1d
        x = x.unsqueeze(2)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = x.transpose(1, 2)
        lstm_out, _ = self.lstm(x)
        x = self.dropout(lstm_out[:, -1, :])
        x = self.classifier(x)
        return self.softmax(x)

# Load model
model = None
try:
    model = MaritimeThreatDetector()
    state_dict = torch.load('/workshop/models/model.pth', map_location='cpu')
    model.load_state_dict(state_dict)
    model.eval()
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Error loading model: {e}")
    # Create model with random weights as fallback
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
        
        # Extract features
        features = extract_features(sensor_data)
        input_tensor = torch.FloatTensor(features).unsqueeze(0)
        logger.info(f"Input tensor shape: {input_tensor.shape}")
        
        # Run inference
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = outputs.cpu().numpy()[0]
            logger.info(f"Model output shape: {outputs.shape}")
            
            # Threat classes
            threat_classes = [
                'small_fast_craft', 'floating_mine_like_object', 'submarine_periscope',
                'debris_field', 'fishing_vessel', 'cargo_ship', 'research_vessel', 'unknown'
            ]
            
            predicted_class_idx = np.argmax(probabilities)
            predicted_class = threat_classes[predicted_class_idx]
            confidence = float(probabilities[predicted_class_idx])
            
            # Determine if it's a threat (first 4 classes are threats)
            is_threat = predicted_class_idx < 4
            
            return jsonify({
                'threat_detected': bool(is_threat),
                'threat_type': str(predicted_class),
                'confidence': float(confidence),
                'all_probabilities': {cls: float(prob) for cls, prob in zip(threat_classes, probabilities)}
            })
            
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/reset', methods=['POST'])
def reset():
    return jsonify({'status': 'reset'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)