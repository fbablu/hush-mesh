import json
import torch
import torch.nn as nn
import numpy as np
import logging

logger = logging.getLogger(__name__)

class MaritimeThreatDetector(nn.Module):
    def __init__(self, input_size=15, hidden_size=64, num_classes=8):
        super(MaritimeThreatDetector, self).__init__()
        self.conv1d = nn.Conv1d(1, 32, kernel_size=3, padding=1)
        self.lstm = nn.LSTM(32, hidden_size, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.3)
        self.fc1 = nn.Linear(hidden_size * 2, 32)
        self.fc2 = nn.Linear(32, num_classes)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.relu(self.conv1d(x))
        x = x.transpose(1, 2)
        lstm_out, _ = self.lstm(x)
        x = self.dropout(lstm_out[:, -1, :])
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return self.softmax(x)

def model_fn(model_dir):
    """Load the PyTorch model from the model_dir"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MaritimeThreatDetector()
    
    try:
        model_path = f"{model_dir}/model.pth"
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        logger.info("Model loaded successfully")
        return model
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise

def input_fn(request_body, request_content_type):
    """Parse input data for inference"""
    if request_content_type == 'application/json':
        input_data = json.loads(request_body)
        sensor_data = input_data.get('sensor_data', {})
        
        # Extract features (same as training)
        features = extract_features(sensor_data)
        return torch.FloatTensor(features).unsqueeze(0)
    else:
        raise ValueError(f"Unsupported content type: {request_content_type}")

def predict_fn(input_data, model):
    """Run inference on the input data"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_data = input_data.to(device)
    
    with torch.no_grad():
        outputs = model(input_data)
        probabilities = outputs.cpu().numpy()[0]
        
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
        
        return {
            'threat_detected': is_threat,
            'threat_type': predicted_class,
            'confidence': confidence,
            'all_probabilities': {cls: float(prob) for cls, prob in zip(threat_classes, probabilities)}
        }

def output_fn(prediction, content_type):
    """Format the prediction output"""
    if content_type == 'application/json':
        return json.dumps(prediction)
    else:
        raise ValueError(f"Unsupported content type: {content_type}")

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
        # Return default features
        return [12.0, 4, 2, 15.0, 10.0, 2.0, 0.9, 180.0, 1, 1, 1, 95.0, 3, 0, 0.0]