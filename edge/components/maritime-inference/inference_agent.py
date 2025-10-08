#!/usr/bin/env python3
import json
import time
import torch
import numpy as np
import paho.mqtt.client as mqtt
import threading
from datetime import datetime
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MaritimeInferenceAgent:
    def __init__(self, config_path='/greengrass/v2/config/config.yaml'):
        self.model = None
        self.mqtt_client = None
        self.running = False
        self.config = self._load_config()
        self.sensor_buffer = []
        
    def _load_config(self):
        """Load configuration from Greengrass"""
        return {
            'model_path': os.environ.get('MODEL_PATH', '/tmp/model.pt'),
            'inference_interval': int(os.environ.get('INFERENCE_INTERVAL', '5')),
            'confidence_threshold': float(os.environ.get('CONFIDENCE_THRESHOLD', '0.7')),
            'iot_topic': os.environ.get('IOT_TOPIC', 'maritime/detections'),
            'device_name': os.environ.get('AWS_IOT_THING_NAME', 'convoy-edge-01')
        }
    
    def load_model(self):
        """Load TorchScript model for inference"""
        try:
            self.model = torch.jit.load(self.config['model_path'])
            self.model.eval()
            logger.info(f"Model loaded from {self.config['model_path']}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            # Fallback to heuristic rules
            self.model = None
    
    def setup_mqtt(self):
        """Setup MQTT connection to IoT Core"""
        self.mqtt_client = mqtt.Client()
        
        # Load certificates
        cert_path = '/greengrass/v2/work/aws.greengrass.clientdevices.mqtt.Moquette/certs'
        
        try:
            self.mqtt_client.tls_set(
                ca_certs=f'{cert_path}/AmazonRootCA1.pem',
                certfile=f'{cert_path}/device.pem.crt',
                keyfile=f'{cert_path}/private.pem.key'
            )
            
            # Connect to local Greengrass MQTT broker
            self.mqtt_client.connect('localhost', 8883, 60)
            self.mqtt_client.loop_start()
            logger.info("Connected to MQTT broker")
            
        except Exception as e:
            logger.error(f"MQTT connection failed: {e}")
    
    def simulate_sensor_data(self):
        """Simulate incoming sensor data"""
        while self.running:
            # Simulate maritime sensor frame
            sensor_frame = {
                'timestamp': datetime.utcnow().isoformat(),
                'device_id': self.config['device_name'],
                'detections': self._generate_mock_detections(),
                'weather': {
                    'sea_state': np.random.randint(1, 5),
                    'visibility_km': 10 + np.random.uniform(-3, 5),
                    'wind_speed_knots': 15 + np.random.uniform(-5, 10)
                },
                'sensor_data': {
                    'radar_contacts': np.random.randint(0, 5),
                    'acoustic_level_db': 45 + np.random.uniform(-10, 20),
                    'rf_signals': np.random.randint(0, 3)
                }
            }
            
            self.sensor_buffer.append(sensor_frame)
            
            # Keep only last 60 frames (1 minute at 1Hz)
            if len(self.sensor_buffer) > 60:
                self.sensor_buffer.pop(0)
            
            time.sleep(1)  # 1Hz sensor data
    
    def _generate_mock_detections(self):
        """Generate mock detections for simulation"""
        detections = []
        
        # Random chance of detection
        if np.random.random() < 0.3:
            detection = {
                'detection_id': f'det_{int(time.time())}',
                'object_type': 'vessel',
                'confidence': np.random.uniform(0.5, 0.95),
                'distance_m': np.random.uniform(500, 5000),
                'bearing_deg': np.random.uniform(0, 360),
                'speed_knots': np.random.uniform(5, 30)
            }
            detections.append(detection)
        
        return detections
    
    def run_inference(self):
        """Run inference on sensor buffer"""
        if len(self.sensor_buffer) < 60:
            return None
        
        try:
            # Extract features from buffer
            features = self._extract_features(self.sensor_buffer[-60:])
            
            if self.model is not None:
                # ML inference
                with torch.no_grad():
                    input_tensor = torch.FloatTensor(features).unsqueeze(0)
                    output = self.model(input_tensor)
                    probabilities = torch.softmax(output, dim=1)
                    
                    max_prob, predicted_class = torch.max(probabilities, 1)
                    
                    threat_types = [
                        'none', 'small_fast_craft', 'suspicious_loitering_vessel',
                        'unregistered_vessel', 'ais_spoofing', 'drone_overwater',
                        'diver_or_swimmer', 'floating_mine_like_object', 'collision_risk'
                    ]
                    
                    result = {
                        'threat_detected': max_prob.item() > self.config['confidence_threshold'],
                        'threat_type': threat_types[predicted_class.item()],
                        'confidence': max_prob.item(),
                        'timestamp': datetime.utcnow().isoformat()
                    }
            else:
                # Fallback heuristic rules
                result = self._heuristic_detection(self.sensor_buffer[-1])
            
            return result
            
        except Exception as e:
            logger.error(f"Inference failed: {e}")
            return None
    
    def _extract_features(self, frames):
        """Extract feature vector from sensor frames"""
        features = []
        
        for frame in frames:
            feature_vector = [
                len(frame.get('detections', [])),
                frame.get('weather', {}).get('sea_state', 0),
                frame.get('weather', {}).get('visibility_km', 10) / 20.0,
                frame.get('weather', {}).get('wind_speed_knots', 0) / 50.0,
                frame.get('sensor_data', {}).get('radar_contacts', 0) / 10.0,
                frame.get('sensor_data', {}).get('acoustic_level_db', 50) / 100.0,
                frame.get('sensor_data', {}).get('rf_signals', 0) / 10.0,
                0.0,  # placeholder for convoy speed
                0.0,  # placeholder for drone altitude
                1.0   # placeholder for tether status
            ]
            features.append(feature_vector)
        
        return features
    
    def _heuristic_detection(self, frame):
        """Fallback heuristic threat detection"""
        detections = frame.get('detections', [])
        
        # Simple rules
        threat_detected = False
        threat_type = 'none'
        confidence = 0.0
        
        for detection in detections:
            # Fast approaching vessel
            if (detection.get('speed_knots', 0) > 20 and 
                detection.get('distance_m', 9999) < 1000):
                threat_detected = True
                threat_type = 'small_fast_craft'
                confidence = 0.8
                break
        
        # High acoustic signature
        if frame.get('sensor_data', {}).get('acoustic_level_db', 0) > 70:
            threat_detected = True
            threat_type = 'acoustic_gunshot'
            confidence = 0.6
        
        return {
            'threat_detected': threat_detected,
            'threat_type': threat_type,
            'confidence': confidence,
            'timestamp': datetime.utcnow().isoformat(),
            'method': 'heuristic'
        }
    
    def publish_detection(self, detection_result):
        """Publish detection to IoT Core"""
        if detection_result and self.mqtt_client:
            message = {
                'device_id': self.config['device_name'],
                'detection': detection_result,
                'location': {
                    'lat': 25.7617,  # Mock coordinates
                    'lon': -80.1918
                }
            }
            
            try:
                self.mqtt_client.publish(
                    self.config['iot_topic'],
                    json.dumps(message),
                    qos=1
                )
                logger.info(f"Published detection: {detection_result['threat_type']}")
            except Exception as e:
                logger.error(f"Failed to publish: {e}")
    
    def run(self):
        """Main execution loop"""
        logger.info("Starting Maritime Inference Agent")
        
        self.load_model()
        self.setup_mqtt()
        
        self.running = True
        
        # Start sensor simulation thread
        sensor_thread = threading.Thread(target=self.simulate_sensor_data)
        sensor_thread.daemon = True
        sensor_thread.start()
        
        # Main inference loop
        while self.running:
            try:
                result = self.run_inference()
                if result and result['threat_detected']:
                    self.publish_detection(result)
                
                time.sleep(self.config['inference_interval'])
                
            except KeyboardInterrupt:
                logger.info("Shutting down...")
                self.running = False
                break
            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                time.sleep(5)

if __name__ == '__main__':
    agent = MaritimeInferenceAgent()
    agent.run()