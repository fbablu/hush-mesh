#!/usr/bin/env python3
import json
import time
import random
import argparse
import boto3
from datetime import datetime, timedelta
import threading
import math

class MaritimeSimulator:
    def __init__(self, iot_endpoint, device_name):
        self.iot_client = boto3.client('iot-data')
        self.device_name = device_name
        self.running = False
        
        # Simulation state
        self.convoy_position = {'lat': 25.7617, 'lon': -80.1918, 'heading': 90}
        self.convoy_speed = 12  # knots
        self.threats = []
        
    def start_scenario(self, scenario_type, duration_minutes=60):
        """Start maritime scenario simulation"""
        print(f"Starting scenario: {scenario_type}")
        
        self.running = True
        start_time = datetime.utcnow()
        
        # Initialize scenario-specific threats
        if scenario_type == 'piracy_ambush':
            self._setup_piracy_scenario()
        elif scenario_type == 'swarm_interdiction':
            self._setup_swarm_scenario()
        elif scenario_type == 'mine_detection':
            self._setup_mine_scenario()
        
        # Main simulation loop
        frame_count = 0
        while self.running and frame_count < duration_minutes * 60:  # 1Hz
            current_time = start_time + timedelta(seconds=frame_count)
            
            # Update convoy position
            self._update_convoy_position()
            
            # Update threats
            self._update_threats(frame_count, scenario_type)
            
            # Generate sensor frame
            sensor_frame = self._generate_sensor_frame(current_time, scenario_type)
            
            # Publish to IoT Core
            self._publish_telemetry(sensor_frame)
            
            frame_count += 1
            time.sleep(1)  # 1Hz simulation
        
        print(f"Scenario {scenario_type} completed")
    
    def _setup_piracy_scenario(self):
        """Setup piracy ambush scenario"""
        # Small fast craft will appear after 30 seconds
        self.threats = [{
            'type': 'small_fast_craft',
            'position': {'lat': 25.7717, 'lon': -80.1818},
            'speed': 25,
            'heading': 225,  # Intercept course
            'active_time': 30,
            'detection_probability': 0.8
        }]
    
    def _setup_swarm_scenario(self):
        """Setup swarm interdiction scenario"""
        # Multiple small craft
        self.threats = []
        for i in range(3):
            self.threats.append({
                'type': 'small_fast_craft',
                'position': {
                    'lat': 25.7617 + 0.01 + i * 0.005,
                    'lon': -80.1818 + i * 0.01
                },
                'speed': 20 + random.uniform(-3, 5),
                'heading': 180 + i * 30,
                'active_time': 20 + i * 10,
                'detection_probability': 0.7
            })
    
    def _setup_mine_scenario(self):
        """Setup mine detection scenario"""
        # Floating mines
        self.threats = [{
            'type': 'floating_mine_like_object',
            'position': {'lat': 25.7667, 'lon': -80.1868},
            'speed': 0,
            'heading': 0,
            'active_time': 0,  # Always active
            'detection_probability': 0.6
        }]
    
    def _update_convoy_position(self):
        """Update convoy position based on speed and heading"""
        # Convert speed from knots to degrees per second (rough approximation)
        speed_deg_per_sec = (self.convoy_speed * 0.000514444) / 3600
        
        # Update position
        heading_rad = math.radians(self.convoy_position['heading'])
        self.convoy_position['lat'] += speed_deg_per_sec * math.cos(heading_rad)
        self.convoy_position['lon'] += speed_deg_per_sec * math.sin(heading_rad)
    
    def _update_threats(self, frame_count, scenario_type):
        """Update threat positions and states"""
        for threat in self.threats:
            if frame_count >= threat['active_time']:
                # Update threat position
                if threat['speed'] > 0:
                    speed_deg_per_sec = (threat['speed'] * 0.000514444) / 3600
                    heading_rad = math.radians(threat['heading'])
                    
                    threat['position']['lat'] += speed_deg_per_sec * math.cos(heading_rad)
                    threat['position']['lon'] += speed_deg_per_sec * math.sin(heading_rad)
                
                # Update heading for intercept course (piracy scenario)
                if threat['type'] == 'small_fast_craft' and scenario_type == 'piracy_ambush':
                    # Calculate intercept heading
                    dlat = self.convoy_position['lat'] - threat['position']['lat']
                    dlon = self.convoy_position['lon'] - threat['position']['lon']
                    threat['heading'] = math.degrees(math.atan2(dlon, dlat))
    
    def _generate_sensor_frame(self, timestamp, scenario_type):
        """Generate sensor data frame"""
        detections = []
        
        # Check for threat detections
        for threat in self.threats:
            if random.random() < threat['detection_probability']:
                # Calculate distance to convoy
                dlat = threat['position']['lat'] - self.convoy_position['lat']
                dlon = threat['position']['lon'] - self.convoy_position['lon']
                distance_m = math.sqrt(dlat*dlat + dlon*dlon) * 111000  # Rough conversion
                
                detection = {
                    'detection_id': f"det_{int(timestamp.timestamp())}_{threat['type']}",
                    'object_type': 'vessel' if 'craft' in threat['type'] else 'object',
                    'confidence': threat['detection_probability'] + random.uniform(-0.1, 0.1),
                    'bbox': [100, 150, 200, 250],  # Mock bounding box
                    'distance_m': distance_m,
                    'bearing_deg': math.degrees(math.atan2(dlon, dlat)),
                    'speed_knots': threat['speed']
                }
                detections.append(detection)
        
        # Determine ground truth
        ground_truth_threat = len(detections) > 0
        threat_type = detections[0]['object_type'] if detections else None
        
        # Map to threat categories
        if detections:
            if 'craft' in str(detections[0]):
                threat_type = 'small_fast_craft'
            elif 'mine' in str(detections[0]):
                threat_type = 'floating_mine_like_object'
        
        return {
            'timestamp': timestamp.isoformat(),
            'scenario_id': scenario_type,
            'vehicle_id': 'convoy_01',
            'drone_id': 'drone_01',
            'drone_pose': {
                'lat': self.convoy_position['lat'] + 0.001,
                'lon': self.convoy_position['lon'] + 0.001,
                'altitude_m': 100,
                'heading_deg': self.convoy_position['heading']
            },
            'convoy_pose': self.convoy_position.copy(),
            'weather': {
                'sea_state': random.randint(2, 4),
                'visibility_km': 10 + random.uniform(-2, 5),
                'wind_speed_knots': 15 + random.uniform(-5, 10)
            },
            'sensor_data': {
                'camera_frame_id': f"frame_{int(timestamp.timestamp())}",
                'radar_contacts': len(detections),
                'acoustic_level_db': 45 + random.uniform(-10, 20),
                'rf_signals': random.randint(0, 2)
            },
            'detections': detections,
            'ground_truth_threat': ground_truth_threat,
            'threat_type': threat_type,
            'threat_severity': 'high' if ground_truth_threat else 'none'
        }
    
    def _publish_telemetry(self, sensor_frame):
        """Publish sensor frame to IoT Core"""
        try:
            # Format for edge inference agent
            message = {
                'device_id': self.device_name,
                'sensor_frame': sensor_frame,
                'timestamp': datetime.utcnow().isoformat()
            }
            
            self.iot_client.publish(
                topic='maritime/sensor_data',
                qos=1,
                payload=json.dumps(message)
            )
            
            # Also simulate detection if threats present
            if sensor_frame['ground_truth_threat']:
                detection_message = {
                    'device_id': self.device_name,
                    'detection': {
                        'threat_detected': True,
                        'threat_type': sensor_frame['threat_type'],
                        'confidence': 0.85,
                        'timestamp': sensor_frame['timestamp']
                    },
                    'location': {
                        'lat': self.convoy_position['lat'],
                        'lon': self.convoy_position['lon']
                    }
                }
                
                self.iot_client.publish(
                    topic='maritime/detections',
                    qos=1,
                    payload=json.dumps(detection_message)
                )
                
                print(f"🚨 Threat detected: {sensor_frame['threat_type']}")
        
        except Exception as e:
            print(f"Failed to publish telemetry: {e}")
    
    def stop(self):
        """Stop simulation"""
        self.running = False

def main():
    parser = argparse.ArgumentParser(description='Maritime scenario simulator')
    parser.add_argument('--scenario', required=True, 
                       choices=['piracy_ambush', 'swarm_interdiction', 'mine_detection'])
    parser.add_argument('--duration', type=int, default=300, help='Duration in seconds')
    parser.add_argument('--device-name', default='convoy-edge-01')
    parser.add_argument('--iot-endpoint', default='localhost')
    
    args = parser.parse_args()
    
    simulator = MaritimeSimulator(args.iot_endpoint, args.device_name)
    
    try:
        simulator.start_scenario(args.scenario, args.duration // 60)
    except KeyboardInterrupt:
        print("\nStopping simulation...")
        simulator.stop()

if __name__ == '__main__':
    main()