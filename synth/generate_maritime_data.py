#!/usr/bin/env python3
import json
import random
import argparse
from datetime import datetime, timedelta
import boto3
import numpy as np

class MaritimeDataGenerator:
    def __init__(self):
        self.threat_types = [
            'small_fast_craft', 'suspicious_loitering_vessel', 'unregistered_vessel',
            'ais_spoofing', 'drone_overwater', 'diver_or_swimmer',
            'floating_mine_like_object', 'collision_risk', 'acoustic_gunshot',
            'submarine_periscope', 'debris_field', 'fishing_nets', 'oil_spill',
            'weather_front', 'shallow_water', 'restricted_zone'
        ]
        self.weather_conditions = ['clear', 'overcast', 'rain', 'fog', 'storm']
        self.sea_states = [1, 2, 3, 4, 5, 6]  # Beaufort scale
        
    def generate_scenario(self, scenario_type, duration_minutes=60):
        """Generate synthetic maritime scenario data"""
        frames = []
        start_time = datetime.utcnow()
        
        for minute in range(duration_minutes):
            timestamp = start_time + timedelta(minutes=minute)
            frame = self._generate_frame(scenario_type, timestamp, minute)
            frames.append(frame)
            
        return frames
    
    def _generate_frame(self, scenario_type, timestamp, minute):
        """Generate single data frame"""
        # Base convoy position (moving east)
        convoy_lat = 25.7617 + (minute * 0.001)  # Miami area
        convoy_lon = -80.1918 + (minute * 0.002)
        
        # Generate detections based on scenario
        detections = []
        ground_truth_threat = False
        threat_type = None
        
        if scenario_type == 'piracy_ambush' and minute > 30:
            # Small fast craft approaching
            detections.append({
                'detection_id': f'det_{minute}_001',
                'object_type': 'vessel',
                'confidence': 0.85 + random.uniform(-0.1, 0.1),
                'bbox': [100, 150, 200, 250],
                'distance_m': 2000 - (minute * 50),
                'bearing_deg': 45 + random.uniform(-10, 10),
                'speed_knots': 25 + random.uniform(-5, 5)
            })
            ground_truth_threat = True
            threat_type = 'small_fast_craft'
            
        elif scenario_type == 'swarm_interdiction' and minute > 20:
            # Multiple small contacts
            for i in range(3):
                detections.append({
                    'detection_id': f'det_{minute}_{i:03d}',
                    'object_type': 'vessel',
                    'confidence': 0.7 + random.uniform(-0.2, 0.2),
                    'bbox': [50 + i*100, 100, 150 + i*100, 200],
                    'distance_m': 1500 + i*200,
                    'bearing_deg': 30 + i*15,
                    'speed_knots': 20 + random.uniform(-3, 3)
                })
            ground_truth_threat = True
            threat_type = 'small_fast_craft'
        
        return {
            'timestamp': timestamp.isoformat(),
            'scenario_id': scenario_type,
            'vehicle_id': 'convoy_01',
            'drone_id': 'drone_01',
            'drone_pose': {
                'lat': convoy_lat + 0.001,
                'lon': convoy_lon + 0.001,
                'altitude_m': 100,
                'heading_deg': 90
            },
            'convoy_pose': {
                'lat': convoy_lat,
                'lon': convoy_lon,
                'heading_deg': 90,
                'speed_knots': 12
            },
            'weather': {
                'sea_state': random.randint(1, 4),
                'visibility_km': 10 + random.uniform(-3, 5),
                'wind_speed_knots': 15 + random.uniform(-5, 10)
            },
            'sensor_data': {
                'camera_frame_id': f'frame_{minute:04d}',
                'radar_contacts': len(detections),
                'acoustic_level_db': 45 + random.uniform(-10, 20),
                'rf_signals': random.randint(0, 3)
            },
            'detections': detections,
            'ground_truth_threat': ground_truth_threat,
            'threat_type': threat_type,
            'threat_severity': 'high' if ground_truth_threat else 'none',
            'evidence_reference': f's3://evidence/{scenario_type}/{minute:04d}.jpg'
        }
    
    def save_to_s3(self, frames, bucket_name, scenario_type):
        """Save generated data to S3"""
        s3 = boto3.client('s3')
        
        # Save as NDJSON
        ndjson_content = '\n'.join([json.dumps(frame) for frame in frames])
        key = f'raw/{scenario_type}/data.jsonl'
        
        s3.put_object(
            Bucket=bucket_name,
            Key=key,
            Body=ndjson_content,
            ContentType='application/x-ndjson'
        )
        
        print(f"Saved {len(frames)} frames to s3://{bucket_name}/{key}")

def main():
    parser = argparse.ArgumentParser(description='Generate maritime synthetic data')
    parser.add_argument('--scenarios', default='piracy_ambush,swarm_interdiction')
    parser.add_argument('--output-bucket', required=True)
    parser.add_argument('--duration', type=int, default=60)
    
    args = parser.parse_args()
    
    generator = MaritimeDataGenerator()
    scenarios = args.scenarios.split(',')
    
    for scenario in scenarios:
        print(f"Generating scenario: {scenario}")
        frames = generator.generate_scenario(scenario.strip(), args.duration)
        generator.save_to_s3(frames, args.output_bucket, scenario.strip())

if __name__ == '__main__':
    main()