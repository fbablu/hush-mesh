#!/usr/bin/env python3
import json
import random
import argparse
from datetime import datetime, timedelta
import boto3
import numpy as np
import math

class EnhancedMaritimeDataGenerator:
    def __init__(self):
        self.threat_types = [
            'small_fast_craft', 'suspicious_loitering_vessel', 'unregistered_vessel',
            'ais_spoofing', 'drone_overwater', 'diver_or_swimmer',
            'floating_mine_like_object', 'collision_risk', 'acoustic_gunshot',
            'submarine_periscope', 'debris_field', 'fishing_nets', 'oil_spill',
            'weather_front', 'shallow_water', 'restricted_zone'
        ]
        self.weather_conditions = ['clear', 'overcast', 'rain', 'fog', 'storm']
        self.sea_states = [1, 2, 3, 4, 5, 6]
        
    def generate_scenario(self, scenario_type, duration_minutes=60):
        """Generate enhanced maritime scenario with realistic drone sensor data"""
        frames = []
        start_time = datetime.utcnow()
        
        for minute in range(duration_minutes):
            timestamp = start_time + timedelta(minutes=minute)
            frame = self._generate_enhanced_frame(scenario_type, timestamp, minute)
            frames.append(frame)
            
        return frames
    
    def _generate_enhanced_frame(self, scenario_type, timestamp, minute):
        """Generate frame with realistic multi-drone sensor data"""
        # Convoy position with natural drift
        convoy_lat = 25.7617 + (minute * 0.001) + random.uniform(-0.0003, 0.0003)
        convoy_lon = -80.1918 + (minute * 0.002) + random.uniform(-0.0003, 0.0003)
        convoy_speed = 12 + random.uniform(-1, 2)
        convoy_heading = 90 + random.uniform(-5, 5)
        
        # Multi-drone formation
        drone_array = []
        for i in range(4):
            angle = i * 90 + random.uniform(-10, 10)
            radius = 0.003 + random.uniform(-0.001, 0.001)
            drone_lat = convoy_lat + radius * math.cos(math.radians(angle))
            drone_lon = convoy_lon + radius * math.sin(math.radians(angle))
            
            drone_array.append({
                'drone_id': f'tethered_drone_{i+1:02d}',
                'position': {
                    'lat': drone_lat,
                    'lon': drone_lon,
                    'altitude_m': 180 + random.uniform(-30, 50),
                    'heading_deg': random.uniform(0, 360)
                },
                'tether_status': {
                    'length_m': random.uniform(200, 250),
                    'tension_n': random.uniform(50, 150),
                    'integrity': random.uniform(0.85, 1.0)
                },
                'power_status': {
                    'battery_pct': random.uniform(70, 95),
                    'power_consumption_w': random.uniform(180, 220),
                    'tether_power': True
                },
                'sensor_suite': self._generate_drone_sensors(minute, scenario_type)
            })
        
        # Environmental conditions
        weather = self._generate_weather_conditions()
        
        # Threat detection logic
        detections = []
        ground_truth_threat = False
        threat_type = None
        
        # Scenario-specific threat generation
        if scenario_type == 'piracy_ambush' and minute > 20:
            if random.random() < 0.75:
                detection = self._generate_piracy_threat(minute, convoy_lat, convoy_lon)
                detections.append(detection)
                ground_truth_threat = True
                threat_type = 'small_fast_craft'
                
        elif scenario_type == 'swarm_interdiction' and minute > 15:
            swarm_size = random.randint(3, 6)
            for i in range(swarm_size):
                if random.random() < 0.65:
                    detection = self._generate_swarm_threat(minute, i, convoy_lat, convoy_lon)
                    detections.append(detection)
            if detections:
                ground_truth_threat = True
                threat_type = 'small_fast_craft'
                
        elif scenario_type == 'mine_field':
            if random.random() < 0.25:
                detection = self._generate_mine_threat(minute, convoy_lat, convoy_lon)
                detections.append(detection)
                ground_truth_threat = True
                threat_type = 'floating_mine_like_object'
                
        elif scenario_type == 'submarine_contact' and minute > 25:
            if random.random() < 0.3:
                detection = self._generate_submarine_threat(minute, convoy_lat, convoy_lon)
                detections.append(detection)
                ground_truth_threat = True
                threat_type = 'submarine_periscope'
        
        # Add environmental hazards
        if random.random() < 0.1:
            env_detection = self._generate_environmental_hazard(minute, convoy_lat, convoy_lon)
            detections.append(env_detection)
            if env_detection['threat_level'] > 0.6:
                ground_truth_threat = True
                threat_type = env_detection['hazard_type']
        
        # False positives for realism
        if random.random() < 0.12:
            false_positive = self._generate_false_positive(minute, convoy_lat, convoy_lon)
            detections.append(false_positive)
        
        return {
            'timestamp': timestamp.isoformat(),
            'scenario_id': scenario_type,
            'mission_id': f'convoy_mission_{random.randint(1000, 9999)}',
            'convoy_data': {
                'position': {'lat': convoy_lat, 'lon': convoy_lon},
                'heading_deg': convoy_heading,
                'speed_knots': convoy_speed,
                'formation': 'diamond',
                'vessel_count': random.randint(3, 8)
            },
            'drone_array': drone_array,
            'environmental_conditions': weather,
            'navigation_data': {
                'planned_route': self._generate_route_waypoints(convoy_lat, convoy_lon),
                'current_waypoint': random.randint(1, 10),
                'eta_minutes': random.randint(45, 180),
                'fuel_remaining_pct': random.uniform(60, 90)
            },
            'threat_detections': detections,
            'ground_truth': {
                'threat_present': ground_truth_threat,
                'threat_type': threat_type,
                'threat_severity': self._calculate_threat_severity(detections),
                'recommended_action': self._get_recommended_action(ground_truth_threat, threat_type)
            },
            'data_quality': {
                'sensor_reliability': random.uniform(0.85, 0.98),
                'weather_impact': weather['visibility_impact'],
                'communication_strength': random.uniform(0.7, 1.0)
            }
        }
    
    def _generate_drone_sensors(self, minute, scenario_type):
        """Generate realistic drone sensor suite data"""
        return {
            'electro_optical': {
                'visible_spectrum': {
                    'resolution': '4K',
                    'zoom_level': random.uniform(1.0, 12.0),
                    'stabilization': random.uniform(0.9, 1.0),
                    'objects_detected': random.randint(0, 15)
                },
                'infrared': {
                    'temperature_range_c': [random.uniform(-5, 35), random.uniform(35, 60)],
                    'thermal_signatures': random.randint(0, 8),
                    'heat_anomalies': random.randint(0, 3)
                }
            },
            'radar': {
                'range_km': random.uniform(8, 15),
                'contacts': random.randint(0, 12),
                'sea_clutter_level': random.uniform(0.1, 0.4),
                'doppler_data': [random.uniform(-25, 25) for _ in range(random.randint(0, 5))]
            },
            'acoustic': {
                'hydrophone_data': {
                    'ambient_noise_db': random.uniform(85, 110),
                    'frequency_peaks_hz': [random.uniform(50, 2000) for _ in range(random.randint(1, 5))],
                    'cavitation_detected': random.random() < 0.15,
                    'propeller_signatures': random.randint(0, 4)
                },
                'sonar': {
                    'active_pings': random.randint(0, 3),
                    'echo_returns': random.randint(0, 8),
                    'bottom_depth_m': random.uniform(15, 200)
                }
            },
            'electronic_warfare': {
                'rf_spectrum': {
                    'signals_detected': random.randint(0, 20),
                    'frequency_bands': ['VHF', 'UHF', 'L', 'S', 'C'],
                    'signal_strength_dbm': [random.uniform(-80, -20) for _ in range(random.randint(1, 8))],
                    'jamming_detected': random.random() < 0.05
                },
                'ais_data': {
                    'vessels_tracked': random.randint(0, 15),
                    'spoofing_indicators': random.random() < 0.08,
                    'missing_transponders': random.randint(0, 3)
                }
            },
            'environmental': {
                'wind_speed_ms': random.uniform(2, 15),
                'wind_direction_deg': random.uniform(0, 360),
                'wave_height_m': random.uniform(0.5, 4.0),
                'water_temperature_c': random.uniform(18, 28),
                'salinity_ppt': random.uniform(34, 37)
            }
        }
    
    def _generate_weather_conditions(self):
        """Generate realistic weather data"""
        condition = random.choice(self.weather_conditions)
        sea_state = random.choice(self.sea_states)
        
        visibility_km = {
            'clear': random.uniform(15, 25),
            'overcast': random.uniform(10, 18),
            'rain': random.uniform(3, 12),
            'fog': random.uniform(0.5, 5),
            'storm': random.uniform(1, 8)
        }[condition]
        
        return {
            'condition': condition,
            'sea_state': sea_state,
            'visibility_km': visibility_km,
            'visibility_impact': 1.0 - (visibility_km / 25.0),
            'wind_speed_knots': random.uniform(5, 25),
            'wave_height_m': sea_state * random.uniform(0.8, 1.2),
            'precipitation_mm': random.uniform(0, 15) if condition in ['rain', 'storm'] else 0,
            'cloud_cover_pct': random.uniform(10, 95) if condition != 'clear' else random.uniform(0, 20)
        }
    
    def _generate_piracy_threat(self, minute, convoy_lat, convoy_lon):
        """Generate piracy threat detection"""
        distance = random.uniform(800, 3000)
        bearing = random.uniform(0, 360)
        
        return {
            'detection_id': f'piracy_{minute}_{random.randint(100, 999)}',
            'object_type': 'small_fast_craft',
            'confidence': random.uniform(0.75, 0.95),
            'position': {
                'distance_m': distance,
                'bearing_deg': bearing,
                'estimated_lat': convoy_lat + (distance/111000) * math.cos(math.radians(bearing)),
                'estimated_lon': convoy_lon + (distance/111000) * math.sin(math.radians(bearing))
            },
            'kinematics': {
                'speed_knots': random.uniform(20, 35),
                'heading_deg': random.uniform(0, 360),
                'acceleration_ms2': random.uniform(-2, 3)
            },
            'characteristics': {
                'length_m': random.uniform(8, 15),
                'beam_m': random.uniform(2, 4),
                'crew_estimated': random.randint(2, 8),
                'weapons_visible': random.random() < 0.6
            },
            'threat_indicators': {
                'intercept_course': random.random() < 0.8,
                'high_speed_approach': True,
                'erratic_movement': random.random() < 0.4,
                'no_ais_transponder': True
            }
        }
    
    def _generate_route_waypoints(self, current_lat, current_lon):
        """Generate navigation waypoints for path planning"""
        waypoints = []
        for i in range(5):
            waypoints.append({
                'waypoint_id': f'WP_{i+1:02d}',
                'lat': current_lat + (i * 0.01) + random.uniform(-0.005, 0.005),
                'lon': current_lon + (i * 0.015) + random.uniform(-0.005, 0.005),
                'eta_minutes': (i + 1) * 30 + random.randint(-10, 10)
            })
        return waypoints
    
    def _calculate_threat_severity(self, detections):
        """Calculate overall threat severity"""
        if not detections:
            return 'none'
        
        max_threat = max([d.get('threat_indicators', {}).get('intercept_course', False) for d in detections])
        threat_count = len(detections)
        
        if threat_count >= 3 or max_threat:
            return 'high'
        elif threat_count >= 2:
            return 'medium'
        else:
            return 'low'
    
    def _get_recommended_action(self, threat_present, threat_type):
        """Get recommended evasive action"""
        if not threat_present:
            return 'continue_course'
        
        actions = {
            'small_fast_craft': 'evasive_maneuver_alpha',
            'floating_mine_like_object': 'route_deviation_beta',
            'submarine_periscope': 'emergency_surface_protocol',
            'debris_field': 'slow_navigation_gamma'
        }
        
        return actions.get(threat_type, 'general_evasion')
    
    def _generate_swarm_threat(self, minute, index, convoy_lat, convoy_lon):
        """Generate swarm threat detection"""
        base_distance = random.uniform(1200, 2500)
        base_bearing = random.uniform(0, 360)
        
        # Spread swarm members around base position
        distance = base_distance + random.uniform(-300, 300)
        bearing = base_bearing + random.uniform(-45, 45)
        
        return {
            'detection_id': f'swarm_{minute}_{index}_{random.randint(100, 999)}',
            'object_type': 'small_fast_craft',
            'confidence': random.uniform(0.65, 0.85),
            'swarm_member': True,
            'swarm_id': f'swarm_alpha_{minute}',
            'position': {
                'distance_m': distance,
                'bearing_deg': bearing
            },
            'kinematics': {
                'speed_knots': random.uniform(18, 28),
                'heading_deg': random.uniform(0, 360)
            },
            'threat_indicators': {
                'coordinated_movement': True,
                'formation_pattern': random.choice(['line', 'wedge', 'diamond']),
                'communication_intercepts': random.random() < 0.3
            }
        }
    
    def _generate_mine_threat(self, minute, convoy_lat, convoy_lon):
        """Generate mine threat detection"""
        return {
            'detection_id': f'mine_{minute}_{random.randint(100, 999)}',
            'object_type': 'floating_mine_like_object',
            'confidence': random.uniform(0.6, 0.8),
            'position': {
                'distance_m': random.uniform(200, 1000),
                'bearing_deg': random.uniform(0, 360)
            },
            'characteristics': {
                'size_estimate_m': random.uniform(0.8, 2.0),
                'metallic_signature': random.random() < 0.7,
                'magnetic_anomaly': random.random() < 0.5
            }
        }
    
    def _generate_submarine_threat(self, minute, convoy_lat, convoy_lon):
        """Generate submarine threat detection"""
        return {
            'detection_id': f'sub_{minute}_{random.randint(100, 999)}',
            'object_type': 'submarine_periscope',
            'confidence': random.uniform(0.5, 0.75),
            'position': {
                'distance_m': random.uniform(1500, 4000),
                'bearing_deg': random.uniform(0, 360)
            },
            'acoustic_signature': {
                'propeller_count': random.randint(1, 2),
                'frequency_hz': random.uniform(50, 200),
                'cavitation_noise': random.random() < 0.6
            }
        }
    
    def _generate_environmental_hazard(self, minute, convoy_lat, convoy_lon):
        """Generate environmental hazard detection"""
        hazard_types = ['debris_field', 'shallow_water', 'oil_spill', 'fishing_nets']
        hazard_type = random.choice(hazard_types)
        
        return {
            'detection_id': f'env_{minute}_{random.randint(100, 999)}',
            'object_type': 'environmental_hazard',
            'hazard_type': hazard_type,
            'threat_level': random.uniform(0.3, 0.8),
            'position': {
                'distance_m': random.uniform(500, 2000),
                'bearing_deg': random.uniform(0, 360)
            },
            'extent': {
                'width_m': random.uniform(50, 500),
                'length_m': random.uniform(100, 1000)
            }
        }
    
    def _generate_false_positive(self, minute, convoy_lat, convoy_lon):
        """Generate false positive detection"""
        return {
            'detection_id': f'false_{minute}_{random.randint(100, 999)}',
            'object_type': 'unknown',
            'confidence': random.uniform(0.3, 0.6),
            'position': {
                'distance_m': random.uniform(1000, 5000),
                'bearing_deg': random.uniform(0, 360)
            },
            'classification': 'marine_life_or_debris',
            'false_positive_probability': random.uniform(0.7, 0.9)
        }
    
    def save_to_s3(self, frames, bucket_name, scenario_type):
        """Save generated data to S3"""
        s3 = boto3.client('s3')
        
        ndjson_content = '\n'.join([json.dumps(frame) for frame in frames])
        key = f'enhanced/{scenario_type}/data.jsonl'
        
        s3.put_object(
            Bucket=bucket_name,
            Key=key,
            Body=ndjson_content,
            ContentType='application/x-ndjson'
        )
        
        print(f"Saved {len(frames)} enhanced frames to s3://{bucket_name}/{key}")

def main():
    parser = argparse.ArgumentParser(description='Generate enhanced maritime training data')
    parser.add_argument('--scenarios', default='piracy_ambush,swarm_interdiction,mine_field,submarine_contact')
    parser.add_argument('--output-bucket', required=True)
    parser.add_argument('--duration', type=int, default=90)
    
    args = parser.parse_args()
    
    generator = EnhancedMaritimeDataGenerator()
    scenarios = args.scenarios.split(',')
    
    for scenario in scenarios:
        print(f"Generating enhanced scenario: {scenario}")
        frames = generator.generate_scenario(scenario.strip(), args.duration)
        generator.save_to_s3(frames, args.output_bucket, scenario.strip())

if __name__ == '__main__':
    main()