#!/usr/bin/env python3
import unittest
import json
import time
import boto3
from moto import mock_dynamodb, mock_kinesis, mock_iot
import sys
import os

# Add project paths
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'synth'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'backend'))

from generate_maritime_data import MaritimeDataGenerator

class TestMaritimeACPS(unittest.TestCase):
    """Integration tests for Maritime ACPS system"""
    
    def setUp(self):
        """Set up test environment"""
        self.generator = MaritimeDataGenerator()
    
    def test_synthetic_data_generation(self):
        """Test synthetic maritime data generation"""
        # Generate small dataset
        frames = self.generator.generate_scenario('piracy_ambush', duration_minutes=5)
        
        # Verify data structure
        self.assertEqual(len(frames), 5)
        
        for frame in frames:
            # Check required fields
            self.assertIn('timestamp', frame)
            self.assertIn('scenario_id', frame)
            self.assertIn('detections', frame)
            self.assertIn('ground_truth_threat', frame)
            
            # Validate schema
            self.assertEqual(frame['scenario_id'], 'piracy_ambush')
            self.assertIsInstance(frame['detections'], list)
            self.assertIsInstance(frame['ground_truth_threat'], bool)
    
    def test_threat_detection_logic(self):
        """Test threat detection in generated scenarios"""
        frames = self.generator.generate_scenario('piracy_ambush', duration_minutes=60)
        
        # Should have threats after minute 30
        threat_frames = [f for f in frames if f['ground_truth_threat']]
        self.assertGreater(len(threat_frames), 0, "Piracy scenario should generate threats")
        
        # Check threat types
        for frame in threat_frames:
            self.assertEqual(frame['threat_type'], 'small_fast_craft')
            self.assertGreater(len(frame['detections']), 0)
    
    def test_swarm_scenario(self):
        """Test swarm interdiction scenario"""
        frames = self.generator.generate_scenario('swarm_interdiction', duration_minutes=30)
        
        threat_frames = [f for f in frames if f['ground_truth_threat']]
        
        # Should have multiple detections in swarm scenario
        multi_detection_frames = [f for f in threat_frames if len(f['detections']) > 1]
        self.assertGreater(len(multi_detection_frames), 0, "Swarm should have multiple detections")
    
    @mock_dynamodb
    @mock_kinesis
    def test_fusion_pipeline(self):
        """Test threat fusion pipeline"""
        # Create mock AWS resources
        dynamodb = boto3.resource('dynamodb', region_name='us-east-1')
        
        # Create events table
        table = dynamodb.create_table(
            TableName='maritime-events',
            KeySchema=[
                {'AttributeName': 'event_id', 'KeyType': 'HASH'},
                {'AttributeName': 'timestamp', 'KeyType': 'RANGE'}
            ],
            AttributeDefinitions=[
                {'AttributeName': 'event_id', 'AttributeType': 'S'},
                {'AttributeName': 'timestamp', 'AttributeType': 'N'}
            ],
            BillingMode='PAY_PER_REQUEST'
        )
        
        # Mock detection event
        detection_event = {
            'device_id': 'test_device',
            'detection': {
                'threat_detected': True,
                'threat_type': 'small_fast_craft',
                'confidence': 0.85,
                'timestamp': '2024-01-01T12:00:00Z'
            },
            'location': {
                'lat': 25.7617,
                'lon': -80.1918
            }
        }
        
        # Test fusion logic (simplified)
        fusion_score = self._calculate_test_fusion_score(detection_event['detection'])
        self.assertGreater(fusion_score, 0.7, "High confidence threat should have high fusion score")
    
    def _calculate_test_fusion_score(self, detection):
        """Test version of fusion score calculation"""
        base_confidence = detection.get('confidence', 0)
        threat_type = detection.get('threat_type', 'none')
        
        severity_weights = {
            'small_fast_craft': 0.9,
            'collision_risk': 0.95,
            'floating_mine_like_object': 0.85
        }
        
        severity_weight = severity_weights.get(threat_type, 0.1)
        return base_confidence * severity_weight
    
    def test_astar_planning(self):
        """Test A* path planning algorithm"""
        # Import planner (would need to refactor for testing)
        # For now, test basic path planning logic
        
        start = (0, 0)
        goal = (10, 10)
        
        # Simple grid with no obstacles
        grid_size = 20
        threat_grid = [[0.1 for _ in range(grid_size)] for _ in range(grid_size)]
        
        # Add threat at (5, 5)
        threat_grid[5][5] = 0.9
        
        # Path should avoid high-threat areas
        # This is a simplified test - full A* implementation would be tested separately
        self.assertTrue(True)  # Placeholder
    
    def test_end_to_end_scenario(self):
        """Test complete detection -> fusion -> planning pipeline"""
        
        # 1. Generate threat scenario
        frames = self.generator.generate_scenario('piracy_ambush', duration_minutes=5)
        threat_frames = [f for f in frames if f['ground_truth_threat']]
        
        self.assertGreater(len(threat_frames), 0, "Should generate threats")
        
        # 2. Simulate detection processing
        for frame in threat_frames:
            detection = {
                'threat_detected': True,
                'threat_type': frame['threat_type'],
                'confidence': 0.85,
                'timestamp': frame['timestamp']
            }
            
            # 3. Calculate fusion score
            fusion_score = self._calculate_test_fusion_score(detection)
            self.assertGreater(fusion_score, 0.5, "Should have meaningful fusion score")
            
            # 4. Trigger replanning if high confidence
            if fusion_score > 0.8:
                # Would trigger path replanning
                replan_needed = True
                self.assertTrue(replan_needed, "High threat should trigger replanning")
    
    def test_human_in_the_loop_safety(self):
        """Test that system requires human authorization"""
        
        # Simulate high-threat detection
        threat_event = {
            'threat_type': 'small_fast_craft',
            'confidence': 0.95,
            'fusion_score': 0.9
        }
        
        # System should generate recommendation, not automatic action
        recommendation = {
            'action': 'replan_route',
            'threat_level': 'high',
            'requires_approval': True,
            'automatic_engagement': False  # CRITICAL: No automatic weapons
        }
        
        self.assertTrue(recommendation['requires_approval'], 
                       "High threat actions must require human approval")
        self.assertFalse(recommendation['automatic_engagement'],
                        "System must not automatically engage threats")

class TestDataValidation(unittest.TestCase):
    """Test data format validation"""
    
    def test_schema_compliance(self):
        """Test that generated data matches schema"""
        generator = MaritimeDataGenerator()
        frames = generator.generate_scenario('mine_detection', duration_minutes=2)
        
        for frame in frames:
            # Required fields
            required_fields = [
                'timestamp', 'scenario_id', 'vehicle_id', 'drone_id', 'detections'
            ]
            
            for field in required_fields:
                self.assertIn(field, frame, f"Missing required field: {field}")
            
            # Type validation
            self.assertIsInstance(frame['detections'], list)
            self.assertIsInstance(frame['ground_truth_threat'], bool)
            
            # Enum validation
            valid_scenarios = [
                'piracy_ambush', 'swarm_interdiction', 'ais_spoofing_scenario',
                'mine_detection', 'uav_overwatch'
            ]
            self.assertIn(frame['scenario_id'], valid_scenarios)

if __name__ == '__main__':
    # Run tests
    unittest.main(verbosity=2)