#!/usr/bin/env python3
"""
Maritime ACPS Integration with SageMaker Ship Classification
"""

import boto3
import json
import time
from datetime import datetime

class MaritimeACPSIntegration:
    def __init__(self):
        self.sagemaker_runtime = boto3.client('sagemaker-runtime')
        self.iot_client = boto3.client('iot-data')
        self.dynamodb = boto3.resource('dynamodb')
        self.kinesis = boto3.client('kinesis')
        
        # Maritime ACPS configuration
        self.endpoint_name = 'maritime-acps-ship-classifier'
        self.threat_table = 'maritime-threats'
        self.kinesis_stream = 'maritime-telemetry'
    
    def classify_vessel(self, image_data, vessel_metadata):
        """Classify vessel using SageMaker endpoint"""
        try:
            response = self.sagemaker_runtime.invoke_endpoint(
                EndpointName=self.endpoint_name,
                ContentType='application/x-image',
                Body=image_data
            )
            
            result = json.loads(response['Body'].read().decode())
            
            # Add metadata
            result.update({
                'timestamp': datetime.utcnow().isoformat(),
                'vessel_id': vessel_metadata.get('vessel_id'),
                'location': vessel_metadata.get('location'),
                'convoy_id': vessel_metadata.get('convoy_id')
            })
            
            return result
            
        except Exception as e:
            print(f"Classification error: {e}")
            return None
    
    def assess_threat_level(self, classification_result):
        """Assess threat level for convoy protection"""
        ship_class = classification_result.get('class')
        confidence = classification_result.get('confidence', 0)
        
        # Threat assessment rules
        threat_rules = {
            'military': 'high',
            'carrier': 'medium', 
            'cargo': 'low',
            'cruise': 'low',
            'tankers': 'low',
            'trawlers': 'low',
            'tugboat': 'low',
            'yacht': 'low'
        }
        
        base_threat = threat_rules.get(ship_class, 'unknown')
        
        # Adjust based on confidence and context
        if confidence < 0.7:
            return 'unknown'
        elif base_threat == 'high' and confidence > 0.9:
            return 'critical'
        
        return base_threat
    
    def store_threat_event(self, classification_result, threat_level):
        """Store threat event in DynamoDB"""
        table = self.dynamodb.Table(self.threat_table)
        
        event = {
            'event_id': f"{classification_result['vessel_id']}_{int(time.time())}",
            'timestamp': classification_result['timestamp'],
            'vessel_id': classification_result['vessel_id'],
            'ship_class': classification_result['class'],
            'confidence': classification_result['confidence'],
            'threat_level': threat_level,
            'location': classification_result.get('location'),
            'convoy_id': classification_result.get('convoy_id'),
            'requires_human_review': threat_level in ['high', 'critical']
        }
        
        table.put_item(Item=event)
        return event
    
    def publish_to_kinesis(self, event_data):
        """Publish threat event to Kinesis for real-time processing"""
        self.kinesis.put_record(
            StreamName=self.kinesis_stream,
            Data=json.dumps(event_data),
            PartitionKey=event_data['vessel_id']
        )
    
    def send_iot_alert(self, threat_event):
        """Send alert to IoT devices and dashboard"""
        if threat_event['threat_level'] in ['high', 'critical']:
            alert_message = {
                'alert_type': 'vessel_threat',
                'threat_level': threat_event['threat_level'],
                'vessel_class': threat_event['ship_class'],
                'location': threat_event['location'],
                'timestamp': threat_event['timestamp'],
                'requires_action': True
            }
            
            # Publish to IoT topic
            self.iot_client.publish(
                topic='maritime/acps/alerts',
                qos=1,
                payload=json.dumps(alert_message)
            )
    
    def process_vessel_detection(self, image_data, vessel_metadata):
        """Complete vessel processing pipeline"""
        # Classify vessel
        classification = self.classify_vessel(image_data, vessel_metadata)
        
        if not classification:
            return None
        
        # Assess threat
        threat_level = self.assess_threat_level(classification)
        
        # Store event
        threat_event = self.store_threat_event(classification, threat_level)
        
        # Publish to streams
        self.publish_to_kinesis(threat_event)
        
        # Send alerts if needed
        self.send_iot_alert(threat_event)
        
        return threat_event

def demo_integration():
    """Demo Maritime ACPS integration"""
    print("=== Maritime ACPS Integration Demo ===")
    
    acps = MaritimeACPSIntegration()
    
    # Simulate vessel detection
    vessel_metadata = {
        'vessel_id': 'VESSEL_001',
        'location': {'lat': 25.7617, 'lon': -80.1918},
        'convoy_id': 'CONVOY_ALPHA'
    }
    
    # Simulate image data (would come from camera sensors)
    image_data = b"simulated_image_data"
    
    print("Processing vessel detection...")
    
    # Note: This would fail without actual SageMaker endpoint
    # but shows the integration pattern
    try:
        result = acps.process_vessel_detection(image_data, vessel_metadata)
        print(f"Threat assessment: {result}")
    except Exception as e:
        print(f"Demo error (expected without AWS setup): {e}")
        
        # Show expected workflow
        print("\nExpected Maritime ACPS Workflow:")
        print("1. Camera captures vessel image")
        print("2. SageMaker classifies ship type")
        print("3. Threat level assessed")
        print("4. Event stored in DynamoDB")
        print("5. Real-time alert sent via Kinesis")
        print("6. IoT devices receive threat notification")
        print("7. Human operator reviews high-threat vessels")

if __name__ == "__main__":
    demo_integration()