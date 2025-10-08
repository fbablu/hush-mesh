import json
import boto3
import os
from datetime import datetime
import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)

dynamodb = boto3.resource('dynamodb')
kinesis = boto3.client('kinesis')

def handler(event, context):
    """Lambda function to process and fuse maritime threat detections"""
    
    events_table = dynamodb.Table(os.environ['EVENTS_TABLE'])
    
    try:
        # Process Kinesis records
        for record in event['Records']:
            # Decode Kinesis data
            payload = json.loads(record['kinesis']['data'])
            
            # Extract detection data
            detection = payload.get('detection', {})
            device_id = payload.get('device_id')
            location = payload.get('location', {})
            
            if detection.get('threat_detected'):
                # Store in DynamoDB
                event_item = {
                    'event_id': f"{device_id}_{int(datetime.utcnow().timestamp())}",
                    'timestamp': int(datetime.utcnow().timestamp()),
                    'device_id': device_id,
                    'threat_type': detection.get('threat_type'),
                    'confidence': detection.get('confidence'),
                    'location': location,
                    'status': 'active',
                    'fusion_score': calculate_fusion_score(detection, location)
                }
                
                events_table.put_item(Item=event_item)
                
                # Trigger path replanning if high confidence threat
                if detection.get('confidence', 0) > 0.8:
                    trigger_replanning(event_item)
                
                logger.info(f"Processed threat detection: {detection['threat_type']}")
        
        return {
            'statusCode': 200,
            'body': json.dumps({'processed': len(event['Records'])})
        }
        
    except Exception as e:
        logger.error(f"Error processing detections: {str(e)}")
        return {
            'statusCode': 500,
            'body': json.dumps({'error': str(e)})
        }

def calculate_fusion_score(detection, location):
    """Calculate fused threat score based on multiple factors"""
    
    base_confidence = detection.get('confidence', 0)
    threat_type = detection.get('threat_type', 'none')
    
    # Threat type severity weights
    severity_weights = {
        'small_fast_craft': 0.9,
        'collision_risk': 0.95,
        'floating_mine_like_object': 0.85,
        'drone_overwater': 0.7,
        'acoustic_gunshot': 0.8,
        'ais_spoofing': 0.6,
        'suspicious_loitering_vessel': 0.5,
        'unregistered_vessel': 0.4,
        'diver_or_swimmer': 0.3
    }
    
    severity_weight = severity_weights.get(threat_type, 0.1)
    
    # Distance factor (closer = higher threat)
    # Assume location contains distance if available
    distance_factor = 1.0  # Default
    
    # Calculate fusion score
    fusion_score = base_confidence * severity_weight * distance_factor
    
    return min(fusion_score, 1.0)

def trigger_replanning(threat_event):
    """Trigger convoy path replanning for high-confidence threats"""
    
    try:
        # Publish to replanning topic
        kinesis.put_record(
            StreamName='maritime-replanning',
            Data=json.dumps({
                'action': 'replan',
                'threat_event': threat_event,
                'timestamp': datetime.utcnow().isoformat()
            }),
            PartitionKey=threat_event['device_id']
        )
        
        logger.info(f"Triggered replanning for threat: {threat_event['event_id']}")
        
    except Exception as e:
        logger.error(f"Failed to trigger replanning: {str(e)}")