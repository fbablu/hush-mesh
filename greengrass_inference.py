#!/usr/bin/env python3
"""
Greengrass Edge Inference for Maritime ACPS
"""

import json
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import awsiot.greengrasscoreipc
from awsiot.greengrasscoreipc.model import PublishToTopicRequest, PublishMessage

class GreengrassShipDetector:
    def __init__(self, model_path='/opt/ml/model/model.pth'):
        self.device = torch.device('cpu')  # Edge device uses CPU
        self.model = self.load_model(model_path)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.class_names = ['cargo', 'military', 'carrier', 'cruise', 'tankers', 'trawlers', 'tugboat', 'yacht']
        self.ipc_client = awsiot.greengrasscoreipc.connect()
    
    def load_model(self, model_path):
        """Load compiled Neo model"""
        from dlr import DLRModel
        model = DLRModel(model_path, 'cpu')
        return model
    
    def predict(self, image_data):
        """Predict ship class from image"""
        # Preprocess image
        if isinstance(image_data, bytes):
            image = Image.open(io.BytesIO(image_data)).convert('RGB')
        else:
            image = Image.open(image_data).convert('RGB')
        
        input_tensor = self.transform(image).unsqueeze(0)
        
        # Run inference
        with torch.no_grad():
            output = self.model.run(input_tensor.numpy())
            probabilities = torch.softmax(torch.tensor(output[0]), dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_class].item()
        
        return {
            'class': self.class_names[predicted_class],
            'confidence': confidence,
            'threat_level': self.assess_threat(predicted_class, confidence)
        }
    
    def assess_threat(self, class_id, confidence):
        """Assess threat level based on ship type"""
        threat_levels = {
            0: 'low',      # cargo
            1: 'high',     # military
            2: 'medium',   # carrier
            3: 'low',      # cruise
            4: 'low',      # tankers
            5: 'low',      # trawlers
            6: 'low',      # tugboat
            7: 'low'       # yacht
        }
        
        base_threat = threat_levels.get(class_id, 'unknown')
        
        # Adjust based on confidence
        if confidence < 0.7:
            return 'unknown'
        elif base_threat == 'high' and confidence > 0.9:
            return 'critical'
        
        return base_threat
    
    def publish_detection(self, detection_result, vessel_id):
        """Publish detection to IoT Core"""
        message = {
            'vessel_id': vessel_id,
            'timestamp': int(time.time()),
            'detection': detection_result,
            'source': 'greengrass_edge'
        }
        
        request = PublishToTopicRequest()
        request.topic = 'maritime/acps/detections'
        request.publish_message = PublishMessage()
        request.publish_message.json_message = json.dumps(message)
        
        self.ipc_client.publish_to_topic(request)

def lambda_handler(event, context):
    """Greengrass Lambda handler for ship detection"""
    detector = GreengrassShipDetector()
    
    # Process incoming image
    image_data = event.get('image_data')
    vessel_id = event.get('vessel_id', 'unknown')
    
    if image_data:
        result = detector.predict(image_data)
        detector.publish_detection(result, vessel_id)
        
        return {
            'statusCode': 200,
            'body': json.dumps(result)
        }
    
    return {
        'statusCode': 400,
        'body': json.dumps({'error': 'No image data provided'})
    }