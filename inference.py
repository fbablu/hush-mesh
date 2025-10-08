#!/usr/bin/env python3
"""
Inference script for ship classification model
"""

import json
import numpy as np
from PIL import Image
import tensorflow as tf
from pathlib import Path

class ShipInference:
    def __init__(self, model_path="ship_classifier_model.h5", class_names_path="class_names.json"):
        self.model = tf.keras.models.load_model(model_path)
        
        with open(class_names_path, 'r') as f:
            self.class_names = json.load(f)
            
        self.img_size = (224, 224)
    
    def predict_single_image(self, image_path):
        """Predict class for a single image"""
        img = Image.open(image_path).convert('RGB')
        img = img.resize(self.img_size)
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        predictions = self.model.predict(img_array, verbose=0)
        predicted_class_idx = np.argmax(predictions[0])
        confidence = predictions[0][predicted_class_idx]
        
        return {
            'class': self.class_names[predicted_class_idx],
            'confidence': float(confidence),
            'all_predictions': {
                self.class_names[i]: float(predictions[0][i]) 
                for i in range(len(self.class_names))
            }
        }
    
    def predict_batch(self, image_paths):
        """Predict classes for multiple images"""
        results = []
        for img_path in image_paths:
            result = self.predict_single_image(img_path)
            result['image_path'] = str(img_path)
            results.append(result)
        return results

def main():
    # Example usage
    inference = ShipInference()
    
    # Test with a sample image (replace with actual image path)
    test_image = "test_ship.jpg"
    if Path(test_image).exists():
        result = inference.predict_single_image(test_image)
        print(f"Prediction: {result['class']} (confidence: {result['confidence']:.4f})")
    else:
        print("No test image found. Place a ship image as 'test_ship.jpg' to test inference.")

if __name__ == "__main__":
    main()