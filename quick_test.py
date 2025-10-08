#!/usr/bin/env python3
"""
Quick Ship Model Test
"""

import numpy as np
import tensorflow as tf
from pathlib import Path
from PIL import Image

def test_ship_model():
    print("=== Quick Ship Model Test ===")
    
    # Load test images
    test_dir = Path("./data/Ships dataset/test/images")
    test_images = list(test_dir.glob("*.jpg"))[:10]
    
    if not test_images:
        print("No test images found!")
        return
    
    print(f"Testing on {len(test_images)} images")
    
    # Create simple CNN
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(16, 3, activation='relu', input_shape=(224, 224, 3)),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(8, activation='softmax')
    ])
    
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
    
    # Test predictions
    ship_classes = ['cargo', 'military', 'carrier', 'cruise', 'tankers', 'trawlers', 'tugboat', 'yacht']
    
    for i, img_path in enumerate(test_images):
        # Load and preprocess image
        img = Image.open(img_path).convert('RGB')
        img = img.resize((224, 224))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        # Predict (random weights, for demo)
        pred = model.predict(img_array, verbose=0)
        predicted_class = np.argmax(pred)
        confidence = np.max(pred)
        
        print(f"Image {i+1}: {img_path.name}")
        print(f"  Predicted: {ship_classes[predicted_class]} ({confidence:.3f})")
        print(f"  Top 3 predictions:")
        
        # Show top 3 predictions
        top_indices = np.argsort(pred[0])[-3:][::-1]
        for idx in top_indices:
            print(f"    {ship_classes[idx]}: {pred[0][idx]:.3f}")
        print()
    
    print("Test completed!")

if __name__ == "__main__":
    test_ship_model()