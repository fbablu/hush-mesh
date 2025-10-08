#!/usr/bin/env python3
"""
Simplified Ship Classification Model Training
"""

import os
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from pathlib import Path

# Set Kaggle credentials
os.environ['KAGGLE_USERNAME'] = 'wasihussain'
os.environ['KAGGLE_KEY'] = 'b950d284514080a980ebc92ee3fb969e'

def download_dataset():
    """Download ship dataset from Kaggle"""
    try:
        import kaggle
        print("Downloading ships dataset...")
        kaggle.api.dataset_download_files('vinayakshanawad/ships-dataset', path='./data', unzip=True)
        print("Dataset downloaded successfully!")
        return True
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        return False

def create_sample_data():
    """Create sample synthetic data for demonstration"""
    print("Creating sample synthetic data...")
    
    # Create sample images (random data for demo)
    np.random.seed(42)
    
    # Simulate 4 ship classes
    classes = ['cargo', 'military', 'passenger', 'fishing']
    samples_per_class = 100
    
    X_data = []
    y_data = []
    
    for class_idx, class_name in enumerate(classes):
        for i in range(samples_per_class):
            # Generate random image-like data
            img = np.random.rand(224, 224, 3)
            X_data.append(img)
            y_data.append(class_idx)
    
    X_data = np.array(X_data)
    y_data = np.array(y_data)
    
    # Convert to categorical
    y_data = keras.utils.to_categorical(y_data, len(classes))
    
    return X_data, y_data, classes

def build_cnn_model(num_classes):
    """Build CNN model for ship classification"""
    model = keras.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def main():
    print("=== Ship Classification Model Training ===")
    
    # Try to download real dataset first
    dataset_downloaded = download_dataset()
    
    if not dataset_downloaded:
        print("Using synthetic sample data for demonstration...")
        X_data, y_data, class_names = create_sample_data()
        
        # Split data
        split_idx = int(0.8 * len(X_data))
        X_train, X_test = X_data[:split_idx], X_data[split_idx:]
        y_train, y_test = y_data[:split_idx], y_data[split_idx:]
        
        print(f"Training data shape: {X_train.shape}")
        print(f"Test data shape: {X_test.shape}")
        print(f"Classes: {class_names}")
        
        # Build model
        model = build_cnn_model(len(class_names))
        print("\nModel Architecture:")
        model.summary()
        
        # Train model
        print("\nTraining model...")
        history = model.fit(
            X_train, y_train,
            batch_size=32,
            epochs=5,  # Reduced for demo
            validation_data=(X_test, y_test),
            verbose=1
        )
        
        # Evaluate
        test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
        print(f"\nTest Accuracy: {test_accuracy:.4f}")
        
        # Save model
        model.save('models/ship_classifier_demo.h5')
        
        # Save class names
        with open('models/class_names.json', 'w') as f:
            json.dump(class_names, f)
        
        print("Model saved successfully!")
        
        # Create inference example
        print("\n=== Inference Example ===")
        sample_img = X_test[0:1]  # Take first test image
        prediction = model.predict(sample_img, verbose=0)
        predicted_class = class_names[np.argmax(prediction)]
        confidence = np.max(prediction)
        
        print(f"Sample prediction: {predicted_class} (confidence: {confidence:.4f})")
    
    else:
        print("Real dataset downloaded! You can now use the full ship_classifier.py")

if __name__ == "__main__":
    main()