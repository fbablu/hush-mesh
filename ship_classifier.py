#!/usr/bin/env python3
"""
Ship Image Classification Model
Uses CNN architecture for classifying ship images from Kaggle dataset
"""

import os
import json
import zipfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from PIL import Image
import kaggle

class ShipClassifier:
    def __init__(self, img_size=(224, 224), batch_size=32):
        self.img_size = img_size
        self.batch_size = batch_size
        self.model = None
        self.class_names = []
        
    def setup_kaggle_api(self, username, key):
        """Setup Kaggle API credentials"""
        os.environ['KAGGLE_USERNAME'] = username
        os.environ['KAGGLE_KEY'] = key
        
    def download_dataset(self, dataset_name="vinayakshanawad/ships-dataset"):
        """Download and extract Kaggle dataset"""
        print("Downloading dataset...")
        kaggle.api.dataset_download_files(dataset_name, path="./data", unzip=True)
        print("Dataset downloaded and extracted to ./data")
        
    def load_and_preprocess_data(self, data_dir="./data"):
        """Load images and create train/validation splits"""
        data_path = Path(data_dir)
        
        # Find the actual data directory structure
        if (data_path / "ships").exists():
            ships_dir = data_path / "ships"
        else:
            # Look for any subdirectory with images
            subdirs = [d for d in data_path.iterdir() if d.is_dir()]
            ships_dir = subdirs[0] if subdirs else data_path
            
        print(f"Loading data from: {ships_dir}")
        
        # Get class names from subdirectories
        self.class_names = sorted([d.name for d in ships_dir.iterdir() if d.is_dir()])
        print(f"Found classes: {self.class_names}")
        
        # Create data generators
        datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            validation_split=0.2
        )
        
        self.train_generator = datagen.flow_from_directory(
            ships_dir,
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            subset='training'
        )
        
        self.val_generator = datagen.flow_from_directory(
            ships_dir,
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            subset='validation'
        )
        
        return self.train_generator, self.val_generator
    
    def build_model(self):
        """Build CNN model for ship classification"""
        num_classes = len(self.class_names)
        
        model = keras.Sequential([
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=(*self.img_size, 3)),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dropout(0.5),
            layers.Dense(512, activation='relu'),
            layers.Dense(num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        return model
    
    def train_model(self, epochs=20):
        """Train the model"""
        if not self.model:
            self.build_model()
            
        callbacks = [
            keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(factor=0.2, patience=3)
        ]
        
        history = self.model.fit(
            self.train_generator,
            epochs=epochs,
            validation_data=self.val_generator,
            callbacks=callbacks
        )
        
        return history
    
    def evaluate_model(self):
        """Evaluate model performance"""
        # Get predictions
        val_loss, val_accuracy = self.model.evaluate(self.val_generator)
        print(f"Validation Accuracy: {val_accuracy:.4f}")
        
        # Generate predictions for confusion matrix
        self.val_generator.reset()
        predictions = self.model.predict(self.val_generator)
        predicted_classes = np.argmax(predictions, axis=1)
        true_classes = self.val_generator.classes
        
        # Classification report
        print("\nClassification Report:")
        print(classification_report(true_classes, predicted_classes, 
                                  target_names=self.class_names))
        
        # Confusion matrix
        cm = confusion_matrix(true_classes, predicted_classes)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.class_names, yticklabels=self.class_names)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig('confusion_matrix.png')
        plt.show()
        
        return val_accuracy
    
    def save_model(self, filepath="ship_classifier_model.h5"):
        """Save the trained model"""
        self.model.save(filepath)
        
        # Save class names
        with open("class_names.json", "w") as f:
            json.dump(self.class_names, f)
        
        print(f"Model saved to {filepath}")
    
    def predict_image(self, image_path):
        """Predict class for a single image"""
        img = Image.open(image_path).convert('RGB')
        img = img.resize(self.img_size)
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        predictions = self.model.predict(img_array)
        predicted_class_idx = np.argmax(predictions[0])
        confidence = predictions[0][predicted_class_idx]
        
        return self.class_names[predicted_class_idx], confidence

def main():
    # Initialize classifier
    classifier = ShipClassifier()
    
    # Setup Kaggle API
    api_credentials = {"username": "wasihussain", "key": "b950d284514080a980ebc92ee3fb969e"}
    classifier.setup_kaggle_api(api_credentials["username"], api_credentials["key"])
    
    # Download dataset
    classifier.download_dataset()
    
    # Load and preprocess data
    train_gen, val_gen = classifier.load_and_preprocess_data()
    
    # Build and train model
    model = classifier.build_model()
    print(model.summary())
    
    # Train the model
    print("Starting training...")
    history = classifier.train_model(epochs=20)
    
    # Evaluate model
    accuracy = classifier.evaluate_model()
    
    # Save model
    classifier.save_model()
    
    print(f"Training completed! Final accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    main()