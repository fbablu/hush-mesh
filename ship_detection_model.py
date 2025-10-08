#!/usr/bin/env python3
"""
Ship Detection and Classification Model
Converts YOLO format dataset to classification format
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from PIL import Image


class ShipDetectionClassifier:
    def __init__(self, img_size=(224, 224), batch_size=32):
        self.img_size = img_size
        self.batch_size = batch_size
        self.model = None
        self.class_names = {
            0: 'cargo',
            1: 'military', 
            2: 'carrier',
            3: 'cruise',
            4: 'tankers',
            5: 'trawlers',
            6: 'tugboat',
            7: 'yacht'
        }
        
    def load_yolo_data(self, data_dir="./data/Ships dataset"):
        """Load and convert YOLO format data to classification format"""
        data_path = Path(data_dir)
        
        # Process train, test, valid splits
        splits = ['train', 'test', 'valid']
        all_images = []
        all_labels = []
        
        for split in splits:
            images_dir = data_path / split / 'images'
            labels_dir = data_path / split / 'labels'
            
            if not images_dir.exists() or not labels_dir.exists():
                continue
                
            print(f"Processing {split} split...")
            
            for img_file in images_dir.glob('*.jpg'):
                label_file = labels_dir / f"{img_file.stem}.txt"
                
                if label_file.exists():
                    # Read label (first class in YOLO format)
                    with open(label_file, 'r') as f:
                        line = f.readline().strip()
                        if line:
                            class_id = int(line.split()[0])
                            all_images.append(str(img_file))
                            all_labels.append(class_id)
        
        print(f"Total images loaded: {len(all_images)}")
        print(f"Class distribution: {np.bincount(all_labels)}")
        
        return all_images, all_labels
    
    def create_data_generators(self, image_paths, labels):
        """Create TensorFlow data generators"""
        
        def load_and_preprocess_image(image_path, label):
            # Load image
            image = tf.io.read_file(image_path)
            image = tf.image.decode_jpeg(image, channels=3)
            image = tf.image.resize(image, self.img_size)
            image = tf.cast(image, tf.float32) / 255.0
            
            # Convert label to one-hot
            label = tf.one_hot(label, len(self.class_names))
            
            return image, label
        
        # Split data
        split_idx = int(0.8 * len(image_paths))
        
        train_paths = image_paths[:split_idx]
        train_labels = labels[:split_idx]
        val_paths = image_paths[split_idx:]
        val_labels = labels[split_idx:]
        
        # Create datasets
        train_dataset = tf.data.Dataset.from_tensor_slices((train_paths, train_labels))
        train_dataset = train_dataset.map(load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
        train_dataset = train_dataset.batch(self.batch_size).prefetch(tf.data.AUTOTUNE)
        
        val_dataset = tf.data.Dataset.from_tensor_slices((val_paths, val_labels))
        val_dataset = val_dataset.map(load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
        val_dataset = val_dataset.batch(self.batch_size).prefetch(tf.data.AUTOTUNE)
        
        return train_dataset, val_dataset
    
    def build_model(self):
        """Build CNN model for ship classification"""
        num_classes = len(self.class_names)
        
        model = keras.Sequential([
            # Feature extraction layers
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=(*self.img_size, 3)),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            
            layers.Conv2D(256, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            
            # Classification layers
            layers.GlobalAveragePooling2D(),
            layers.Dropout(0.5),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        return model
    
    def train_model(self, train_dataset, val_dataset, epochs=20):
        """Train the model"""
        if not self.model:
            self.build_model()
        
        callbacks = [
            keras.callbacks.EarlyStopping(
                patience=5, 
                restore_best_weights=True,
                monitor='val_accuracy'
            ),
            keras.callbacks.ReduceLROnPlateau(
                factor=0.2, 
                patience=3,
                monitor='val_loss'
            ),
            keras.callbacks.ModelCheckpoint(
                'models/best_ship_model.h5',
                save_best_only=True,
                monitor='val_accuracy'
            )
        ]
        
        history = self.model.fit(
            train_dataset,
            epochs=epochs,
            validation_data=val_dataset,
            callbacks=callbacks,
            verbose=1
        )
        
        return history
    
    def evaluate_model(self, val_dataset):
        """Evaluate model performance"""
        # Get predictions
        predictions = []
        true_labels = []
        
        for batch_images, batch_labels in val_dataset:
            batch_preds = self.model.predict(batch_images, verbose=0)
            predictions.extend(np.argmax(batch_preds, axis=1))
            true_labels.extend(np.argmax(batch_labels, axis=1))
        
        # Calculate accuracy
        accuracy = np.mean(np.array(predictions) == np.array(true_labels))
        print(f"Validation Accuracy: {accuracy:.4f}")
        
        # Classification report
        class_names_list = [self.class_names[i] for i in range(len(self.class_names))]
        print("\nClassification Report:")
        print(classification_report(true_labels, predictions, target_names=class_names_list))
        
        # Confusion matrix
        cm = confusion_matrix(true_labels, predictions)
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names_list, yticklabels=class_names_list)
        plt.title('Ship Classification Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig('results/confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return accuracy
    
    def save_model(self, filepath="models/ship_classifier.h5"):
        """Save the trained model"""
        self.model.save(filepath)
        
        # Save class names
        with open("models/class_names.json", "w") as f:
            json.dump(self.class_names, f)
        
        print(f"Model saved to {filepath}")
    
    def predict_image(self, image_path):
        """Predict class for a single image"""
        img = Image.open(image_path).convert('RGB')
        img = img.resize(self.img_size)
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        predictions = self.model.predict(img_array, verbose=0)
        predicted_class_idx = np.argmax(predictions[0])
        confidence = predictions[0][predicted_class_idx]
        
        return self.class_names[predicted_class_idx], confidence

def main():
    print("=== Ship Detection and Classification Model ===")
    
    # Initialize classifier
    classifier = ShipDetectionClassifier()
    
    # Load YOLO format data
    print("Loading YOLO dataset...")
    image_paths, labels = classifier.load_yolo_data()
    
    if len(image_paths) == 0:
        print("No data found! Please check dataset path.")
        return
    
    # Create data generators
    print("Creating data generators...")
    train_dataset, val_dataset = classifier.create_data_generators(image_paths, labels)
    
    # Build model
    print("Building model...")
    model = classifier.build_model()
    print("\nModel Architecture:")
    model.summary()
    
    # Train model
    print("\nStarting training...")
    history = classifier.train_model(train_dataset, val_dataset, epochs=15)
    
    # Evaluate model
    print("\nEvaluating model...")
    accuracy = classifier.evaluate_model(val_dataset)
    
    # Save model
    classifier.save_model()
    
    print(f"\nTraining completed! Final accuracy: {accuracy:.4f}")
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('results/training_history.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    main()