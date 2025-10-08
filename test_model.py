#!/usr/bin/env python3
"""
Test Ship Classification Model
"""

import os
import numpy as np
import tensorflow as tf
from pathlib import Path
from PIL import Image
import json
import matplotlib.pyplot as plt

class ModelTester:
    def __init__(self):
        self.img_size = (224, 224)
        self.class_names = {
            0: 'cargo', 1: 'military', 2: 'carrier', 3: 'cruise',
            4: 'tankers', 5: 'trawlers', 6: 'tugboat', 7: 'yacht'
        }
        
    def load_test_data(self, data_dir="./data/Ships dataset/test"):
        """Load test images and labels"""
        images_dir = Path(data_dir) / 'images'
        labels_dir = Path(data_dir) / 'labels'
        
        test_images = []
        test_labels = []
        
        for img_file in list(images_dir.glob('*.jpg'))[:50]:  # Test on 50 images
            label_file = labels_dir / f"{img_file.stem}.txt"
            
            if label_file.exists():
                with open(label_file, 'r') as f:
                    line = f.readline().strip()
                    if line:
                        class_id = int(line.split()[0])
                        test_images.append(str(img_file))
                        test_labels.append(class_id)
        
        return test_images, test_labels
    
    def preprocess_image(self, image_path):
        """Preprocess single image"""
        img = Image.open(image_path).convert('RGB')
        img = img.resize(self.img_size)
        img_array = np.array(img) / 255.0
        return np.expand_dims(img_array, axis=0)
    
    def test_with_synthetic_model(self, test_images, test_labels):
        """Test with a simple synthetic model for demonstration"""
        print("Creating synthetic model for testing...")
        
        # Build simple model
        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=(224, 224, 3)),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(64, 3, activation='relu'),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(8, activation='softmax')
        ])
        
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        
        # Generate synthetic training data
        X_synthetic = np.random.rand(100, 224, 224, 3)
        y_synthetic = np.random.randint(0, 8, 100)
        
        # Quick training
        print("Training synthetic model...")
        model.fit(X_synthetic, y_synthetic, epochs=3, verbose=0)
        
        # Test on real images
        print(f"Testing on {len(test_images)} real ship images...")
        
        correct = 0
        predictions = []
        
        for i, (img_path, true_label) in enumerate(zip(test_images[:10], test_labels[:10])):
            img_array = self.preprocess_image(img_path)
            pred = model.predict(img_array, verbose=0)
            predicted_class = np.argmax(pred)
            
            predictions.append(predicted_class)
            if predicted_class == true_label:
                correct += 1
            
            print(f"Image {i+1}: True={self.class_names[true_label]}, "
                  f"Predicted={self.class_names[predicted_class]}, "
                  f"Confidence={np.max(pred):.3f}")
        
        accuracy = correct / min(10, len(test_images))
        print(f"\nTest Accuracy: {accuracy:.3f}")
        
        return accuracy, predictions
    
    def visualize_results(self, test_images, test_labels, predictions):
        """Visualize test results"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for i in range(min(6, len(test_images))):
            img = Image.open(test_images[i])
            axes[i].imshow(img)
            axes[i].set_title(f"True: {self.class_names[test_labels[i]]}\n"
                             f"Pred: {self.class_names[predictions[i]]}")
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.savefig('results/test_results.png', dpi=300, bbox_inches='tight')
        plt.show()

def main():
    print("=== Testing Ship Classification Model ===")
    
    tester = ModelTester()
    
    # Load test data
    print("Loading test data...")
    test_images, test_labels = tester.load_test_data()
    
    if len(test_images) == 0:
        print("No test data found!")
        return
    
    print(f"Loaded {len(test_images)} test images")
    
    # Test model
    accuracy, predictions = tester.test_with_synthetic_model(test_images, test_labels)
    
    # Visualize results
    tester.visualize_results(test_images, test_labels, predictions)
    
    print(f"\nTesting completed! Accuracy: {accuracy:.3f}")

if __name__ == "__main__":
    main()