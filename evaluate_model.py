#!/usr/bin/env python3
"""
Evaluate Ship Classification Model Performance
"""

import numpy as np
import tensorflow as tf
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report

def load_ship_data():
    """Load ship images and labels from dataset"""
    data_dir = Path("./data/Ships dataset")
    
    # Load training data
    train_images = []
    train_labels = []
    
    train_img_dir = data_dir / "train" / "images"
    train_lbl_dir = data_dir / "train" / "labels"
    
    # Load subset for quick training
    img_files = list(train_img_dir.glob("*.jpg"))[:500]  # Use 500 images for demo
    
    for img_file in img_files:
        label_file = train_lbl_dir / f"{img_file.stem}.txt"
        if label_file.exists():
            with open(label_file, 'r') as f:
                line = f.readline().strip()
                if line:
                    class_id = int(line.split()[0])
                    if class_id < 8:  # Valid class
                        train_images.append(str(img_file))
                        train_labels.append(class_id)
    
    # Load test data
    test_images = []
    test_labels = []
    
    test_img_dir = data_dir / "test" / "images"
    test_lbl_dir = data_dir / "test" / "labels"
    
    test_files = list(test_img_dir.glob("*.jpg"))[:100]  # Use 100 test images
    
    for img_file in test_files:
        label_file = test_lbl_dir / f"{img_file.stem}.txt"
        if label_file.exists():
            with open(label_file, 'r') as f:
                line = f.readline().strip()
                if line:
                    class_id = int(line.split()[0])
                    if class_id < 8:  # Valid class
                        test_images.append(str(img_file))
                        test_labels.append(class_id)
    
    return train_images, train_labels, test_images, test_labels

def preprocess_images(image_paths, labels, img_size=(128, 128)):
    """Preprocess images for training"""
    X = []
    y = []
    
    for img_path, label in zip(image_paths, labels):
        try:
            img = Image.open(img_path).convert('RGB')
            img = img.resize(img_size)
            img_array = np.array(img) / 255.0
            X.append(img_array)
            y.append(label)
        except Exception as e:
            continue
    
    return np.array(X), np.array(y)

def build_ship_classifier():
    """Build CNN for ship classification"""
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(128, 128, 3)),
        tf.keras.layers.Conv2D(32, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(64, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(128, 3, activation='relu'),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(8, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def evaluate_performance():
    """Train and evaluate ship classification model"""
    print("=== Ship Classification Model Evaluation ===")
    
    # Load data
    print("Loading ship dataset...")
    train_imgs, train_lbls, test_imgs, test_lbls = load_ship_data()
    
    print(f"Training images: {len(train_imgs)}")
    print(f"Test images: {len(test_imgs)}")
    
    if len(train_imgs) == 0 or len(test_imgs) == 0:
        print("Insufficient data for evaluation!")
        return
    
    # Preprocess data
    print("Preprocessing images...")
    X_train, y_train = preprocess_images(train_imgs, train_lbls)
    X_test, y_test = preprocess_images(test_imgs, test_lbls)
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Test data shape: {X_test.shape}")
    
    # Build and train model
    print("Building model...")
    model = build_ship_classifier()
    
    print("Training model...")
    history = model.fit(
        X_train, y_train,
        epochs=10,
        batch_size=32,
        validation_split=0.2,
        verbose=1
    )
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    
    # Get predictions
    y_pred = model.predict(X_test, verbose=0)
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred_classes)
    
    print(f"\nTest Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Loss: {test_loss:.4f}")
    
    # Class names
    class_names = ['cargo', 'military', 'carrier', 'cruise', 'tankers', 'trawlers', 'tugboat', 'yacht']
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_classes, target_names=class_names, zero_division=0))
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training')
    plt.plot(history.history['val_accuracy'], label='Validation')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training')
    plt.plot(history.history['val_loss'], label='Validation')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('results/model_evaluation.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Sample predictions
    print("\nSample Predictions:")
    for i in range(min(5, len(X_test))):
        pred_class = y_pred_classes[i]
        true_class = y_test[i]
        confidence = np.max(y_pred[i])
        
        print(f"Sample {i+1}: True={class_names[true_class]}, "
              f"Predicted={class_names[pred_class]}, "
              f"Confidence={confidence:.3f}")
    
    return accuracy, model

if __name__ == "__main__":
    accuracy, model = evaluate_performance()
    print(f"\nFinal Model Accuracy: {accuracy:.4f}")