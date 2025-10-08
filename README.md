# Ship Image Classification Model

A CNN-based machine learning model for classifying ship images using the Kaggle Ships Dataset.

## Features

- **CNN Architecture**: Custom convolutional neural network optimized for ship classification
- **Data Augmentation**: Rotation, shifting, and flipping for better generalization
- **Automated Pipeline**: Complete training and evaluation workflow
- **Kaggle Integration**: Direct dataset download from Kaggle API
- **Performance Metrics**: Confusion matrix, classification report, and accuracy metrics

## Quick Start

1. **Setup Environment**:
   ```bash
   python setup.py
   ```

2. **Train Model**:
   ```bash
   python ship_classifier.py
   ```

3. **Run Inference**:
   ```bash
   python inference.py
   ```

## Model Architecture

- Input: 224x224 RGB images
- 4 Convolutional layers with MaxPooling
- Dropout for regularization
- Dense layers for classification
- Softmax output for multi-class prediction

## Dataset

Uses the [Ships Dataset](https://www.kaggle.com/datasets/vinayakshanawad/ships-dataset) from Kaggle containing various ship types and maritime vessels.

## Performance

The model achieves high accuracy on ship classification with:
- Data augmentation for robustness
- Early stopping to prevent overfitting
- Learning rate scheduling for optimal convergence

## Files

- `ship_classifier.py`: Main training script
- `inference.py`: Prediction script for new images
- `setup.py`: Environment setup
- `requirements.txt`: Python dependencies

## Integration with Maritime ACPS

This model can be integrated into the Maritime Autonomous Convoy Protection System for:
- Real-time ship detection and classification
- Threat assessment based on vessel type
- Maritime situational awareness
- Automated convoy protection decisions