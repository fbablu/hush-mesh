#!/usr/bin/env python3
"""
Script to download YOLO models for edge inference
"""

import os
import sys
from pathlib import Path

def download_yolo_models():
    """Download YOLO models"""
    print("Downloading YOLO models...")
    
    try:
        from ultralytics import YOLO
        
        models_dir = Path(__file__).parent.parent / "models"
        models_dir.mkdir(exist_ok=True)
        
        # Download YOLOv8 models
        models = [
            ("yolov8n.pt", "YOLOv8 Nano (CPU-friendly)"),
            ("yolov8s.pt", "YOLOv8 Small (GPU recommended)")
        ]
        
        for model_name, description in models:
            print(f"Downloading {description}...")
            try:
                model = YOLO(model_name)
                print(f"✓ {model_name} downloaded successfully")
            except Exception as e:
                print(f"✗ Failed to download {model_name}: {e}")
                
        print("\nModel download completed!")
        print("Models are cached in ~/.ultralytics/")
        
    except ImportError:
        print("Error: ultralytics package not installed")
        print("Install with: pip install ultralytics")
        sys.exit(1)
    except Exception as e:
        print(f"Error downloading models: {e}")
        sys.exit(1)

def create_model_info():
    """Create model information file"""
    models_dir = Path(__file__).parent.parent / "models"
    models_dir.mkdir(exist_ok=True)
    
    info_content = """# YOLO Models for ACPS

## Available Models

### YOLOv8n (Nano)
- **File**: yolov8n.pt
- **Size**: ~6MB
- **Speed**: Fast (CPU-friendly)
- **Accuracy**: Good for real-time detection
- **Use case**: Edge devices, CPU inference

### YOLOv8s (Small)  
- **File**: yolov8s.pt
- **Size**: ~22MB
- **Speed**: Medium (GPU recommended)
- **Accuracy**: Better than nano
- **Use case**: GPU-enabled edge devices

## Usage

The models are automatically downloaded when first used by the YOLO inference module.

For manual download:
```bash
python scripts/download_models.py
```

## Custom Models

To use custom trained models:

1. Place your .pt file in the models/ directory
2. Update the model path in edge/yolo_infer.py
3. Ensure the model is compatible with Ultralytics YOLO format

## Model Performance

Benchmark your models with:
```bash
python edge/benchmark_inference.py
```

## AWS Integration Notes

For production deployment with AWS:

- Store models in S3 for centralized distribution
- Use SageMaker for model training and optimization
- Consider AWS Panorama for optimized edge inference
- Use IoT Greengrass for model deployment and updates
"""
    
    info_file = models_dir / "README.md"
    with open(info_file, 'w') as f:
        f.write(info_content)
        
    print(f"Created model info file: {info_file}")

def main():
    print("ACPS Model Download Script")
    print("=" * 30)
    
    create_model_info()
    download_yolo_models()

if __name__ == "__main__":
    main()