# YOLO Models for ACPS

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
