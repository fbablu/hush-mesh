import asyncio
import cv2
import numpy as np
import torch
import random
from typing import List, Dict, Any, Optional
from ultralytics import YOLO
import os

class YOLOInference:
    def __init__(self, use_minimal: bool = True):
        self.use_minimal = use_minimal
        self.model = None
        self.device = "cpu"  # Default to CPU for compatibility
        self.model_path = None
        self.confidence_threshold = 0.5
        self.nms_threshold = 0.4
        
        # Class names for COCO dataset
        self.class_names = [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
            'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
            'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
            'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
            'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
            'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
            'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
            'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
            'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
            'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
            'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
            'toothbrush'
        ]
        
    async def initialize(self):
        """Initialize YOLO model"""
        try:
            if self.use_minimal:
                # Use smallest YOLOv8 model for CPU compatibility
                self.model_path = "yolov8n.pt"
                print("Loading YOLOv8n (nano) model for CPU inference")
            else:
                # Use larger model if GPU available
                if torch.cuda.is_available():
                    self.device = "cuda"
                    self.model_path = "yolov8s.pt"
                    print("Loading YOLOv8s model for GPU inference")
                else:
                    self.model_path = "yolov8n.pt"
                    print("GPU not available, using YOLOv8n model")
                    
            # Load model
            self.model = YOLO(self.model_path)
            self.model.to(self.device)
            
            print(f"YOLO model initialized on {self.device}")
            
        except Exception as e:
            print(f"Error initializing YOLO model: {e}")
            print("Falling back to synthetic detection mode")
            self.model = None
            
    async def detect(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """Run YOLO detection on frame"""
        if self.model is None:
            # Fallback to synthetic detections
            return self.generate_synthetic_detections(frame)
            
        try:
            # Run inference
            results = self.model(frame, conf=self.confidence_threshold, iou=self.nms_threshold, verbose=False)
            
            detections = []
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        # Extract detection data
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        confidence = float(box.conf[0].cpu().numpy())
                        class_id = int(box.cls[0].cpu().numpy())
                        
                        if class_id < len(self.class_names):
                            class_name = self.class_names[class_id]
                            
                            detection = {
                                "bbox": [int(x1), int(y1), int(x2), int(y2)],
                                "class": class_name,
                                "confidence": confidence,
                                "priority": self.calculate_priority(class_name, confidence)
                            }
                            detections.append(detection)
                            
            return detections
            
        except Exception as e:
            print(f"Error during YOLO inference: {e}")
            return self.generate_synthetic_detections(frame)
            
    def generate_synthetic_detections(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """Generate synthetic detections for demo purposes"""
        detections = []
        
        # Randomly generate 0-3 detections
        num_detections = random.randint(0, 3)
        
        for _ in range(num_detections):
            # Random bounding box
            x1 = random.randint(0, frame.shape[1] - 100)
            y1 = random.randint(0, frame.shape[0] - 100)
            x2 = x1 + random.randint(50, 150)
            y2 = y1 + random.randint(50, 150)
            
            # Ensure box is within frame
            x2 = min(x2, frame.shape[1])
            y2 = min(y2, frame.shape[0])
            
            # Random class (focus on relevant classes for convoy protection)
            relevant_classes = ["person", "car", "truck", "motorcycle", "bicycle"]
            class_name = random.choice(relevant_classes)
            confidence = random.uniform(0.5, 0.95)
            
            detection = {
                "bbox": [x1, y1, x2, y2],
                "class": class_name,
                "confidence": confidence,
                "priority": self.calculate_priority(class_name, confidence)
            }
            detections.append(detection)
            
        return detections
        
    def calculate_priority(self, class_name: str, confidence: float) -> int:
        """Calculate detection priority based on class and confidence"""
        # Priority scoring for convoy protection
        class_priorities = {
            "person": 8,
            "car": 6,
            "truck": 7,
            "motorcycle": 7,
            "bicycle": 5,
            "bus": 6,
            "airplane": 4,
            "boat": 3
        }
        
        base_priority = class_priorities.get(class_name, 3)
        confidence_bonus = int(confidence * 3)  # 0-3 bonus based on confidence
        
        return min(10, base_priority + confidence_bonus)
        
    def draw_detections(self, frame: np.ndarray, detections: List[Dict[str, Any]]) -> np.ndarray:
        """Draw detection bounding boxes on frame"""
        annotated_frame = frame.copy()
        
        for detection in detections:
            bbox = detection["bbox"]
            class_name = detection["class"]
            confidence = detection["confidence"]
            priority = detection["priority"]
            
            # Color based on priority
            if priority >= 8:
                color = (0, 0, 255)  # Red for high priority
            elif priority >= 6:
                color = (0, 165, 255)  # Orange for medium priority
            else:
                color = (0, 255, 0)  # Green for low priority
                
            # Draw bounding box
            cv2.rectangle(annotated_frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
            
            # Draw label
            label = f"{class_name}: {confidence:.2f} (P{priority})"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(annotated_frame, (bbox[0], bbox[1] - label_size[1] - 10), 
                         (bbox[0] + label_size[0], bbox[1]), color, -1)
            cv2.putText(annotated_frame, label, (bbox[0], bbox[1] - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                       
        return annotated_frame
        
    async def benchmark_inference(self, num_frames: int = 100) -> Dict[str, float]:
        """Benchmark inference performance"""
        if self.model is None:
            return {"error": "Model not initialized"}
            
        # Generate test frame
        test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        import time
        times = []
        
        print(f"Running benchmark with {num_frames} frames...")
        
        for i in range(num_frames):
            start_time = time.time()
            await self.detect(test_frame)
            end_time = time.time()
            times.append(end_time - start_time)
            
            if (i + 1) % 10 == 0:
                print(f"Processed {i + 1}/{num_frames} frames")
                
        avg_time = sum(times) / len(times)
        min_time = min(times)
        max_time = max(times)
        fps = 1.0 / avg_time
        
        return {
            "avg_inference_time": avg_time,
            "min_inference_time": min_time,
            "max_inference_time": max_time,
            "fps": fps,
            "total_frames": num_frames,
            "device": self.device,
            "model": self.model_path
        }

class DummyYOLO(YOLOInference):
    """Dummy YOLO implementation for machines without PyTorch"""
    
    def __init__(self):
        super().__init__(use_minimal=True)
        self.model = "dummy"
        
    async def initialize(self):
        print("Initialized dummy YOLO detector (no PyTorch required)")
        
    async def detect(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """Always return synthetic detections"""
        await asyncio.sleep(0.05)  # Simulate inference time
        return self.generate_synthetic_detections(frame)