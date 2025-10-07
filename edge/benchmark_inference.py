#!/usr/bin/env python3
"""
Benchmark script for YOLO inference performance
"""

import asyncio
import time
import psutil
import os
from yolo_infer import YOLOInference, DummyYOLO
from camera_adapter import CameraAdapter

async def benchmark_yolo():
    """Benchmark YOLO inference performance"""
    print("=== YOLO Inference Benchmark ===\n")
    
    # Test both minimal and full modes
    configs = [
        {"use_minimal": True, "name": "YOLOv8n (CPU)"},
        {"use_minimal": False, "name": "YOLOv8s (GPU if available)"}
    ]
    
    for config in configs:
        print(f"Testing {config['name']}...")
        
        yolo = YOLOInference(use_minimal=config["use_minimal"])
        await yolo.initialize()
        
        # Run benchmark
        results = await yolo.benchmark_inference(num_frames=50)
        
        print(f"Results for {config['name']}:")
        print(f"  Average inference time: {results['avg_inference_time']:.4f}s")
        print(f"  Min inference time: {results['min_inference_time']:.4f}s")
        print(f"  Max inference time: {results['max_inference_time']:.4f}s")
        print(f"  Average FPS: {results['fps']:.2f}")
        print(f"  Device: {results['device']}")
        print(f"  Model: {results['model']}")
        print()

async def benchmark_camera():
    """Benchmark camera frame generation"""
    print("=== Camera Benchmark ===\n")
    
    camera = CameraAdapter("simulator", use_minimal=True)
    await camera.initialize()
    
    num_frames = 100
    times = []
    
    print(f"Generating {num_frames} synthetic frames...")
    
    for i in range(num_frames):
        start_time = time.time()
        frame = await camera.capture_frame()
        end_time = time.time()
        times.append(end_time - start_time)
        
        if (i + 1) % 20 == 0:
            print(f"Generated {i + 1}/{num_frames} frames")
    
    avg_time = sum(times) / len(times)
    fps = 1.0 / avg_time
    
    print(f"\nCamera Performance:")
    print(f"  Average frame generation time: {avg_time:.4f}s")
    print(f"  Average FPS: {fps:.2f}")
    print()

def get_system_info():
    """Get system information"""
    print("=== System Information ===\n")
    
    # CPU info
    print(f"CPU: {psutil.cpu_count()} cores")
    print(f"CPU Usage: {psutil.cpu_percent()}%")
    
    # Memory info
    memory = psutil.virtual_memory()
    print(f"Memory: {memory.total // (1024**3)}GB total, {memory.available // (1024**3)}GB available")
    
    # GPU info (if available)
    try:
        import torch
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"CUDA Version: {torch.version.cuda}")
        else:
            print("GPU: Not available")
    except ImportError:
        print("PyTorch not available")
    
    print()

async def end_to_end_benchmark():
    """Benchmark complete edge agent pipeline"""
    print("=== End-to-End Pipeline Benchmark ===\n")
    
    camera = CameraAdapter("simulator", use_minimal=True)
    yolo = YOLOInference(use_minimal=True)
    
    await camera.initialize()
    await yolo.initialize()
    
    num_iterations = 30
    total_times = []
    detection_counts = []
    
    print(f"Running {num_iterations} complete pipeline iterations...")
    
    for i in range(num_iterations):
        start_time = time.time()
        
        # Capture frame
        frame = await camera.capture_frame()
        
        # Run detection
        detections = await yolo.detect(frame)
        
        end_time = time.time()
        
        total_times.append(end_time - start_time)
        detection_counts.append(len(detections))
        
        if (i + 1) % 10 == 0:
            print(f"Completed {i + 1}/{num_iterations} iterations")
    
    avg_total_time = sum(total_times) / len(total_times)
    avg_detections = sum(detection_counts) / len(detection_counts)
    pipeline_fps = 1.0 / avg_total_time
    
    print(f"\nEnd-to-End Pipeline Performance:")
    print(f"  Average total time per frame: {avg_total_time:.4f}s")
    print(f"  Pipeline FPS: {pipeline_fps:.2f}")
    print(f"  Average detections per frame: {avg_detections:.1f}")
    print()

async def memory_usage_test():
    """Test memory usage during operation"""
    print("=== Memory Usage Test ===\n")
    
    process = psutil.Process(os.getpid())
    initial_memory = process.memory_info().rss / (1024 * 1024)  # MB
    
    print(f"Initial memory usage: {initial_memory:.1f} MB")
    
    # Initialize components
    camera = CameraAdapter("simulator", use_minimal=True)
    yolo = YOLOInference(use_minimal=True)
    
    await camera.initialize()
    await yolo.initialize()
    
    post_init_memory = process.memory_info().rss / (1024 * 1024)
    print(f"Memory after initialization: {post_init_memory:.1f} MB")
    print(f"Initialization overhead: {post_init_memory - initial_memory:.1f} MB")
    
    # Run processing loop
    for i in range(50):
        frame = await camera.capture_frame()
        detections = await yolo.detect(frame)
        
        if (i + 1) % 10 == 0:
            current_memory = process.memory_info().rss / (1024 * 1024)
            print(f"Memory after {i + 1} frames: {current_memory:.1f} MB")
    
    final_memory = process.memory_info().rss / (1024 * 1024)
    print(f"Final memory usage: {final_memory:.1f} MB")
    print(f"Total memory increase: {final_memory - initial_memory:.1f} MB")
    print()

async def main():
    """Main benchmark function"""
    print("ACPS Edge Agent Performance Benchmark")
    print("=" * 50)
    print()
    
    get_system_info()
    
    await benchmark_camera()
    await benchmark_yolo()
    await end_to_end_benchmark()
    await memory_usage_test()
    
    print("Benchmark completed!")

if __name__ == "__main__":
    asyncio.run(main())