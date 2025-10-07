import asyncio
import cv2
import numpy as np
import random
from typing import Optional
from PIL import Image, ImageDraw, ImageFont
import io
import base64

class CameraAdapter:
    def __init__(self, source: str = "simulator", use_minimal: bool = True):
        self.source = source
        self.use_minimal = use_minimal
        self.cap = None
        self.frame_width = 640
        self.frame_height = 480
        
        # Synthetic object templates for simulation
        self.synthetic_objects = {
            "person": {"color": (255, 0, 0), "size": (30, 60)},
            "car": {"color": (0, 255, 0), "size": (80, 40)},
            "truck": {"color": (0, 0, 255), "size": (120, 50)},
            "motorcycle": {"color": (255, 255, 0), "size": (40, 25)}
        }
        
    async def initialize(self):
        """Initialize camera source"""
        if self.source == "simulator":
            print("Initialized synthetic camera simulator")
        elif self.source == "webcam":
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                print("Warning: Could not open webcam, falling back to simulator")
                self.source = "simulator"
        elif self.source.startswith("airsim"):
            # TODO: Initialize AirSim camera connection
            print("AirSim camera not implemented, using simulator")
            self.source = "simulator"
        else:
            print(f"Unknown camera source: {self.source}, using simulator")
            self.source = "simulator"
            
    async def capture_frame(self) -> Optional[np.ndarray]:
        """Capture a frame from the camera source"""
        if self.source == "simulator":
            return self.generate_synthetic_frame()
        elif self.source == "webcam" and self.cap:
            ret, frame = self.cap.read()
            if ret:
                return cv2.resize(frame, (self.frame_width, self.frame_height))
        
        return None
        
    def generate_synthetic_frame(self) -> np.ndarray:
        """Generate synthetic frame with random objects"""
        # Create base frame (road/terrain background)
        frame = np.zeros((self.frame_height, self.frame_width, 3), dtype=np.uint8)
        
        # Add road texture
        road_color = (60, 60, 60)  # Dark gray
        frame[:, :] = road_color
        
        # Add road markings
        cv2.line(frame, (self.frame_width//2, 0), (self.frame_width//2, self.frame_height), (255, 255, 255), 2)
        
        # Add random terrain features
        for _ in range(random.randint(2, 5)):
            x = random.randint(0, self.frame_width)
            y = random.randint(0, self.frame_height)
            radius = random.randint(10, 30)
            color = (random.randint(40, 80), random.randint(60, 100), random.randint(40, 80))
            cv2.circle(frame, (x, y), radius, color, -1)
            
        # Add synthetic objects randomly
        if random.random() < 0.3:  # 30% chance of object in frame
            self.add_synthetic_objects(frame)
            
        # Add noise for realism
        noise = np.random.normal(0, 10, frame.shape).astype(np.uint8)
        frame = cv2.add(frame, noise)
        
        return frame
        
    def add_synthetic_objects(self, frame: np.ndarray):
        """Add synthetic objects to frame"""
        num_objects = random.randint(1, 3)
        
        for _ in range(num_objects):
            obj_type = random.choice(list(self.synthetic_objects.keys()))
            obj_info = self.synthetic_objects[obj_type]
            
            # Random position
            x = random.randint(50, self.frame_width - 100)
            y = random.randint(50, self.frame_height - 100)
            
            # Object size with some variation
            width = obj_info["size"][0] + random.randint(-10, 10)
            height = obj_info["size"][1] + random.randint(-10, 10)
            
            # Draw object as rectangle (simplified)
            color = obj_info["color"]
            cv2.rectangle(frame, (x, y), (x + width, y + height), color, -1)
            
            # Add some detail
            if obj_type == "person":
                # Add head
                cv2.circle(frame, (x + width//2, y + 10), 8, color, -1)
            elif obj_type in ["car", "truck"]:
                # Add windows
                window_color = (100, 150, 200)
                cv2.rectangle(frame, (x + 5, y + 5), (x + width - 5, y + height//2), window_color, -1)
                
    def frame_to_base64(self, frame: np.ndarray) -> str:
        """Convert frame to base64 string for transmission"""
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        return base64.b64encode(buffer).decode('utf-8')
        
    async def cleanup(self):
        """Cleanup camera resources"""
        if self.cap:
            self.cap.release()
            
class AirSimCameraAdapter(CameraAdapter):
    """AirSim-specific camera adapter (placeholder for future implementation)"""
    
    def __init__(self, vehicle_name: str = "Drone1"):
        super().__init__("airsim", False)
        self.vehicle_name = vehicle_name
        self.airsim_client = None
        
    async def initialize(self):
        """Initialize AirSim connection"""
        try:
            # TODO: Import and initialize AirSim client
            # import airsim
            # self.airsim_client = airsim.MultirotorClient()
            # self.airsim_client.confirmConnection()
            print("AirSim camera adapter initialized (placeholder)")
        except ImportError:
            print("AirSim not available, falling back to simulator")
            self.source = "simulator"
            await super().initialize()
            
    async def capture_frame(self) -> Optional[np.ndarray]:
        """Capture frame from AirSim camera"""
        if self.airsim_client:
            try:
                # TODO: Implement AirSim frame capture
                # response = self.airsim_client.simGetImage("0", airsim.ImageType.Scene)
                # img1d = np.fromstring(response, np.uint8)
                # frame = cv2.imdecode(img1d, cv2.IMREAD_COLOR)
                # return frame
                pass
            except Exception as e:
                print(f"AirSim capture error: {e}")
                
        # Fallback to synthetic frame
        return self.generate_synthetic_frame()