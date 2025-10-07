import asyncio
import json
import os
import time
import random
from datetime import datetime
from typing import Optional, Dict, Any
import paho.mqtt.client as mqtt
from camera_adapter import CameraAdapter
from yolo_infer import YOLOInference

class EdgeAgent:
    def __init__(self):
        self.drone_id = os.getenv("DRONE_ID", "drone-01")
        self.vehicle_id = os.getenv("VEHICLE_ID", "veh-01")
        self.mqtt_broker = os.getenv("MQTT_BROKER", "localhost")
        self.mqtt_port = int(os.getenv("MQTT_PORT", "1883"))
        self.use_minimal = os.getenv("USE_MINIMAL", "true").lower() == "true"
        self.camera_source = os.getenv("CAMERA_SOURCE", "simulator")
        
        # Initialize components
        self.camera = CameraAdapter(self.camera_source, self.use_minimal)
        self.yolo = YOLOInference(self.use_minimal)
        
        # MQTT client
        self.mqtt_client = mqtt.Client()
        self.mqtt_client.on_connect = self.on_mqtt_connect
        self.mqtt_client.on_message = self.on_mqtt_message
        
        # Agent state
        self.running = False
        self.current_position = {"lat": 42.3601, "lon": -71.0589, "alt": 50.0}
        self.status = "idle"
        self.commands_queue = []
        
        # Performance metrics
        self.detections_count = 0
        self.inference_times = []
        
    def on_mqtt_connect(self, client, userdata, flags, rc):
        print(f"Edge agent {self.drone_id} connected to MQTT with result code {rc}")
        # Subscribe to commands for this drone
        client.subscribe(f"acps/commands/{self.drone_id}")
        
    def on_mqtt_message(self, client, userdata, msg):
        try:
            command = json.loads(msg.payload.decode())
            self.commands_queue.append(command)
            print(f"Received command: {command}")
        except Exception as e:
            print(f"Error processing command: {e}")
            
    async def start(self):
        """Start the edge agent"""
        try:
            # Connect to MQTT
            self.mqtt_client.connect(self.mqtt_broker, self.mqtt_port, 60)
            self.mqtt_client.loop_start()
            
            # Initialize camera and YOLO
            await self.camera.initialize()
            await self.yolo.initialize()
            
            self.running = True
            print(f"Edge agent {self.drone_id} started")
            
            # Start main processing loop
            await self.main_loop()
            
        except Exception as e:
            print(f"Error starting edge agent: {e}")
            
    async def stop(self):
        """Stop the edge agent"""
        self.running = False
        self.mqtt_client.loop_stop()
        self.mqtt_client.disconnect()
        await self.camera.cleanup()
        
    async def main_loop(self):
        """Main processing loop"""
        frame_count = 0
        
        while self.running:
            try:
                # Process any pending commands
                await self.process_commands()
                
                # Capture frame
                frame = await self.camera.capture_frame()
                if frame is None:
                    await asyncio.sleep(0.1)
                    continue
                    
                # Run inference
                start_time = time.time()
                detections = await self.yolo.detect(frame)
                inference_time = time.time() - start_time
                self.inference_times.append(inference_time)
                
                # Process detections
                for detection in detections:
                    await self.process_detection(detection, frame_count)
                    
                # Send telemetry periodically
                if frame_count % 30 == 0:  # Every 30 frames
                    await self.send_telemetry()
                    
                frame_count += 1
                
                # Simulate realistic frame rate
                await asyncio.sleep(0.1)  # ~10 FPS
                
            except Exception as e:
                print(f"Error in main loop: {e}")
                await asyncio.sleep(1.0)
                
    async def process_commands(self):
        """Process pending commands"""
        while self.commands_queue:
            command = self.commands_queue.pop(0)
            await self.execute_command(command)
            
    async def execute_command(self, command: Dict[str, Any]):
        """Execute a command from the mission planner"""
        command_type = command.get("command")
        
        if command_type == "inspect_aoi":
            await self.inspect_aoi(command)
        elif command_type == "loiter":
            await self.loiter(command)
        elif command_type == "return_to_tether":
            await self.return_to_tether(command)
        elif command_type == "adjust_altitude":
            await self.adjust_altitude(command)
        else:
            print(f"Unknown command: {command_type}")
            
    async def inspect_aoi(self, command: Dict[str, Any]):
        """Inspect Area of Interest"""
        aoi_id = command.get("aoi_id")
        params = command.get("params", {})
        
        print(f"Inspecting AOI {aoi_id}")
        self.status = "inspecting"
        
        # Simulate AOI inspection
        await asyncio.sleep(params.get("loiter_time_s", 10))
        
        # Generate focused detections for AOI
        for _ in range(3):  # Generate 3 detections
            detection = self.generate_synthetic_detection(high_priority=True)
            await self.process_detection(detection, 0)
            await asyncio.sleep(1)
            
        self.status = "following"
        print(f"Completed AOI {aoi_id} inspection")
        
    async def loiter(self, command: Dict[str, Any]):
        """Loiter at current position"""
        duration = command.get("duration_s", 30)
        print(f"Loitering for {duration} seconds")
        self.status = "loitering"
        await asyncio.sleep(duration)
        self.status = "following"
        
    async def return_to_tether(self, command: Dict[str, Any]):
        """Return to tether position"""
        print("Returning to tether position")
        self.status = "returning"
        await asyncio.sleep(5)  # Simulate return time
        self.status = "following"
        
    async def adjust_altitude(self, command: Dict[str, Any]):
        """Adjust flight altitude"""
        new_altitude = command.get("altitude_m", 50)
        print(f"Adjusting altitude to {new_altitude}m")
        self.current_position["alt"] = new_altitude
        
    async def process_detection(self, detection: Dict[str, Any], frame_count: int):
        """Process and publish detection"""
        # Add metadata
        detection.update({
            "drone_id": self.drone_id,
            "vehicle_id": self.vehicle_id,
            "timestamp": datetime.utcnow().isoformat(),
            "geo": self.get_current_geo_position(),
            "frame_id": frame_count
        })
        
        # Apply local safety rules
        if self.should_trigger_immediate_action(detection):
            await self.trigger_immediate_action(detection)
            
        # Publish detection
        topic = f"acps/detections/{self.drone_id}"
        self.mqtt_client.publish(topic, json.dumps(detection))
        
        self.detections_count += 1
        print(f"Published detection: {detection['class']} (confidence: {detection['confidence']:.2f})")
        
    def should_trigger_immediate_action(self, detection: Dict[str, Any]) -> bool:
        """Check if detection requires immediate local action"""
        threat_classes = ["weapon", "explosion", "fire"]
        return (detection.get("class") in threat_classes and 
                detection.get("confidence", 0) > 0.8)
                
    async def trigger_immediate_action(self, detection: Dict[str, Any]):
        """Trigger immediate local safety action"""
        print(f"IMMEDIATE ACTION: {detection['class']} detected with high confidence")
        # In real implementation, this would trigger evasive maneuvers
        
    def get_current_geo_position(self) -> Dict[str, float]:
        """Get current geographic position with some movement simulation"""
        # Simulate drone movement around vehicle
        base_lat = self.current_position["lat"]
        base_lon = self.current_position["lon"]
        
        # Add small random movement
        lat_offset = random.uniform(-0.0002, 0.0002)  # ~20m
        lon_offset = random.uniform(-0.0002, 0.0002)
        
        return {
            "lat": base_lat + lat_offset,
            "lon": base_lon + lon_offset,
            "alt": self.current_position["alt"]
        }
        
    async def send_telemetry(self):
        """Send telemetry data"""
        telemetry = {
            "type": "drone",
            "drone_id": self.drone_id,
            "vehicle_id": self.vehicle_id,
            "timestamp": datetime.utcnow().isoformat(),
            "status": self.status,
            "position": self.get_current_geo_position(),
            "battery_level": random.uniform(70, 100),  # Simulate battery
            "detections_count": self.detections_count,
            "avg_inference_time": sum(self.inference_times[-10:]) / min(10, len(self.inference_times)) if self.inference_times else 0
        }
        
        topic = f"acps/telemetry/{self.drone_id}"
        self.mqtt_client.publish(topic, json.dumps(telemetry))
        
    def generate_synthetic_detection(self, high_priority: bool = False) -> Dict[str, Any]:
        """Generate synthetic detection for demo purposes"""
        classes = ["person", "car", "truck", "motorcycle"]
        if high_priority:
            classes = ["person", "weapon", "fire"]
            
        return {
            "bbox": [
                random.randint(50, 200),
                random.randint(50, 200),
                random.randint(250, 400),
                random.randint(250, 400)
            ],
            "class": random.choice(classes),
            "confidence": random.uniform(0.6, 0.95),
            "priority": random.randint(7, 10) if high_priority else random.randint(3, 7)
        }

async def main():
    """Main entry point"""
    agent = EdgeAgent()
    
    try:
        await agent.start()
    except KeyboardInterrupt:
        print("Shutting down edge agent...")
        await agent.stop()

if __name__ == "__main__":
    asyncio.run(main())