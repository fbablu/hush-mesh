import random
import time
from typing import List, Dict, Any
import math

class ConvoySimulator:
    def __init__(self):
        self.vehicles = []
        self.drones = []
        self.threats = []
        self.obstacles = []
        self.convoy_speed = 5.0
        self.time_step = 0
        self.map_width = 800
        self.map_height = 600
        
        # Initialize convoy
        self._initialize_convoy()
        
    def _initialize_convoy(self):
        # 3 vehicles moving forward
        self.vehicles = [
            {"id": "V1", "x": 100, "y": 250, "type": "lead", "health": 100},
            {"id": "V2", "x": 70, "y": 280, "type": "main", "health": 100},
            {"id": "V3", "x": 40, "y": 310, "type": "rear", "health": 100}
        ]
        
        # 6 drones in protective formation
        self.drones = [
            {"id": "D1", "x": 150, "y": 200, "role": "scout", "health": 100, "active": True},
            {"id": "D2", "x": 150, "y": 350, "role": "scout", "health": 100, "active": True},
            {"id": "D3", "x": 80, "y": 200, "role": "escort", "health": 100, "active": True},
            {"id": "D4", "x": 80, "y": 350, "role": "escort", "health": 100, "active": True},
            {"id": "D5", "x": 20, "y": 250, "role": "overwatch", "health": 100, "active": True},
            {"id": "D6", "x": 120, "y": 280, "role": "reserve", "health": 100, "active": True}
        ]

    def update(self) -> Dict[str, Any]:
        self.time_step += 1
        
        # Calculate optimal path based on threats/obstacles ahead
        lead_vehicle = self.vehicles[0] if self.vehicles else None
        if lead_vehicle:
            optimal_direction = self._calculate_optimal_direction(lead_vehicle)
            
        # Move convoy forward continuously
        for i, vehicle in enumerate(self.vehicles):
            if lead_vehicle:
                if i == 0:  # Lead vehicle
                    vehicle["x"] += optimal_direction["dx"] * self.convoy_speed
                    vehicle["y"] += optimal_direction["dy"] * self.convoy_speed
                else:  # Follow previous vehicle
                    prev_vehicle = self.vehicles[i-1]
                    dx = prev_vehicle["x"] - vehicle["x"]
                    dy = prev_vehicle["y"] - vehicle["y"]
                    distance = (dx**2 + dy**2)**0.5
                    if distance > 40:  # Maintain spacing
                        vehicle["x"] += (dx/distance) * self.convoy_speed
                        vehicle["y"] += (dy/distance) * self.convoy_speed
            
            # Wrap around when reaching edge
            if vehicle["x"] > self.map_width:
                vehicle["x"] = -50
            if vehicle["y"] < 0:
                vehicle["y"] = self.map_height
            elif vehicle["y"] > self.map_height:
                vehicle["y"] = 0
                
        # Update drone positions to maintain formation
        self._update_drone_formation()
        
        return self.get_state()

    def _calculate_optimal_direction(self, lead_vehicle):
        """Calculate optimal movement direction avoiding threats/obstacles"""
        # Check what drones detect
        detections = self._get_drone_detections()
        
        # Default direction
        dx, dy = 1.0, 0.0
        
        # Adjust based on detections
        for detection in detections:
            if detection["type"] == "threat":
                # Strong avoidance for threats
                if detection["relative_y"] < 0:  # Threat above
                    dy += 0.8  # Go down
                else:  # Threat below
                    dy -= 0.8  # Go up
            elif detection["type"] == "obstacle":
                # Moderate avoidance for obstacles
                if detection["relative_y"] < 0:  # Obstacle above
                    dy += 0.5  # Go down
                else:  # Obstacle below
                    dy -= 0.5  # Go up
        
        # Normalize and limit direction change
        dy = max(-0.8, min(0.8, dy))
        return {"dx": dx, "dy": dy}
    
    def _get_drone_detections(self):
        """Get what drones detect ahead of convoy"""
        detections = []
        convoy_center_x = sum(v["x"] for v in self.vehicles) / len(self.vehicles)
        convoy_center_y = sum(v["y"] for v in self.vehicles) / len(self.vehicles)
        
        # Reset all drone detection states
        for drone in self.drones:
            drone["detecting"] = False
        
        # Check each active drone's detection range
        for drone in self.drones:
            if not drone["active"]:
                continue
                
            detection_range = 120
            
            # Check for threats in drone's detection cone
            for threat in self.threats:
                dx = threat["x"] - drone["x"]
                dy = threat["y"] - drone["y"]
                distance = (dx**2 + dy**2)**0.5
                
                if distance < detection_range and dx > 0:  # Ahead of convoy
                    drone["detecting"] = True
                    detections.append({
                        "type": "threat",
                        "drone_id": drone["id"],
                        "distance": distance,
                        "relative_y": dy
                    })
            
            # Check for obstacles in drone's detection cone
            for obstacle in self.obstacles:
                dx = obstacle["x"] - drone["x"]
                dy = obstacle["y"] - drone["y"]
                distance = (dx**2 + dy**2)**0.5
                
                if distance < detection_range and dx > 0:  # Ahead of convoy
                    drone["detecting"] = True
                    detections.append({
                        "type": "obstacle",
                        "drone_id": drone["id"],
                        "distance": distance,
                        "relative_y": dy
                    })
        
        return detections
    
    def _update_drone_formation(self):
        convoy_center_x = sum(v["x"] for v in self.vehicles) / len(self.vehicles)
        convoy_center_y = sum(v["y"] for v in self.vehicles) / len(self.vehicles)
        
        active_drones = [d for d in self.drones if d["active"]]
        
        # Evenly distribute active drones in circular formation
        if len(active_drones) > 0:
            import math
            radius = 80
            angle_step = 2 * math.pi / len(active_drones)
            
            for i, drone in enumerate(active_drones):
                angle = i * angle_step
                target_x = convoy_center_x + radius * math.cos(angle)
                target_y = convoy_center_y + radius * math.sin(angle)
                
                # Move towards target faster
                dx = target_x - drone["x"]
                dy = target_y - drone["y"]
                
                drone["x"] += min(8, max(-8, dx * 0.2))
                drone["y"] += min(8, max(-8, dy * 0.2))
                
                # Keep within map bounds
                drone["x"] = max(10, min(drone["x"], self.map_width - 10))
                drone["y"] = max(10, min(drone["y"], self.map_height - 10))

    def add_obstacle(self, x: float, y: float, obstacle_type: str = "physical"):
        """Add obstacle at specified coordinates"""
        self.obstacles.append({
            "id": f"OBS_{len(self.obstacles)}",
            "x": x,
            "y": y,
            "type": obstacle_type
        })
        
    def add_threat(self, x: float, y: float, threat_type: str = "ambush"):
        """Add threat at specified coordinates"""
        self.threats.append({
            "id": f"THR_{len(self.threats)}",
            "x": x,
            "y": y,
            "type": threat_type,
            "severity": random.uniform(0.7, 0.9)
        })
        
    def clear_all(self):
        """Clear all threats and obstacles"""
        self.threats = []
        self.obstacles = []
        
    def eliminate_drone(self, drone_id: str):
        """Eliminate a drone and trigger formation reform"""
        for drone in self.drones:
            if drone["id"] == drone_id:
                drone["active"] = False
                drone["health"] = 0
                break
                
    def reset_mission(self):
        self.threats = []
        self.obstacles = []
        self.time_step = 0
        self.convoy_speed = 5.0  # Faster movement
        
        # Reset all drones to active
        for drone in self.drones:
            drone["active"] = True
            drone["health"] = 100
            
        # Reset vehicle positions
        self._initialize_convoy()

    def get_state(self) -> Dict[str, Any]:
        detections = self._get_drone_detections()
        return {
            "vehicles": self.vehicles,
            "drones": self.drones,
            "threats": self.threats,
            "obstacles": self.obstacles,
            "detections": detections,
            "time_step": self.time_step,
            "map_width": self.map_width,
            "map_height": self.map_height
        }