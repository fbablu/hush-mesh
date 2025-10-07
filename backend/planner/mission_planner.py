import numpy as np
import json
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import heapq
from dataclasses import dataclass

@dataclass
class GridCell:
    x: int
    y: int
    base_cost: float
    threat_score: float
    total_cost: float

class MissionPlanner:
    def __init__(self, grid_size: int = 100, cell_size: float = 10.0):
        self.grid_size = grid_size
        self.cell_size = cell_size  # meters per cell
        self.threat_map = np.zeros((grid_size, grid_size))
        self.base_cost_map = np.ones((grid_size, grid_size))
        self.aois: Dict[str, dict] = {}
        self.active_mission = None
        
        # Planner parameters
        self.threat_weight = 2.0
        self.distance_weight = 1.0
        self.weather_weight = 0.5
        
        # Center coordinates (lat, lon) - example coordinates
        self.center_lat = 42.3601
        self.center_lon = -71.0589
        
    def start_mission(self, scenario: str):
        """Initialize mission with scenario parameters"""
        self.active_mission = scenario
        self.threat_map.fill(0)
        print(f"Mission planner started for scenario: {scenario}")
        
    def stop_mission(self):
        """Stop current mission"""
        self.active_mission = None
        self.threat_map.fill(0)
        self.aois.clear()
        
    def add_aoi(self, aoi_id: str, aoi_data: dict):
        """Add Area of Interest for drone inspection"""
        self.aois[aoi_id] = aoi_data
        
        # Convert AOI geometry to grid cells and mark for inspection
        geometry = aoi_data.get("geometry", {})
        if geometry.get("type") == "Polygon":
            self._mark_aoi_cells(geometry["coordinates"][0], aoi_data["priority"])
            
    def _mark_aoi_cells(self, coordinates: List[List[float]], priority: int):
        """Mark AOI cells in the grid for higher inspection priority"""
        for coord in coordinates:
            lon, lat = coord
            grid_x, grid_y = self._geo_to_grid(lat, lon)
            if 0 <= grid_x < self.grid_size and 0 <= grid_y < self.grid_size:
                # Higher priority AOIs get lower base cost (more attractive for path planning)
                self.base_cost_map[grid_x, grid_y] = max(0.1, 1.0 - (priority / 10.0))
                
    def update_threat_map(self, detection_data: dict) -> dict:
        """Update threat heatmap with new detection"""
        try:
            geo = detection_data.get("geo", {})
            lat = geo.get("lat")
            lon = geo.get("lon")
            confidence = detection_data.get("confidence", 0.5)
            threat_class = detection_data.get("class", "unknown")
            
            if lat is None or lon is None:
                return self._get_heatmap_geojson()
                
            # Convert to grid coordinates
            grid_x, grid_y = self._geo_to_grid(lat, lon)
            
            if 0 <= grid_x < self.grid_size and 0 <= grid_y < self.grid_size:
                # Threat scoring based on class and confidence
                threat_multiplier = self._get_threat_multiplier(threat_class)
                threat_value = confidence * threat_multiplier
                
                # Add threat with gaussian spread
                self._add_threat_gaussian(grid_x, grid_y, threat_value, radius=3)
                
        except Exception as e:
            print(f"Error updating threat map: {e}")
            
        return self._get_heatmap_geojson()
        
    def _get_threat_multiplier(self, threat_class: str) -> float:
        """Get threat multiplier based on detected class"""
        multipliers = {
            "person": 0.7,
            "car": 0.4,
            "truck": 0.6,
            "motorcycle": 0.5,
            "weapon": 1.0,
            "explosion": 1.0,
            "fire": 0.8
        }
        return multipliers.get(threat_class, 0.3)
        
    def _add_threat_gaussian(self, center_x: int, center_y: int, value: float, radius: int):
        """Add threat value with gaussian distribution"""
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                x, y = center_x + dx, center_y + dy
                if 0 <= x < self.grid_size and 0 <= y < self.grid_size:
                    distance = np.sqrt(dx*dx + dy*dy)
                    if distance <= radius:
                        gaussian_weight = np.exp(-(distance**2) / (2 * (radius/3)**2))
                        self.threat_map[x, y] += value * gaussian_weight
                        
    def _geo_to_grid(self, lat: float, lon: float) -> Tuple[int, int]:
        """Convert geographic coordinates to grid coordinates"""
        # Simple conversion - in real implementation would use proper projection
        lat_offset = (lat - self.center_lat) * 111000  # meters per degree lat
        lon_offset = (lon - self.center_lon) * 111000 * np.cos(np.radians(lat))
        
        grid_x = int((lat_offset / self.cell_size) + self.grid_size // 2)
        grid_y = int((lon_offset / self.cell_size) + self.grid_size // 2)
        
        return grid_x, grid_y
        
    def _grid_to_geo(self, grid_x: int, grid_y: int) -> Tuple[float, float]:
        """Convert grid coordinates to geographic coordinates"""
        lat_offset = (grid_x - self.grid_size // 2) * self.cell_size
        lon_offset = (grid_y - self.grid_size // 2) * self.cell_size
        
        lat = self.center_lat + lat_offset / 111000
        lon = self.center_lon + lon_offset / (111000 * np.cos(np.radians(lat)))
        
        return lat, lon
        
    def get_route_recommendation(self, convoy_status: dict, detection_data: dict) -> dict:
        """Generate route recommendation based on current threat map"""
        try:
            vehicles = convoy_status.get("vehicles", [])
            if not vehicles:
                return {"type": "no_action", "reason": "No vehicles in convoy"}
                
            # Get current convoy position (use first vehicle)
            current_vehicle = vehicles[0]
            current_lat = current_vehicle.get("lat", self.center_lat)
            current_lon = current_vehicle.get("lon", self.center_lon)
            
            # Convert to grid
            start_x, start_y = self._geo_to_grid(current_lat, current_lon)
            
            # Define destination (example: 1km ahead)
            dest_lat = current_lat + 0.01  # ~1km north
            dest_lon = current_lon
            dest_x, dest_y = self._geo_to_grid(dest_lat, dest_lon)
            
            # Find optimal path using A*
            path = self._find_optimal_path(start_x, start_y, dest_x, dest_y)
            
            if not path:
                return {"type": "no_path", "reason": "No safe path found"}
                
            # Analyze path for recommendations
            recommendation = self._analyze_path_threats(path, detection_data)
            
            return recommendation
            
        except Exception as e:
            print(f"Error generating route recommendation: {e}")
            return {"type": "error", "reason": str(e)}
            
    def _find_optimal_path(self, start_x: int, start_y: int, dest_x: int, dest_y: int) -> List[Tuple[int, int]]:
        """A* pathfinding with dynamic cost map"""
        if not (0 <= start_x < self.grid_size and 0 <= start_y < self.grid_size):
            return []
        if not (0 <= dest_x < self.grid_size and 0 <= dest_y < self.grid_size):
            return []
            
        # A* implementation
        open_set = [(0, start_x, start_y)]
        came_from = {}
        g_score = {(start_x, start_y): 0}
        f_score = {(start_x, start_y): self._heuristic(start_x, start_y, dest_x, dest_y)}
        
        while open_set:
            current_f, current_x, current_y = heapq.heappop(open_set)
            
            if current_x == dest_x and current_y == dest_y:
                # Reconstruct path
                path = []
                while (current_x, current_y) in came_from:
                    path.append((current_x, current_y))
                    current_x, current_y = came_from[(current_x, current_y)]
                path.append((start_x, start_y))
                return path[::-1]
                
            # Check neighbors
            for dx, dy in [(-1,0), (1,0), (0,-1), (0,1), (-1,-1), (-1,1), (1,-1), (1,1)]:
                neighbor_x, neighbor_y = current_x + dx, current_y + dy
                
                if not (0 <= neighbor_x < self.grid_size and 0 <= neighbor_y < self.grid_size):
                    continue
                    
                # Calculate cost
                move_cost = 1.4 if abs(dx) + abs(dy) == 2 else 1.0  # Diagonal cost
                threat_cost = self.threat_map[neighbor_x, neighbor_y] * self.threat_weight
                base_cost = self.base_cost_map[neighbor_x, neighbor_y] * self.distance_weight
                
                tentative_g = g_score[(current_x, current_y)] + move_cost + threat_cost + base_cost
                
                if (neighbor_x, neighbor_y) not in g_score or tentative_g < g_score[(neighbor_x, neighbor_y)]:
                    came_from[(neighbor_x, neighbor_y)] = (current_x, current_y)
                    g_score[(neighbor_x, neighbor_y)] = tentative_g
                    f_score[(neighbor_x, neighbor_y)] = tentative_g + self._heuristic(neighbor_x, neighbor_y, dest_x, dest_y)
                    heapq.heappush(open_set, (f_score[(neighbor_x, neighbor_y)], neighbor_x, neighbor_y))
                    
        return []  # No path found
        
    def _heuristic(self, x1: int, y1: int, x2: int, y2: int) -> float:
        """Manhattan distance heuristic"""
        return abs(x1 - x2) + abs(y1 - y2)
        
    def _analyze_path_threats(self, path: List[Tuple[int, int]], detection_data: dict) -> dict:
        """Analyze path for threat levels and generate recommendations"""
        if not path:
            return {"type": "no_action"}
            
        # Calculate average threat along path
        total_threat = sum(self.threat_map[x, y] for x, y in path)
        avg_threat = total_threat / len(path)
        
        # Get detection threat level
        detection_threat = detection_data.get("confidence", 0) * self._get_threat_multiplier(detection_data.get("class", "unknown"))
        
        # Generate recommendation based on threat levels
        if avg_threat > 0.7 or detection_threat > 0.8:
            return {
                "type": "reroute",
                "priority": "high",
                "reason": "High threat detected along route",
                "action": "Change to alternate route",
                "threat_level": avg_threat,
                "path": [self._grid_to_geo(x, y) for x, y in path[:5]]  # First 5 waypoints
            }
        elif avg_threat > 0.4 or detection_threat > 0.5:
            return {
                "type": "formation_change",
                "priority": "medium", 
                "reason": "Moderate threat detected",
                "action": "Switch to staggered formation",
                "threat_level": avg_threat
            }
        elif avg_threat > 0.2:
            return {
                "type": "speed_adjust",
                "priority": "low",
                "reason": "Low threat detected",
                "action": "Reduce speed by 20%",
                "threat_level": avg_threat
            }
        else:
            return {
                "type": "continue",
                "priority": "info",
                "reason": "Path clear",
                "action": "Maintain current route and speed",
                "threat_level": avg_threat
            }
            
    def _get_heatmap_geojson(self) -> dict:
        """Convert threat map to GeoJSON format for dashboard"""
        features = []
        
        # Sample every 5th cell to reduce data size
        step = 5
        for i in range(0, self.grid_size, step):
            for j in range(0, self.grid_size, step):
                threat_value = self.threat_map[i, j]
                if threat_value > 0.1:  # Only include cells with significant threat
                    lat, lon = self._grid_to_geo(i, j)
                    
                    # Create cell polygon
                    cell_size_deg = self.cell_size / 111000  # Convert meters to degrees
                    coordinates = [[
                        [lon - cell_size_deg/2, lat - cell_size_deg/2],
                        [lon + cell_size_deg/2, lat - cell_size_deg/2],
                        [lon + cell_size_deg/2, lat + cell_size_deg/2],
                        [lon - cell_size_deg/2, lat + cell_size_deg/2],
                        [lon - cell_size_deg/2, lat - cell_size_deg/2]
                    ]]
                    
                    features.append({
                        "type": "Feature",
                        "geometry": {
                            "type": "Polygon",
                            "coordinates": coordinates
                        },
                        "properties": {
                            "threat_level": float(threat_value),
                            "color": self._threat_to_color(threat_value)
                        }
                    })
                    
        return {
            "type": "FeatureCollection",
            "features": features
        }
        
    def _threat_to_color(self, threat_value: float) -> str:
        """Convert threat value to color for visualization"""
        # Yellow to red gradient
        if threat_value < 0.3:
            return "#FFD400"  # Yellow
        elif threat_value < 0.6:
            return "#FF8C00"  # Orange
        else:
            return "#FF4500"  # Red