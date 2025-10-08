#!/usr/bin/env python3
import heapq
import math
import numpy as np
from typing import List, Tuple, Dict, Optional
import json

class PathPlanner:
    def __init__(self, grid_size_km=50, resolution_m=100):
        """Initialize path planner with A* algorithm"""
        self.grid_size_km = grid_size_km
        self.resolution_m = resolution_m
        self.grid_cells = int((grid_size_km * 1000) / resolution_m)
        
        # Initialize threat grid (0 = safe, 1 = threat)
        self.threat_grid = np.zeros((self.grid_cells, self.grid_cells))
        self.cost_grid = np.ones((self.grid_cells, self.grid_cells))
        
    def update_threats(self, threats: List[Dict], convoy_position: Tuple[float, float]):
        """Update threat grid based on detected threats"""
        # Clear previous threats
        self.threat_grid.fill(0)
        self.cost_grid.fill(1.0)
        
        convoy_lat, convoy_lon = convoy_position
        
        for threat in threats:
            if not threat.get('ground_truth', {}).get('threat_present', False):
                continue
                
            threat_lat = threat.get('position', {}).get('estimated_lat', convoy_lat)
            threat_lon = threat.get('position', {}).get('estimated_lon', convoy_lon)
            
            # Convert to grid coordinates
            grid_x, grid_y = self._lat_lon_to_grid(threat_lat, threat_lon, convoy_lat, convoy_lon)
            
            if 0 <= grid_x < self.grid_cells and 0 <= grid_y < self.grid_cells:
                # Create threat zone based on threat type
                threat_type = threat.get('threat_type', 'unknown')
                radius = self._get_threat_radius(threat_type)
                
                self._add_threat_zone(grid_x, grid_y, radius, threat_type)
    
    def _get_threat_radius(self, threat_type: str) -> int:
        """Get threat avoidance radius in grid cells"""
        radius_map = {
            'small_fast_craft': 15,  # 1.5km radius
            'floating_mine_like_object': 20,  # 2km radius
            'submarine_periscope': 25,  # 2.5km radius
            'swarm': 30,  # 3km radius
            'debris_field': 10,  # 1km radius
            'shallow_water': 8,   # 800m radius
            'oil_spill': 12      # 1.2km radius
        }
        return radius_map.get(threat_type, 15)
    
    def _add_threat_zone(self, center_x: int, center_y: int, radius: int, threat_type: str):
        """Add circular threat zone to grid"""
        threat_severity = {
            'small_fast_craft': 10.0,
            'floating_mine_like_object': 15.0,
            'submarine_periscope': 8.0,
            'swarm': 20.0,
            'debris_field': 5.0,
            'shallow_water': 12.0,
            'oil_spill': 6.0
        }.get(threat_type, 10.0)
        
        for x in range(max(0, center_x - radius), min(self.grid_cells, center_x + radius + 1)):
            for y in range(max(0, center_y - radius), min(self.grid_cells, center_y + radius + 1)):
                distance = math.sqrt((x - center_x)**2 + (y - center_y)**2)
                if distance <= radius:
                    # Gradient threat cost (higher near center)
                    cost_multiplier = threat_severity * (1 - distance / radius) + 1
                    self.cost_grid[x, y] = max(self.cost_grid[x, y], cost_multiplier)
                    
                    if distance <= radius * 0.5:  # Core threat zone
                        self.threat_grid[x, y] = 1
    
    def find_multiple_paths(self, start_pos: Tuple[float, float], 
                           end_pos: Tuple[float, float],
                           convoy_position: Tuple[float, float]) -> Dict:
        """Find multiple path alternatives with different strategies"""
        
        start_grid = self._lat_lon_to_grid(start_pos[0], start_pos[1], convoy_position[0], convoy_position[1])
        end_grid = self._lat_lon_to_grid(end_pos[0], end_pos[1], convoy_position[0], convoy_position[1])
        
        paths = {}
        
        # Strategy 1: Optimal (balanced distance and threat avoidance)
        paths['optimal'] = self._astar_with_strategy(start_grid, end_grid, 'optimal')
        
        # Strategy 2: Fastest (minimize distance, accept some threat exposure)
        paths['fastest'] = self._astar_with_strategy(start_grid, end_grid, 'fastest')
        
        # Strategy 3: Safest (maximum threat avoidance, longer distance OK)
        paths['safest'] = self._astar_with_strategy(start_grid, end_grid, 'safest')
        
        # Convert to coordinates and calculate metrics
        result = {}
        for strategy, path_grid in paths.items():
            if path_grid:
                path_coords = []
                for grid_x, grid_y in path_grid:
                    lat, lon = self._grid_to_lat_lon(grid_x, grid_y, convoy_position[0], convoy_position[1])
                    path_coords.append((lat, lon))
                
                metrics = self.calculate_path_metrics(path_coords)
                result[strategy] = {
                    'path': path_coords,
                    'metrics': metrics,
                    'score': self._calculate_path_score(metrics, strategy)
                }
        
        return result
    
    def find_optimal_path(self, start_pos: Tuple[float, float], 
                         end_pos: Tuple[float, float],
                         convoy_position: Tuple[float, float]) -> List[Tuple[float, float]]:
        """Find single optimal path (backward compatibility)"""
        paths = self.find_multiple_paths(start_pos, end_pos, convoy_position)
        if 'optimal' in paths:
            return paths['optimal']['path']
        return []
    
    def _astar_with_strategy(self, start: Tuple[int, int], goal: Tuple[int, int], strategy: str) -> List[Tuple[int, int]]:
        """A* pathfinding with different strategies"""
        def heuristic(a: Tuple[int, int], b: Tuple[int, int]) -> float:
            return math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)
        
        def get_neighbors(pos: Tuple[int, int]) -> List[Tuple[int, int]]:
            x, y = pos
            neighbors = []
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dx == 0 and dy == 0:
                        continue
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < self.grid_cells and 0 <= ny < self.grid_cells:
                        neighbors.append((nx, ny))
            return neighbors
        
        # Strategy-specific cost weights
        if strategy == 'fastest':
            threat_weight = 0.3  # Low threat avoidance
            distance_weight = 1.0
        elif strategy == 'safest':
            threat_weight = 3.0  # High threat avoidance
            distance_weight = 0.5
        else:  # optimal
            threat_weight = 1.0  # Balanced
            distance_weight = 1.0
        
        open_set = [(0, start)]
        came_from = {}
        g_score = {start: 0}
        f_score = {start: heuristic(start, goal)}
        
        while open_set:
            current = heapq.heappop(open_set)[1]
            
            if current == goal:
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                return path[::-1]
            
            for neighbor in get_neighbors(current):
                base_cost = 1.414 if abs(neighbor[0] - current[0]) + abs(neighbor[1] - current[1]) == 2 else 1.0
                threat_cost = self.cost_grid[neighbor[0], neighbor[1]]
                
                # Apply strategy weights
                total_cost = (base_cost * distance_weight) + ((threat_cost - 1.0) * threat_weight)
                tentative_g_score = g_score[current] + total_cost
                
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + heuristic(neighbor, goal) * distance_weight
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))
        
        return []
    
    def _calculate_path_score(self, metrics: Dict, strategy: str) -> float:
        """Calculate overall path score based on strategy"""
        distance = metrics['total_distance_km']
        threat_exposure = metrics['threat_exposure']
        
        if strategy == 'fastest':
            return 100 - distance * 2 - threat_exposure * 0.5
        elif strategy == 'safest':
            return 100 - threat_exposure * 5 - distance * 0.5
        else:  # optimal
            return 100 - distance * 1.5 - threat_exposure * 2
    
    def _lat_lon_to_grid(self, lat: float, lon: float, 
                        center_lat: float, center_lon: float) -> Tuple[int, int]:
        """Convert lat/lon to grid coordinates"""
        # Approximate conversion (good enough for local navigation)
        lat_diff = lat - center_lat
        lon_diff = lon - center_lon
        
        # Convert to meters (rough approximation)
        lat_m = lat_diff * 111000  # 1 degree lat ≈ 111km
        lon_m = lon_diff * 111000 * math.cos(math.radians(center_lat))
        
        # Convert to grid coordinates (center of grid is convoy position)
        grid_x = int((lat_m / self.resolution_m) + (self.grid_cells // 2))
        grid_y = int((lon_m / self.resolution_m) + (self.grid_cells // 2))
        
        return (grid_x, grid_y)
    
    def _grid_to_lat_lon(self, grid_x: int, grid_y: int,
                        center_lat: float, center_lon: float) -> Tuple[float, float]:
        """Convert grid coordinates back to lat/lon"""
        # Convert grid to meters relative to center
        lat_m = (grid_x - (self.grid_cells // 2)) * self.resolution_m
        lon_m = (grid_y - (self.grid_cells // 2)) * self.resolution_m
        
        # Convert to lat/lon
        lat = center_lat + (lat_m / 111000)
        lon = center_lon + (lon_m / (111000 * math.cos(math.radians(center_lat))))
        
        return (lat, lon)
    
    def calculate_path_metrics(self, path: List[Tuple[float, float]]) -> Dict:
        """Calculate path metrics for evaluation"""
        if len(path) < 2:
            return {'total_distance_km': 0, 'threat_exposure': 0, 'waypoint_count': 0}
        
        total_distance = 0
        threat_exposure = 0
        
        for i in range(len(path) - 1):
            # Calculate distance between waypoints
            lat1, lon1 = path[i]
            lat2, lon2 = path[i + 1]
            
            # Haversine distance
            R = 6371  # Earth radius in km
            dlat = math.radians(lat2 - lat1)
            dlon = math.radians(lon2 - lon1)
            a = (math.sin(dlat/2)**2 + 
                 math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * 
                 math.sin(dlon/2)**2)
            c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
            distance = R * c
            
            total_distance += distance
            
            # Calculate threat exposure along segment
            # (simplified - could be more sophisticated)
            grid_x, grid_y = self._lat_lon_to_grid(lat1, lon1, path[0][0], path[0][1])
            if 0 <= grid_x < self.grid_cells and 0 <= grid_y < self.grid_cells:
                threat_exposure += self.cost_grid[grid_x, grid_y] - 1.0
        
        return {
            'total_distance_km': round(total_distance, 2),
            'threat_exposure': round(threat_exposure, 2),
            'waypoint_count': len(path),
            'avg_threat_level': round(threat_exposure / max(len(path) - 1, 1), 3)
        }
    
    def generate_evasive_maneuver(self, convoy_pos: Tuple[float, float],
                                 threat_bearing: float, 
                                 threat_distance: float) -> List[Tuple[float, float]]:
        """Generate immediate evasive maneuver waypoints"""
        convoy_lat, convoy_lon = convoy_pos
        
        # Calculate evasion direction (perpendicular to threat bearing)
        evasion_bearing_1 = (threat_bearing + 90) % 360
        evasion_bearing_2 = (threat_bearing - 90) % 360
        
        # Choose evasion direction based on threat grid
        evasion_distance_km = min(threat_distance * 0.8, 5.0)  # Stay within reasonable distance
        
        waypoints = []
        
        # Immediate evasion waypoint
        for bearing in [evasion_bearing_1, evasion_bearing_2]:
            lat_offset = (evasion_distance_km / 111.0) * math.cos(math.radians(bearing))
            lon_offset = (evasion_distance_km / (111.0 * math.cos(math.radians(convoy_lat)))) * math.sin(math.radians(bearing))
            
            evasion_lat = convoy_lat + lat_offset
            evasion_lon = convoy_lon + lon_offset
            
            # Check if this direction is safer
            grid_x, grid_y = self._lat_lon_to_grid(evasion_lat, evasion_lon, convoy_lat, convoy_lon)
            if (0 <= grid_x < self.grid_cells and 0 <= grid_y < self.grid_cells and 
                self.cost_grid[grid_x, grid_y] < 3.0):  # Acceptable threat level
                waypoints.append((evasion_lat, evasion_lon))
                break
        
        return waypoints

def main():
    """Test path planning functionality"""
    planner = PathPlanner()
    
    # Example convoy position
    convoy_pos = (25.7617, -80.1918)
    
    # Example threats
    threats = [
        {
            'threat_type': 'small_fast_craft',
            'position': {'estimated_lat': 25.7700, 'estimated_lon': -80.1800},
            'ground_truth': {'threat_present': True}
        },
        {
            'threat_type': 'floating_mine_like_object', 
            'position': {'estimated_lat': 25.7650, 'estimated_lon': -80.1850},
            'ground_truth': {'threat_present': True}
        }
    ]
    
    # Update threat grid
    planner.update_threats(threats, convoy_pos)
    
    # Find path from current position to destination
    start_pos = convoy_pos
    end_pos = (25.8000, -80.1500)  # Destination
    
    path = planner.find_optimal_path(start_pos, end_pos, convoy_pos)
    metrics = planner.calculate_path_metrics(path)
    
    print(f"Optimal path found with {len(path)} waypoints")
    print(f"Path metrics: {json.dumps(metrics, indent=2)}")
    
    # Generate evasive maneuver
    evasion = planner.generate_evasive_maneuver(convoy_pos, 45, 2.0)
    print(f"Evasive maneuver: {evasion}")

if __name__ == '__main__':
    main()