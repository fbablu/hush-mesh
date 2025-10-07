import pytest
import numpy as np
from backend.planner.mission_planner import MissionPlanner

class TestMissionPlanner:
    def setup_method(self):
        """Setup test fixtures"""
        self.planner = MissionPlanner(grid_size=50, cell_size=10.0)
        
    def test_initialization(self):
        """Test planner initialization"""
        assert self.planner.grid_size == 50
        assert self.planner.cell_size == 10.0
        assert self.planner.threat_map.shape == (50, 50)
        assert self.planner.active_mission is None
        
    def test_start_stop_mission(self):
        """Test mission start and stop"""
        self.planner.start_mission("test_scenario")
        assert self.planner.active_mission == "test_scenario"
        
        self.planner.stop_mission()
        assert self.planner.active_mission is None
        assert len(self.planner.aois) == 0
        
    def test_geo_grid_conversion(self):
        """Test geographic to grid coordinate conversion"""
        # Test center point
        lat, lon = self.planner.center_lat, self.planner.center_lon
        grid_x, grid_y = self.planner._geo_to_grid(lat, lon)
        
        # Should be near center of grid
        assert abs(grid_x - self.planner.grid_size // 2) <= 1
        assert abs(grid_y - self.planner.grid_size // 2) <= 1
        
        # Test round trip conversion
        converted_lat, converted_lon = self.planner._grid_to_geo(grid_x, grid_y)
        assert abs(converted_lat - lat) < 0.001
        assert abs(converted_lon - lon) < 0.001
        
    def test_threat_map_update(self):
        """Test threat map updates with detections"""
        detection = {
            "class": "person",
            "confidence": 0.8,
            "geo": {
                "lat": self.planner.center_lat,
                "lon": self.planner.center_lon
            }
        }
        
        # Initial threat map should be empty
        initial_sum = np.sum(self.planner.threat_map)
        assert initial_sum == 0
        
        # Update with detection
        heatmap = self.planner.update_threat_map(detection)
        
        # Threat map should now have values
        updated_sum = np.sum(self.planner.threat_map)
        assert updated_sum > 0
        
        # Heatmap should be returned
        assert heatmap["type"] == "FeatureCollection"
        
    def test_threat_multipliers(self):
        """Test threat multiplier calculation"""
        assert self.planner._get_threat_multiplier("person") == 0.7
        assert self.planner._get_threat_multiplier("weapon") == 1.0
        assert self.planner._get_threat_multiplier("unknown") == 0.3
        
    def test_aoi_management(self):
        """Test AOI addition and management"""
        aoi_data = {
            "type": "polygon",
            "geometry": {
                "type": "Polygon",
                "coordinates": [[
                    [self.planner.center_lon - 0.001, self.planner.center_lat - 0.001],
                    [self.planner.center_lon + 0.001, self.planner.center_lat - 0.001],
                    [self.planner.center_lon + 0.001, self.planner.center_lat + 0.001],
                    [self.planner.center_lon - 0.001, self.planner.center_lat + 0.001],
                    [self.planner.center_lon - 0.001, self.planner.center_lat - 0.001]
                ]]
            },
            "priority": 8
        }
        
        self.planner.add_aoi("test_aoi", aoi_data)
        assert "test_aoi" in self.planner.aois
        assert self.planner.aois["test_aoi"]["priority"] == 8
        
    def test_pathfinding(self):
        """Test A* pathfinding algorithm"""
        # Test simple path from corner to corner
        start_x, start_y = 5, 5
        dest_x, dest_y = 10, 10
        
        path = self.planner._find_optimal_path(start_x, start_y, dest_x, dest_y)
        
        # Should find a path
        assert len(path) > 0
        
        # Path should start and end at correct points
        assert path[0] == (start_x, start_y)
        assert path[-1] == (dest_x, dest_y)
        
    def test_route_recommendation(self):
        """Test route recommendation generation"""
        convoy_status = {
            "vehicles": [{
                "id": "veh-01",
                "lat": self.planner.center_lat,
                "lon": self.planner.center_lon
            }]
        }
        
        detection_data = {
            "class": "person",
            "confidence": 0.9,
            "geo": {
                "lat": self.planner.center_lat + 0.001,
                "lon": self.planner.center_lon + 0.001
            }
        }
        
        # Update threat map first
        self.planner.update_threat_map(detection_data)
        
        # Get recommendation
        recommendation = self.planner.get_route_recommendation(convoy_status, detection_data)
        
        # Should return a valid recommendation
        assert "type" in recommendation
        assert "priority" in recommendation
        assert recommendation["type"] in ["reroute", "formation_change", "speed_adjust", "continue", "no_action"]
        
    def test_heatmap_generation(self):
        """Test heatmap GeoJSON generation"""
        # Add some threat data
        detection = {
            "class": "weapon",
            "confidence": 0.9,
            "geo": {
                "lat": self.planner.center_lat,
                "lon": self.planner.center_lon
            }
        }
        
        heatmap = self.planner.update_threat_map(detection)
        
        # Should be valid GeoJSON
        assert heatmap["type"] == "FeatureCollection"
        assert "features" in heatmap
        
        # Should have at least one feature for high threat
        assert len(heatmap["features"]) > 0
        
        # Features should have proper structure
        for feature in heatmap["features"]:
            assert feature["type"] == "Feature"
            assert "geometry" in feature
            assert "properties" in feature
            assert "threat_level" in feature["properties"]
            
    def test_threat_color_mapping(self):
        """Test threat level to color mapping"""
        assert self.planner._threat_to_color(0.2) == "#FFD400"  # Yellow
        assert self.planner._threat_to_color(0.5) == "#FF8C00"  # Orange  
        assert self.planner._threat_to_color(0.8) == "#FF4500"  # Red