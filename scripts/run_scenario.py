#!/usr/bin/env python3
"""
Script to run predefined scenarios
"""

import asyncio
import argparse
import json
import requests
import time
import sys
from pathlib import Path

class ScenarioRunner:
    def __init__(self, api_url="http://localhost:8000"):
        self.api_url = api_url
        self.scenario_configs = self.load_scenario_configs()
        
    def load_scenario_configs(self):
        """Load scenario configurations"""
        config_dir = Path(__file__).parent.parent / "sim_configs"
        configs = {}
        
        # Default configurations if files don't exist
        configs["ambush"] = {
            "name": "Ambush Test",
            "description": "Convoy encounters potential ambush with person detection",
            "vehicles": 3,
            "drones": 3,
            "duration": 120,
            "threat_spawn_probability": 0.3,
            "weather": "clear"
        }
        
        configs["weather"] = {
            "name": "Weather Conditions",
            "description": "Heavy rain reduces detection capability",
            "vehicles": 3,
            "drones": 3,
            "duration": 180,
            "threat_spawn_probability": 0.2,
            "weather": "heavy_rain",
            "visibility_reduction": 0.4
        }
        
        configs["connectivity"] = {
            "name": "Network Issues",
            "description": "Intermittent connectivity tests edge autonomy",
            "vehicles": 3,
            "drones": 3,
            "duration": 150,
            "threat_spawn_probability": 0.25,
            "weather": "clear",
            "packet_loss": 0.4,
            "network_outages": True
        }
        
        return configs
        
    def run_scenario(self, scenario_name):
        """Run a specific scenario"""
        if scenario_name not in self.scenario_configs:
            print(f"Unknown scenario: {scenario_name}")
            print(f"Available scenarios: {list(self.scenario_configs.keys())}")
            return False
            
        config = self.scenario_configs[scenario_name]
        print(f"Running scenario: {config['name']}")
        print(f"Description: {config['description']}")
        print("-" * 50)
        
        try:
            # Start mission
            response = requests.post(f"{self.api_url}/api/mission/start", json={
                "scenario": scenario_name,
                "vehicles": config["vehicles"],
                "drones": config["drones"]
            })
            
            if response.status_code != 200:
                print(f"Failed to start mission: {response.text}")
                return False
                
            print("Mission started successfully")
            
            # Create scenario-specific AOIs
            if scenario_name == "ambush":
                self.create_ambush_aois()
            elif scenario_name == "weather":
                self.create_weather_aois()
            elif scenario_name == "connectivity":
                self.create_connectivity_aois()
                
            # Run for specified duration
            print(f"Running scenario for {config['duration']} seconds...")
            
            start_time = time.time()
            while time.time() - start_time < config['duration']:
                # Print status updates
                elapsed = int(time.time() - start_time)
                remaining = config['duration'] - elapsed
                
                if elapsed % 30 == 0:  # Every 30 seconds
                    print(f"Scenario progress: {elapsed}s elapsed, {remaining}s remaining")
                    self.print_status()
                    
                time.sleep(1)
                
            # Stop mission
            response = requests.post(f"{self.api_url}/api/mission/stop")
            if response.status_code == 200:
                print("Mission stopped successfully")
            else:
                print(f"Failed to stop mission: {response.text}")
                
            print("Scenario completed!")
            return True
            
        except requests.exceptions.ConnectionError:
            print("Error: Could not connect to ACPS backend")
            print("Make sure the backend is running on", self.api_url)
            return False
        except Exception as e:
            print(f"Error running scenario: {e}")
            return False
            
    def create_ambush_aois(self):
        """Create AOIs for ambush scenario"""
        aois = [
            {
                "type": "polygon",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[
                        [-71.060, 42.361],
                        [-71.059, 42.361],
                        [-71.059, 42.362],
                        [-71.060, 42.362],
                        [-71.060, 42.361]
                    ]]
                },
                "priority": 8,
                "description": "Potential ambush point - roadside"
            },
            {
                "type": "polygon", 
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[
                        [-71.058, 42.359],
                        [-71.057, 42.359],
                        [-71.057, 42.360],
                        [-71.058, 42.360],
                        [-71.058, 42.359]
                    ]]
                },
                "priority": 6,
                "description": "Overwatch position"
            }
        ]
        
        for aoi in aois:
            try:
                response = requests.post(f"{self.api_url}/api/aoi", json=aoi)
                if response.status_code == 200:
                    print(f"Created AOI: {aoi['description']}")
            except Exception as e:
                print(f"Failed to create AOI: {e}")
                
    def create_weather_aois(self):
        """Create AOIs for weather scenario"""
        aois = [
            {
                "type": "polygon",
                "geometry": {
                    "type": "Polygon", 
                    "coordinates": [[
                        [-71.061, 42.360],
                        [-71.060, 42.360],
                        [-71.060, 42.361],
                        [-71.061, 42.361],
                        [-71.061, 42.360]
                    ]]
                },
                "priority": 7,
                "description": "Low visibility area - increased surveillance"
            }
        ]
        
        for aoi in aois:
            try:
                response = requests.post(f"{self.api_url}/api/aoi", json=aoi)
                if response.status_code == 200:
                    print(f"Created AOI: {aoi['description']}")
            except Exception as e:
                print(f"Failed to create AOI: {e}")
                
    def create_connectivity_aois(self):
        """Create AOIs for connectivity scenario"""
        aois = [
            {
                "type": "polygon",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[
                        [-71.059, 42.361],
                        [-71.058, 42.361], 
                        [-71.058, 42.362],
                        [-71.059, 42.362],
                        [-71.059, 42.361]
                    ]]
                },
                "priority": 9,
                "description": "Dead zone - test autonomous operation"
            }
        ]
        
        for aoi in aois:
            try:
                response = requests.post(f"{self.api_url}/api/aoi", json=aoi)
                if response.status_code == 200:
                    print(f"Created AOI: {aoi['description']}")
            except Exception as e:
                print(f"Failed to create AOI: {e}")
                
    def print_status(self):
        """Print current system status"""
        try:
            # Get convoy status
            response = requests.get(f"{self.api_url}/api/convoy")
            if response.status_code == 200:
                convoy = response.json()
                print(f"  Vehicles: {len(convoy.get('vehicles', []))}, Drones: {len(convoy.get('drones', []))}")
                
            # Get detection count
            response = requests.get(f"{self.api_url}/api/detections", params={"limit": 1})
            if response.status_code == 200:
                detections = response.json()
                print(f"  Total detections: {len(detections)}")
                
        except Exception as e:
            print(f"  Status check failed: {e}")

def main():
    parser = argparse.ArgumentParser(description="Run ACPS scenarios")
    parser.add_argument("--scenario", required=True, 
                       choices=["ambush", "weather", "connectivity"],
                       help="Scenario to run")
    parser.add_argument("--api-url", default="http://localhost:8000",
                       help="ACPS backend API URL")
    
    args = parser.parse_args()
    
    runner = ScenarioRunner(args.api_url)
    
    success = runner.run_scenario(args.scenario)
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()