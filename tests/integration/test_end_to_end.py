import pytest
import asyncio
import json
import time
import requests
import subprocess
import signal
import os
from pathlib import Path

class TestEndToEnd:
    """End-to-end integration tests for ACPS system"""
    
    @pytest.fixture(scope="class")
    def system_setup(self):
        """Setup the complete system for testing"""
        processes = []
        
        try:
            # Start MQTT broker
            mqtt_process = subprocess.Popen([
                "docker", "run", "--rm", "-p", "1883:1883", 
                "--name", "test-mosquitto", "eclipse-mosquitto:2.0"
            ])
            processes.append(("mosquitto", mqtt_process))
            time.sleep(3)  # Wait for MQTT to start
            
            # Start backend
            backend_env = os.environ.copy()
            backend_env.update({
                "MQTT_BROKER": "localhost",
                "USE_MINIMAL": "true"
            })
            
            backend_process = subprocess.Popen([
                "python", "-m", "uvicorn", "app:app", "--port", "8001"
            ], cwd=Path(__file__).parent.parent.parent / "backend", env=backend_env)
            processes.append(("backend", backend_process))
            time.sleep(5)  # Wait for backend to start
            
            # Start edge agents
            for i in range(3):
                drone_id = f"test-drone-{i+1:02d}"
                vehicle_id = f"test-veh-{i+1:02d}"
                
                agent_env = os.environ.copy()
                agent_env.update({
                    "DRONE_ID": drone_id,
                    "VEHICLE_ID": vehicle_id,
                    "MQTT_BROKER": "localhost",
                    "USE_MINIMAL": "true",
                    "CAMERA_SOURCE": "simulator"
                })
                
                agent_process = subprocess.Popen([
                    "python", "agent.py"
                ], cwd=Path(__file__).parent.parent.parent / "edge", env=agent_env)
                processes.append((drone_id, agent_process))
                
            time.sleep(5)  # Wait for agents to connect
            
            yield processes
            
        finally:
            # Cleanup all processes
            for name, process in processes:
                print(f"Stopping {name}...")
                try:
                    process.terminate()
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                except Exception as e:
                    print(f"Error stopping {name}: {e}")
                    
            # Stop docker containers
            subprocess.run(["docker", "stop", "test-mosquitto"], 
                         capture_output=True, check=False)
    
    def test_system_health(self, system_setup):
        """Test that all system components are healthy"""
        # Check backend health
        try:
            response = requests.get("http://localhost:8001/api/convoy", timeout=5)
            assert response.status_code == 200
        except requests.exceptions.RequestException:
            pytest.fail("Backend not responding")
            
        # Check metrics endpoint
        try:
            response = requests.get("http://localhost:8001/metrics", timeout=5)
            assert response.status_code == 200
            assert "acps_" in response.text
        except requests.exceptions.RequestException:
            pytest.fail("Metrics endpoint not responding")
    
    def test_detection_flow(self, system_setup):
        """Test detection data flow from edge to backend"""
        base_url = "http://localhost:8001"
        
        # Start mission to activate edge agents
        requests.post(f"{base_url}/api/mission/start", json={
            "scenario": "ambush",
            "vehicles": 3,
            "drones": 3
        })
        
        # Wait for detections to be generated
        time.sleep(10)
        
        # Check for detections
        response = requests.get(f"{base_url}/api/detections", params={"limit": 10})
        assert response.status_code == 200
        detections = response.json()
        
        # Should have at least one detection
        assert len(detections) > 0
        
        # Validate detection structure
        detection = detections[0]
        required_fields = ["drone_id", "class", "confidence", "bbox", "timestamp"]
        for field in required_fields:
            assert field in detection
            
        # Stop mission
        requests.post(f"{base_url}/api/mission/stop")