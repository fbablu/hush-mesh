#!/usr/bin/env python3
"""
Script to spawn convoy vehicles and drones for simulation
"""

import asyncio
import argparse
import os
import sys
import subprocess
import time
from pathlib import Path

def start_edge_agent(drone_id, vehicle_id, mqtt_broker="localhost"):
    """Start an edge agent process"""
    env = os.environ.copy()
    env.update({
        "DRONE_ID": drone_id,
        "VEHICLE_ID": vehicle_id,
        "MQTT_BROKER": mqtt_broker,
        "USE_MINIMAL": "true",
        "CAMERA_SOURCE": "simulator"
    })
    
    # Get the project root directory
    project_root = Path(__file__).parent.parent
    edge_dir = project_root / "edge"
    
    cmd = [sys.executable, "agent.py"]
    
    print(f"Starting edge agent {drone_id} for vehicle {vehicle_id}")
    
    return subprocess.Popen(
        cmd,
        cwd=edge_dir,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )

def main():
    parser = argparse.ArgumentParser(description="Spawn convoy vehicles and drones")
    parser.add_argument("--vehicles", type=int, default=3, help="Number of vehicles")
    parser.add_argument("--drones", type=int, default=3, help="Number of drones")
    parser.add_argument("--mqtt-broker", default="localhost", help="MQTT broker address")
    parser.add_argument("--delay", type=float, default=2.0, help="Delay between spawning agents")
    
    args = parser.parse_args()
    
    print(f"Spawning convoy with {args.vehicles} vehicles and {args.drones} drones")
    print(f"MQTT Broker: {args.mqtt_broker}")
    print("-" * 50)
    
    processes = []
    
    try:
        # Spawn edge agents (one per drone)
        for i in range(args.drones):
            drone_id = f"drone-{i+1:02d}"
            vehicle_id = f"veh-{i+1:02d}" if i < args.vehicles else f"veh-{args.vehicles:02d}"
            
            process = start_edge_agent(drone_id, vehicle_id, args.mqtt_broker)
            processes.append((process, drone_id))
            
            time.sleep(args.delay)
        
        print(f"\nSpawned {len(processes)} edge agents")
        print("Press Ctrl+C to stop all agents")
        
        # Wait for all processes
        while True:
            time.sleep(1)
            
            # Check if any process has died
            for process, drone_id in processes:
                if process.poll() is not None:
                    print(f"Edge agent {drone_id} has stopped")
                    
    except KeyboardInterrupt:
        print("\nShutting down all edge agents...")
        
        for process, drone_id in processes:
            print(f"Stopping {drone_id}...")
            process.terminate()
            
        # Wait for graceful shutdown
        time.sleep(2)
        
        # Force kill if necessary
        for process, drone_id in processes:
            if process.poll() is None:
                print(f"Force killing {drone_id}...")
                process.kill()
                
        print("All edge agents stopped")

if __name__ == "__main__":
    main()