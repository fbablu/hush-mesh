import asyncio
import json
import os
from datetime import datetime
from typing import Dict, List, Optional
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import paho.mqtt.client as mqtt
from planner.mission_planner import MissionPlanner
from sim.convoy_simulator import ConvoySimulator

app = FastAPI(title="ACPS Backend", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
convoy_simulator = ConvoySimulator()
mission_planner = MissionPlanner()
connected_clients: List[WebSocket] = []
convoy_status = {"vehicles": [], "drones": [], "active": False}
aois: Dict[str, dict] = {}
detections: List[dict] = []
heatmap_data = {"type": "FeatureCollection", "features": []}

# MQTT Configuration
MQTT_BROKER = os.getenv("MQTT_BROKER", "localhost")
MQTT_PORT = int(os.getenv("MQTT_PORT", "1883"))

mqtt_client = mqtt.Client()

def on_mqtt_connect(client, userdata, flags, rc):
    print(f"MQTT Connected with result code {rc}")
    client.subscribe("acps/detections/+")
    client.subscribe("acps/telemetry/+")

def on_mqtt_message(client, userdata, msg):
    try:
        topic_parts = msg.topic.split('/')
        message_type = topic_parts[1]
        drone_id = topic_parts[2]
        
        payload = json.loads(msg.payload.decode())
        
        if message_type == "detections":
            handle_detection(payload)
        elif message_type == "telemetry":
            handle_telemetry(payload)
            
    except Exception as e:
        print(f"Error processing MQTT message: {e}")

def handle_detection(detection_data):
    global detections, heatmap_data
    
    detections.append(detection_data)
    
    # Update heatmap
    heatmap_data = mission_planner.update_threat_map(detection_data)
    
    # Get planner recommendation
    recommendation = mission_planner.get_route_recommendation(
        convoy_status, detection_data
    )
    
    # Broadcast to dashboard
    asyncio.create_task(broadcast_to_clients({
        "type": "detection",
        "data": detection_data,
        "heatmap": heatmap_data,
        "recommendation": recommendation
    }))

def handle_telemetry(telemetry_data):
    # Update convoy status
    if telemetry_data.get("type") == "vehicle":
        update_vehicle_status(telemetry_data)
    elif telemetry_data.get("type") == "drone":
        update_drone_status(telemetry_data)
    
    # Broadcast to dashboard
    asyncio.create_task(broadcast_to_clients({
        "type": "telemetry",
        "data": telemetry_data
    }))

def update_vehicle_status(data):
    vehicle_id = data.get("vehicle_id")
    for i, vehicle in enumerate(convoy_status["vehicles"]):
        if vehicle["id"] == vehicle_id:
            convoy_status["vehicles"][i].update(data)
            return
    convoy_status["vehicles"].append(data)

def update_drone_status(data):
    drone_id = data.get("drone_id")
    for i, drone in enumerate(convoy_status["drones"]):
        if drone["id"] == drone_id:
            convoy_status["drones"][i].update(data)
            return
    convoy_status["drones"].append(data)

async def broadcast_to_clients(message):
    if connected_clients:
        disconnected = []
        for client in connected_clients:
            try:
                await client.send_text(json.dumps(message))
            except:
                disconnected.append(client)
        
        for client in disconnected:
            connected_clients.remove(client)

async def update_convoy_status():
    """Periodically update convoy status from simulator"""
    print("Starting convoy status updates")
    while convoy_status["active"]:
        try:
            sim_status = convoy_simulator.get_status()
            convoy_status["vehicles"] = sim_status["vehicles"]
            convoy_status["drones"] = sim_status["drones"]
            
            print(f"Convoy update: {len(convoy_status['vehicles'])} vehicles, {len(convoy_status['drones'])} drones")
            
            # Check for new detections
            if hasattr(convoy_simulator, 'recent_detections'):
                for detection in convoy_simulator.recent_detections:
                    # Process detection through mission planner
                    handle_detection(detection)
                # Clear processed detections
                convoy_simulator.recent_detections = []
            
            # Broadcast updated status
            await broadcast_to_clients({
                "type": "convoy_update",
                "data": convoy_status
            })
            
            await asyncio.sleep(2.0)  # Update every 2 seconds
        except Exception as e:
            print(f"Error updating convoy status: {e}")
            break
    print("Convoy status updates stopped")

# Initialize MQTT
mqtt_client.on_connect = on_mqtt_connect
mqtt_client.on_message = on_mqtt_message

@app.on_event("startup")
async def startup_event():
    try:
        mqtt_client.connect(MQTT_BROKER, MQTT_PORT, 60)
        mqtt_client.loop_start()
        print("MQTT client started")
    except Exception as e:
        print(f"Failed to connect to MQTT broker: {e}")
        print("Continuing without MQTT - using simulator only")

@app.on_event("shutdown")
async def shutdown_event():
    mqtt_client.loop_stop()
    mqtt_client.disconnect()

# Pydantic models
class AOI(BaseModel):
    type: str
    geometry: dict
    priority: int = 5
    description: Optional[str] = None

class MissionStart(BaseModel):
    scenario: str
    vehicles: int = 3
    drones: int = 3

# REST API endpoints
@app.get("/api/convoy")
async def get_convoy_status():
    return convoy_status

@app.post("/api/aoi")
async def create_aoi(aoi: AOI):
    aoi_id = f"aoi-{len(aois) + 1:02d}"
    aois[aoi_id] = {
        "id": aoi_id,
        "type": aoi.type,
        "geometry": aoi.geometry,
        "priority": aoi.priority,
        "description": aoi.description,
        "created_at": datetime.utcnow().isoformat()
    }
    
    # Send AOI to mission planner
    mission_planner.add_aoi(aoi_id, aois[aoi_id])
    
    # Broadcast to clients
    await broadcast_to_clients({
        "type": "aoi_created",
        "data": aois[aoi_id]
    })
    
    return {"aoi_id": aoi_id, "status": "created"}

@app.get("/api/aoi")
async def get_aois():
    return list(aois.values())

@app.post("/api/mission/start")
async def start_mission(mission: MissionStart):
    try:
        # Start convoy simulation
        convoy_status["active"] = True
        
        # Get initial state from simulator
        sim_state = convoy_simulator.get_state()
        convoy_status["vehicles"] = sim_state["vehicles"]
        convoy_status["drones"] = sim_state["drones"]
        
        # Reset simulator
        convoy_simulator.reset_mission()
        
        print(f"Mission started: {mission.scenario}, vehicles: {len(convoy_status['vehicles'])}, drones: {len(convoy_status['drones'])}")
        
        # Send updates to frontend
        await broadcast_to_clients({
            "type": "mission_started",
            "data": {"scenario": mission.scenario, "status": "active"}
        })
        
        print(f"Broadcasting convoy update with {len(convoy_status['vehicles'])} vehicles")
        await broadcast_to_clients({
            "type": "convoy_update",
            "data": convoy_status
        })
        print("Convoy update sent")
        
        # Start continuous simulation
        asyncio.create_task(continuous_simulation())
        
        return {"status": "started", "scenario": mission.scenario}
    except Exception as e:
        print(f"Error starting mission: {e}")
        raise HTTPException(status_code=500, detail=str(e))



async def continuous_simulation():
    """Continuous convoy simulation with drag-drop support"""
    while convoy_status["active"]:
        # Update simulator
        sim_state = convoy_simulator.update()
        convoy_status["vehicles"] = sim_state["vehicles"]
        convoy_status["drones"] = sim_state["drones"]
        
        # Broadcast updates
        await broadcast_to_clients({
            "type": "convoy_update",
            "data": convoy_status
        })
        
        # Send obstacles and threats
        await broadcast_to_clients({
            "type": "obstacles_update",
            "data": sim_state["obstacles"]
        })
        
        await broadcast_to_clients({
            "type": "threats_update",
            "data": sim_state["threats"]
        })
        
        await asyncio.sleep(0.5)  # Faster updates

@app.post("/api/add-obstacle")
async def add_obstacle(request: dict):
    x = request.get("x", 0)
    y = request.get("y", 0)
    obstacle_type = request.get("type", "physical")
    convoy_simulator.add_obstacle(x, y, obstacle_type)
    return {"status": "obstacle_added", "x": x, "y": y}

@app.post("/api/add-threat")
async def add_threat(request: dict):
    x = request.get("x", 0)
    y = request.get("y", 0)
    threat_type = request.get("type", "ambush")
    convoy_simulator.add_threat(x, y, threat_type)
    return {"status": "threat_added", "x": x, "y": y}

@app.post("/api/eliminate-drone")
async def eliminate_drone(request: dict):
    drone_id = request.get("drone_id")
    convoy_simulator.eliminate_drone(drone_id)
    return {"status": "drone_eliminated", "drone_id": drone_id}

@app.post("/api/reset")
async def reset_mission():
    convoy_simulator.reset_mission()
    convoy_simulator.clear_all()
    
    # Reset convoy status
    convoy_status["active"] = False
    convoy_status["vehicles"] = []
    convoy_status["drones"] = []
    
    await broadcast_to_clients({
        "type": "mission_reset",
        "data": {"status": "reset"}
    })
    
    return {"status": "reset"}

@app.post("/api/mission/stop")
async def stop_mission():
    convoy_status["active"] = False
    convoy_status["vehicles"] = []
    convoy_status["drones"] = []
    
    await broadcast_to_clients({
        "type": "mission_stopped",
        "data": {"status": "stopped"}
    })
    
    return {"status": "stopped"}

@app.get("/api/heatmap")
async def get_heatmap():
    return heatmap_data

@app.get("/api/detections")
async def get_detections(limit: int = 100):
    return detections[-limit:]

@app.post("/api/command")
async def send_command(command: dict):
    """Send command to edge agent via MQTT"""
    try:
        drone_id = command.get("drone_id")
        topic = f"acps/commands/{drone_id}"
        
        mqtt_client.publish(topic, json.dumps(command))
        
        return {"status": "sent", "command": command}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metrics")
async def get_metrics():
    """Prometheus-style metrics"""
    metrics = [
        f"acps_detections_total {len(detections)}",
        f"acps_active_drones {len(convoy_status['drones'])}",
        f"acps_active_vehicles {len(convoy_status['vehicles'])}",
        f"acps_connected_clients {len(connected_clients)}",
        f"acps_mission_active {1 if convoy_status['active'] else 0}"
    ]
    return "\n".join(metrics)

# WebSocket endpoint
@app.websocket("/ws/telemetry")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    connected_clients.append(websocket)
    
    try:
        # Send initial state
        await websocket.send_text(json.dumps({
            "type": "initial_state",
            "convoy": convoy_status,
            "aois": list(aois.values()),
            "heatmap": heatmap_data
        }))
        
        print(f"WebSocket client connected. Convoy active: {convoy_status['active']}")
        
        while True:
            # Keep connection alive and handle incoming messages
            data = await websocket.receive_text()
            message = json.loads(data)
            
            if message.get("type") == "ping":
                await websocket.send_text(json.dumps({"type": "pong"}))
                
    except WebSocketDisconnect:
        connected_clients.remove(websocket)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)