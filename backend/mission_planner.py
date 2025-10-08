#!/usr/bin/env python3
import asyncio
import json
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import boto3
from datetime import datetime
import heapq
import math
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Maritime Mission Planner")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# AWS clients
dynamodb = boto3.resource('dynamodb')
iot_client = boto3.client('iot-data')

class AStarPlanner:
    def __init__(self, grid_size=100, cell_size_m=100):
        self.grid_size = grid_size
        self.cell_size_m = cell_size_m
        self.threat_grid = np.zeros((grid_size, grid_size))
        
    def update_threat_grid(self, threats):
        """Update threat costmap with new detections"""
        self.threat_grid.fill(0.1)  # Base cost
        
        for threat in threats:
            if 'location' in threat:
                lat = threat['location'].get('lat', 0)
                lon = threat['location'].get('lon', 0)
                
                # Convert to grid coordinates
                grid_x, grid_y = self.lat_lon_to_grid(lat, lon)
                
                if 0 <= grid_x < self.grid_size and 0 <= grid_y < self.grid_size:
                    # Add threat cost with radius
                    threat_cost = threat.get('fusion_score', 0.5)
                    radius = 5  # Grid cells
                    
                    for dx in range(-radius, radius + 1):
                        for dy in range(-radius, radius + 1):
                            nx, ny = grid_x + dx, grid_y + dy
                            if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
                                distance = math.sqrt(dx*dx + dy*dy)
                                if distance <= radius:
                                    cost = threat_cost * (1 - distance / radius)
                                    self.threat_grid[nx, ny] += cost
    
    def lat_lon_to_grid(self, lat, lon):
        """Convert lat/lon to grid coordinates"""
        # Simple conversion for demo (assumes small area)
        base_lat, base_lon = 25.7617, -80.1918
        
        x = int((lon - base_lon) * 111000 / self.cell_size_m) + self.grid_size // 2
        y = int((lat - base_lat) * 111000 / self.cell_size_m) + self.grid_size // 2
        
        return x, y
    
    def grid_to_lat_lon(self, x, y):
        """Convert grid coordinates to lat/lon"""
        base_lat, base_lon = 25.7617, -80.1918
        
        lon = base_lon + (x - self.grid_size // 2) * self.cell_size_m / 111000
        lat = base_lat + (y - self.grid_size // 2) * self.cell_size_m / 111000
        
        return lat, lon
    
    def plan_path(self, start_lat, start_lon, goal_lat, goal_lon):
        """A* path planning with threat avoidance"""
        
        start_x, start_y = self.lat_lon_to_grid(start_lat, start_lon)
        goal_x, goal_y = self.lat_lon_to_grid(goal_lat, goal_lon)
        
        # A* algorithm
        open_set = [(0, start_x, start_y)]
        came_from = {}
        g_score = {(start_x, start_y): 0}
        f_score = {(start_x, start_y): self.heuristic(start_x, start_y, goal_x, goal_y)}
        
        while open_set:
            current_f, current_x, current_y = heapq.heappop(open_set)
            
            if current_x == goal_x and current_y == goal_y:
                # Reconstruct path
                path = []
                while (current_x, current_y) in came_from:
                    lat, lon = self.grid_to_lat_lon(current_x, current_y)
                    path.append({'lat': lat, 'lon': lon})
                    current_x, current_y = came_from[(current_x, current_y)]
                
                path.reverse()
                return path
            
            # Check neighbors
            for dx, dy in [(-1,0), (1,0), (0,-1), (0,1), (-1,-1), (-1,1), (1,-1), (1,1)]:
                neighbor_x, neighbor_y = current_x + dx, current_y + dy
                
                if (0 <= neighbor_x < self.grid_size and 
                    0 <= neighbor_y < self.grid_size):
                    
                    # Calculate cost including threat
                    base_cost = math.sqrt(dx*dx + dy*dy)
                    threat_cost = self.threat_grid[neighbor_x, neighbor_y]
                    total_cost = base_cost * (1 + threat_cost * 10)  # Threat penalty
                    
                    tentative_g = g_score[(current_x, current_y)] + total_cost
                    
                    if (neighbor_x, neighbor_y) not in g_score or tentative_g < g_score[(neighbor_x, neighbor_y)]:
                        came_from[(neighbor_x, neighbor_y)] = (current_x, current_y)
                        g_score[(neighbor_x, neighbor_y)] = tentative_g
                        f_score[(neighbor_x, neighbor_y)] = tentative_g + self.heuristic(neighbor_x, neighbor_y, goal_x, goal_y)
                        
                        heapq.heappush(open_set, (f_score[(neighbor_x, neighbor_y)], neighbor_x, neighbor_y))
        
        return []  # No path found
    
    def heuristic(self, x1, y1, x2, y2):
        """Heuristic function for A*"""
        return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

# Global planner instance
planner = AStarPlanner()

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}

@app.get("/threats")
async def get_active_threats():
    """Get current active threats"""
    try:
        events_table = dynamodb.Table('maritime-events')
        
        # Scan for active threats (in production, use better indexing)
        response = events_table.scan(
            FilterExpression='#status = :status',
            ExpressionAttributeNames={'#status': 'status'},
            ExpressionAttributeValues={':status': 'active'}
        )
        
        return {"threats": response['Items']}
        
    except Exception as e:
        logger.error(f"Error fetching threats: {e}")
        return {"threats": [], "error": str(e)}

@app.post("/plan")
async def plan_route(request: dict):
    """Generate new route plan avoiding threats"""
    try:
        start_lat = request['start']['lat']
        start_lon = request['start']['lon']
        goal_lat = request['goal']['lat']
        goal_lon = request['goal']['lon']
        
        # Get current threats
        threats_response = await get_active_threats()
        threats = threats_response['threats']
        
        # Update threat grid
        planner.update_threat_grid(threats)
        
        # Plan path
        path = planner.plan_path(start_lat, start_lon, goal_lat, goal_lon)
        
        plan = {
            'plan_id': f"plan_{int(datetime.utcnow().timestamp())}",
            'timestamp': datetime.utcnow().isoformat(),
            'waypoints': path,
            'threats_considered': len(threats),
            'status': 'pending_approval'
        }
        
        return plan
        
    except Exception as e:
        logger.error(f"Error planning route: {e}")
        return {"error": str(e)}

@app.post("/approve_plan")
async def approve_plan(request: dict):
    """Approve and execute route plan"""
    try:
        plan_id = request['plan_id']
        waypoints = request['waypoints']
        
        # Send to convoy via IoT
        command = {
            'command': 'update_route',
            'plan_id': plan_id,
            'waypoints': waypoints,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        iot_client.publish(
            topic='maritime/commands/convoy_01',
            qos=1,
            payload=json.dumps(command)
        )
        
        logger.info(f"Route plan {plan_id} approved and sent to convoy")
        
        return {"status": "approved", "plan_id": plan_id}
        
    except Exception as e:
        logger.error(f"Error approving plan: {e}")
        return {"error": str(e)}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket for real-time updates"""
    await websocket.accept()
    
    try:
        while True:
            # Send periodic updates
            threats_response = await get_active_threats()
            
            update = {
                'type': 'threat_update',
                'timestamp': datetime.utcnow().isoformat(),
                'threats': threats_response['threats']
            }
            
            await websocket.send_text(json.dumps(update))
            await asyncio.sleep(5)  # Update every 5 seconds
            
    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)