# Maritime Autonomous Convoy Protection System (ACPS)

AWS-first maritime threat detection and convoy protection system using ML, edge computing, and real-time path planning.

## Quick Start - Local Development

### 1. Install Dependencies
```bash
pip install flask flask-cors fastapi uvicorn paho-mqtt
```

### 2. Start All Servers
```bash
# Terminal 1: Backend API Server (Port 8000)
cd backend
python3 app.py

# Terminal 2: ML API Server (Port 9000) 
cd ml
python3 ml_api_server.py

# Terminal 3: Demo Server (Port 8081)
cd demo
python3 -m http.server 8081
```

### 3. Verify Servers Running
```bash
# Test Backend (Port 8000)
curl http://localhost:8000/api/convoy

# Test ML API (Port 9000)
curl http://localhost:9000/health

# Test Demo Server (Port 8081)
curl http://localhost:8081/
```

### 4. Access Demos
- **Enhanced Multi-Route Demo**: http://localhost:8081/enhanced_multi_route.html
- **ML Test Page**: http://localhost:8081/test_ml.html
- **Simple CLI Demo**: `python3 simple_demo.py`

## Server Endpoints

### Backend API (Port 8000)
- `GET /api/convoy` - Convoy status
- `POST /api/mission/start` - Start mission
- `WebSocket /ws/telemetry` - Real-time updates

### ML API (Port 9000)
- `GET /health` - Health check
- `POST /predict` - Threat prediction
- `GET /` - Server status

### Demo Server (Port 8081)
- Static file server for HTML demos

## Troubleshooting

### Reset and Restart All Servers
```bash
# Kill all processes
pkill -f "python.*server" && pkill -f "python.*app" && pkill -f "http.server"

# Restart (run each in separate terminal)
cd backend && python3 app.py
cd ml && python3 ml_api_server.py  
cd demo && python3 -m http.server 8081
```

### Check Running Processes
```bash
netstat -tlnp | grep -E ":(8000|8081|9000)"
```

## Architecture

- **Edge**: IoT Greengrass + SageMaker Neo optimized models
- **Cloud**: Kinesis + Lambda + DynamoDB + ECS Fargate
- **ML**: SageMaker training pipeline with maritime threat detection
- **UI**: React dashboard with real-time convoy tracking

## Security Notice

⚠️ **DEFENSIVE SYSTEM ONLY** - All engagement decisions require human authorization. No automated kinetic responses.

See `docs/deploy_instructions.md` for complete deployment guide.