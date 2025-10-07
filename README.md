# Autonomous Convoy Protection System (ACPS) - Hackathon MVP

A real-time convoy protection system with tethered drones, edge AI detection, and centralized mission planning.

## 🎯 Pitch (30 seconds)

ACPS provides autonomous convoy protection through a network of tethered drones performing real-time threat detection using edge AI (YOLO), with a centralized mission planner that dynamically adjusts convoy routes and drone formations. The system features a military-grade dashboard for operator oversight and manual intervention.

## 🚀 Quick Demo (3 minutes)

1. **Setup & Launch** (30s)
2. **Draw AOI & Start Mission** (60s) 
3. **Observe Threat Detection** (60s)
4. **Accept Planner Reroute** (30s)

## 📋 Demo Checklist

- [ ] Convoy movement with 3 vehicles
- [ ] 3 tethered drones following convoy
- [ ] Real-time threat detection (person, vehicle)
- [ ] Dynamic threat heatmap
- [ ] AOI drawing and inspection
- [ ] Planner reroute suggestions
- [ ] Network condition simulation

## 🛠 Setup & Installation

### Prerequisites
- Python 3.10+
- Node.js 18+
- Docker & Docker Compose

### Quick Start
```bash
# Clone and setup
git clone <repo-url>
cd acps-superprompt

# Install Python dependencies
pip install -r requirements.txt

# Install frontend dependencies
cd frontend && npm install && cd ..

# Start the full system
docker-compose up --build
```

### Manual Setup (Development)
```bash
# Terminal 1: Backend
cd backend
python -m uvicorn app:app --reload --port 8000

# Terminal 2: Frontend  
cd frontend
npm run dev

# Terminal 3: Edge Agents
python scripts/spawn_convoy.py --vehicles 3 --drones 3

# Terminal 4: Message Broker
docker run -p 1883:1883 eclipse-mosquitto
```

## 🎮 Demo Script

### 1. Launch System
```bash
docker-compose up --build
```
Wait for all services to start (backend, frontend, mosquitto, edge agents).

### 2. Open Dashboard
Navigate to `http://localhost:3000`

### 3. Start Convoy Simulation
```bash
python scripts/spawn_convoy.py --vehicles 3 --drones 3
```

### 4. Interactive Demo
1. **Draw AOI**: Click "Draw AOI" button, draw polygon on map
2. **Start Scenario**: Select "Ambush Test" and click "Start Mission"
3. **Observe**: Watch drone thumbnails, detection alerts, threat heatmap
4. **Accept Reroute**: When planner suggests route change, click "Accept"
5. **Network Test**: Toggle "Simulate Packet Loss" to test edge autonomy

### 5. Test Scenarios
```bash
# Scenario A: Ambush detection
python scripts/run_scenario.py --scenario ambush

# Scenario B: Weather conditions  
python scripts/run_scenario.py --scenario weather

# Scenario C: Network issues
python scripts/run_scenario.py --scenario connectivity
```

## 🏗 Architecture

### Edge Components (Per Drone)
- **Camera capture** and preprocessing
- **YOLO inference** (YOLOv8n for CPU compatibility)
- **Local safety rules** and immediate responses
- **Offline buffering** for network outages
- **Command execution** (inspect AOI, loiter, return)

### Cloud Components (Centralized)
- **Telemetry fusion** from all drones
- **Global threat mapping** and cost calculation
- **Mission planning** with A* pathfinding
- **Operator dashboard** with manual overrides
- **Evidence storage** and replay capability

### Message Flow
```
Edge Agent → MQTT → Backend → WebSocket → Dashboard
     ↑                ↓
   Commands ← Planner ← Fusion Service
```

## 🔧 Configuration

### Development Mode (CPU-only)
```bash
export USE_MINIMAL=true
python backend/app.py
```

### GPU Mode (Full YOLO)
```bash
export USE_GPU=true
export YOLO_MODEL=yolov8s.pt
python backend/app.py
```

### Network Simulation
```bash
# Simulate 40% packet loss
export PACKET_LOSS=0.4
python edge/agent.py
```

## 🧪 Testing

### Unit Tests
```bash
python -m pytest tests/unit/
```

### Integration Test
```bash
python -m pytest tests/integration/test_end_to_end.py
```

### Performance Benchmark
```bash
python edge/benchmark_inference.py
```

## 📊 Metrics & Monitoring

Access metrics at `http://localhost:8000/metrics`

- Detections per second
- Planner response latency  
- Message drop percentage
- Edge agent health status

## 🔮 AWS Integration Roadmap

### Phase 1: Device Management
- **AWS IoT Core** for device identity and secure connectivity
- **AWS IoT Greengrass** for edge orchestration and OTA updates
- **AWS Systems Manager** for fleet configuration management

### Phase 2: AI/ML Pipeline  
- **Amazon SageMaker** for model retraining with field data
- **AWS Panorama** for optimized edge inference
- **Amazon Rekognition** for enhanced threat classification

### Phase 3: Scale & Analytics
- **Amazon Kinesis** for high-throughput telemetry streaming
- **Amazon S3** for evidence storage and compliance
- **Amazon QuickSight** for mission analytics and reporting

## 🎨 UI Theme

- **Background**: `#0b0b0b` (black)
- **Accent**: `#FFD400` (yellow)  
- **Text**: `#ffffff` (white)
- **Muted**: `#bfbfbf` (grey)

### Keyboard Shortcuts
- `D` - Toggle AOI drawing mode
- `S` - Start/pause simulation
- `L` - Toggle log panel
- `M` - Toggle map layers

## 📁 Project Structure

```
acps-superprompt/
├── README.md
├── docker-compose.yml
├── requirements.txt
├── backend/
│   ├── app.py
│   ├── planner/
│   └── sim/
├── edge/
│   ├── agent.py
│   ├── camera_adapter.py
│   └── yolo_infer.py
├── frontend/
│   ├── src/
│   └── tailwind.config.js
├── planner/
├── sim_configs/
├── scripts/
├── tests/
├── models/
└── slides/
```

## 🚨 Troubleshooting

### Common Issues

**Port conflicts**: Change ports in `docker-compose.yml`
**YOLO model download**: Run `python scripts/download_models.py`
**Permission errors**: Ensure Docker has proper permissions
**Memory issues**: Reduce batch size in `edge/config.py`

### Support

Check logs: `docker-compose logs -f [service-name]`
Reset state: `docker-compose down -v && docker-compose up --build`

---

**Ready for hackathon demo!** 🎖️