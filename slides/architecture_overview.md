# ACPS Architecture Overview

## System Components

### Edge Layer (Per Drone)
- **Camera Capture**: Synthetic frame generation or AirSim integration
- **YOLO Inference**: YOLOv8n for CPU-friendly real-time detection
- **Local Decision Making**: Immediate safety responses for high-threat detections
- **MQTT Communication**: Publish detections and telemetry to cloud
- **Command Execution**: Receive and execute mission planner commands

### Cloud Layer (Centralized)
- **FastAPI Backend**: REST API and WebSocket server
- **Mission Planner**: A* pathfinding with dynamic threat cost mapping
- **Threat Fusion**: Aggregate detections into global threat heatmap
- **Convoy Simulator**: Vehicle and drone movement simulation
- **MQTT Broker**: Message routing between edge and cloud

### Dashboard Layer
- **React Frontend**: Real-time military-style dashboard
- **Leaflet Map**: Interactive map with convoy, drones, AOIs, and threat overlay
- **WebSocket Client**: Real-time updates from backend
- **AOI Drawing**: Polygon drawing tool for area inspection
- **Command Interface**: Accept/reject planner recommendations

## Data Flow

```
Edge Agent → MQTT → Backend → WebSocket → Dashboard
     ↑                ↓
   Commands ← Planner ← Fusion Service
```

### Message Types

1. **Detection Messages** (Edge → Cloud)
   - Bounding box coordinates
   - Object class and confidence
   - Geographic position
   - Priority score

2. **Telemetry Messages** (Edge → Cloud)
   - Drone position and status
   - Battery level and health
   - Performance metrics

3. **Command Messages** (Cloud → Edge)
   - Inspect AOI
   - Loiter at position
   - Adjust altitude
   - Return to tether

4. **Recommendation Messages** (Cloud → Dashboard)
   - Route changes
   - Formation adjustments
   - Speed modifications

## Threat Assessment Pipeline

1. **Detection Ingestion**: Receive detections from all drones
2. **Threat Scoring**: Apply class-based multipliers and confidence weighting
3. **Spatial Mapping**: Project detections onto 2D grid with Gaussian spread
4. **Cost Calculation**: Combine threat scores with base terrain costs
5. **Path Planning**: Run A* algorithm to find optimal routes
6. **Recommendation Generation**: Analyze threat levels and suggest actions

## Scalability Design

### Horizontal Scaling
- **Edge Agents**: Each drone runs independent agent
- **Load Balancing**: Multiple backend instances behind load balancer
- **Database Sharding**: Partition telemetry data by geographic region

### Vertical Scaling
- **GPU Acceleration**: Larger YOLO models for improved accuracy
- **Batch Processing**: Process multiple frames simultaneously
- **Memory Optimization**: Efficient threat map storage and updates

## Security Considerations

### Current Implementation (Demo)
- Simple token-based WebSocket authentication
- Local network communication
- No encryption for development speed

### Production Requirements
- **Device Identity**: X.509 certificates for each drone
- **Transport Security**: TLS 1.3 for all communications
- **Message Integrity**: HMAC signatures on critical commands
- **Access Control**: Role-based permissions for operators

## AWS Integration Points

### Phase 1: Core Services
- **IoT Core**: Device connectivity and message routing
- **IoT Greengrass**: Edge orchestration and local processing
- **S3**: Model storage and evidence archival

### Phase 2: AI/ML Pipeline
- **SageMaker**: Model training and optimization
- **Rekognition**: Enhanced threat classification
- **Panorama**: Optimized edge inference

### Phase 3: Analytics & Scale
- **Kinesis**: High-throughput telemetry streaming
- **QuickSight**: Mission analytics and reporting
- **Lambda**: Serverless event processing

## Performance Targets

### Latency Requirements
- **Detection to Alert**: < 2 seconds end-to-end
- **Command Execution**: < 1 second edge response
- **Dashboard Updates**: < 500ms WebSocket delivery

### Throughput Requirements
- **Detection Rate**: 10 detections/second per drone
- **Telemetry Rate**: 1 Hz position updates
- **Concurrent Users**: 10 operators per mission

### Reliability Requirements
- **Edge Autonomy**: 30 seconds offline operation
- **Message Delivery**: 99.9% success rate
- **System Uptime**: 99.5% availability

## Development Workflow

### Local Development
1. Start MQTT broker: `docker run eclipse-mosquitto`
2. Start backend: `python backend/app.py`
3. Start frontend: `npm run dev`
4. Spawn agents: `python scripts/spawn_convoy.py`

### Testing Pipeline
1. Unit tests: `pytest tests/unit/`
2. Integration tests: `pytest tests/integration/`
3. Performance tests: `python edge/benchmark_inference.py`
4. End-to-end scenarios: `python scripts/run_scenario.py`

### Deployment Pipeline
1. Build containers: `docker-compose build`
2. Run system tests: `docker-compose up --abort-on-container-exit`
3. Deploy to staging: `kubectl apply -f k8s/`
4. Production deployment: Blue/green with health checks