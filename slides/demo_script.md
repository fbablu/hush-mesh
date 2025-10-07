# ACPS Demo Script (3 Minutes)

## Setup (30 seconds)

### Pre-Demo Checklist
- [ ] Docker running
- [ ] Terminal windows prepared
- [ ] Browser tabs open
- [ ] Demo scenario selected

### Opening Statement
> "ACPS provides autonomous convoy protection through AI-powered drone surveillance and intelligent mission planning. Let me show you a live simulation."

### Launch Commands
```bash
# Terminal 1: Start system
docker-compose up --build

# Terminal 2: Spawn convoy
python scripts/spawn_convoy.py --vehicles 3 --drones 3
```

**Talking Points:**
- "Starting our distributed system with edge AI agents"
- "3 vehicles, 3 tethered drones, real-time threat detection"

## Demo Flow (2 minutes)

### 1. Dashboard Overview (30 seconds)
Navigate to `http://localhost:3000`

**Show:**
- Military-style black/yellow interface
- Real-time convoy and drone positions on map
- System status indicators (connected, active drones)

**Script:**
> "This is our operator dashboard. You can see our convoy of 3 vehicles with tethered drones providing overwatch. The system is processing live camera feeds using YOLO AI detection."

### 2. AOI Creation (30 seconds)
**Actions:**
1. Click "DRAW AOI" button
2. Draw polygon on map around convoy route
3. Press Enter to complete

**Script:**
> "Operators can designate Areas of Interest for focused surveillance. I'm drawing an AOI where we suspect potential threats."

### 3. Scenario Execution (45 seconds)
**Actions:**
1. Select "Ambush Test" scenario
2. Click "START MISSION"
3. Watch for detections in right panel

**Script:**
> "Starting our ambush scenario. The drones are now actively scanning for threats using edge AI. Watch the detection panel - we're getting real-time person and vehicle detections with confidence scores."

**Point out:**
- Detection alerts appearing
- Threat heatmap updating on map
- Drone status indicators

### 4. Planner Response (30 seconds)
**Actions:**
1. Wait for recommendation to appear
2. Click "ACCEPT" on route recommendation

**Script:**
> "Our AI mission planner has analyzed the threat pattern and recommends a route change. The system uses A-star pathfinding with dynamic threat costs. I'll accept this recommendation."

**Show:**
- Recommendation details (threat level, suggested action)
- Real-time decision making

### 5. Network Resilience (15 seconds)
**Actions:**
1. Toggle "Simulate Packet Loss"
2. Show continued operation

**Script:**
> "Even with network issues, our edge agents continue autonomous operation, buffering data and making local safety decisions."

## Closing (30 seconds)

### Key Achievements Demonstrated
✅ **Real-time AI Detection**: YOLO inference on synthetic camera feeds
✅ **Intelligent Planning**: Dynamic route optimization based on threat analysis  
✅ **Edge Autonomy**: Continued operation during network outages
✅ **Operator Control**: Interactive AOI creation and mission oversight
✅ **Scalable Architecture**: Distributed system ready for AWS integration

### Next Steps Statement
> "This prototype demonstrates core convoy protection capabilities. Next steps include AWS integration with IoT Core for device management, SageMaker for model optimization, and Kinesis for high-scale telemetry processing."

## Technical Backup Slides

### If Asked About Performance
- **Detection Latency**: < 100ms per frame
- **End-to-end Response**: < 2 seconds threat to alert
- **Throughput**: 10 FPS per drone, 3 drones = 30 detections/second
- **Memory Usage**: < 500MB per edge agent

### If Asked About Scalability
- **Horizontal**: Add more drones by spawning additional edge agents
- **Vertical**: Upgrade to GPU inference for better accuracy
- **Cloud**: Ready for AWS IoT Greengrass deployment

### If Asked About Security
- **Current**: Development mode with basic authentication
- **Production**: X.509 certificates, TLS encryption, RBAC
- **AWS Integration**: IAM roles, VPC isolation, CloudTrail auditing

## Troubleshooting

### Common Issues
- **Port conflicts**: Change ports in docker-compose.yml
- **Slow startup**: Wait 30 seconds for all services
- **No detections**: Check edge agent logs, restart if needed
- **Map not loading**: Check network connectivity

### Recovery Commands
```bash
# Reset system
docker-compose down -v
docker-compose up --build

# Check logs
docker-compose logs -f backend
docker-compose logs -f edge-agent-1

# Manual agent spawn
python scripts/spawn_convoy.py --vehicles 1 --drones 1
```

## Demo Variations

### 2-Minute Version
- Skip AOI creation
- Focus on detection and planning
- Quick network resilience demo

### 5-Minute Version
- Show all three scenarios
- Demonstrate metrics endpoint
- Explain AWS integration roadmap
- Show code architecture briefly

### Technical Deep-Dive (10+ minutes)
- Live code walkthrough
- Performance benchmarking
- Custom model training discussion
- Production deployment considerations