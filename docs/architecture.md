# Maritime ACPS Architecture

## System Overview

The Maritime Autonomous Convoy Protection System (ACPS) is a defensive AWS-first solution for real-time threat detection and convoy protection in maritime environments.

## Architecture Diagram

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Edge Devices  │    │   AWS Cloud     │    │   Dashboard     │
│                 │    │                 │    │                 │
│ ┌─────────────┐ │    │ ┌─────────────┐ │    │ ┌─────────────┐ │
│ │ Greengrass  │ │    │ │ IoT Core    │ │    │ │ React App   │ │
│ │ + ML Model  │◄┼────┼►│ + Rules     │ │    │ │ + Cognito   │ │
│ └─────────────┘ │    │ └─────────────┘ │    │ └─────────────┘ │
│                 │    │        │        │    │        ▲        │
│ ┌─────────────┐ │    │        ▼        │    │        │        │
│ │ Sensors +   │ │    │ ┌─────────────┐ │    │        │        │
│ │ Simulator   │ │    │ │ Kinesis     │ │    │        │        │
│ └─────────────┘ │    │ │ Streams     │ │    │        │        │
└─────────────────┘    │ └─────────────┘ │    │        │        │
                       │        │        │    │        │        │
                       │        ▼        │    │        │        │
                       │ ┌─────────────┐ │    │        │        │
                       │ │ Lambda      │ │    │        │        │
                       │ │ Fusion      │ │    │        │        │
                       │ └─────────────┘ │    │        │        │
                       │        │        │    │        │        │
                       │        ▼        │    │        │        │
                       │ ┌─────────────┐ │    │        │        │
                       │ │ DynamoDB    │ │    │        │        │
                       │ │ Events      │ │    │        │        │
                       │ └─────────────┘ │    │        │        │
                       │        │        │    │        │        │
                       │        ▼        │    │        │        │
                       │ ┌─────────────┐ │    │        │        │
                       │ │ ECS Fargate │ │    │        │        │
                       │ │ Planner     │◄┼────┼────────┘        │
                       │ └─────────────┘ │    │                 │
                       └─────────────────┘    └─────────────────┘
```

## Component Details

### Edge Layer
- **IoT Greengrass v2**: Edge runtime for ML inference
- **SageMaker Neo**: Optimized models for ARM/Jetson
- **Local Sensors**: Camera, radar, acoustic, RF sensors
- **Simulator**: Maritime scenario generator

### Cloud Layer
- **IoT Core**: MQTT broker and device management
- **Kinesis Streams**: Real-time telemetry ingestion
- **Lambda**: Threat fusion and processing
- **DynamoDB**: Event storage and indexing
- **ECS Fargate**: Mission planner service
- **S3**: Data lake and model artifacts

### ML Pipeline
- **SageMaker Processing**: Synthetic data generation
- **SageMaker Training**: PyTorch threat detection models
- **SageMaker Neo**: Edge optimization
- **Model Registry**: Version management

### Frontend
- **React Dashboard**: Real-time threat visualization
- **Cognito**: Authentication and authorization
- **CloudFront**: Global content delivery
- **WebSocket**: Real-time updates

## Data Flow

1. **Sensor Data**: Edge devices collect maritime sensor data
2. **Local Inference**: Greengrass runs ML models for threat detection
3. **Cloud Ingestion**: Detections published to IoT Core → Kinesis
4. **Fusion**: Lambda processes and fuses multi-sensor detections
5. **Storage**: Events stored in DynamoDB for analysis
6. **Planning**: ECS service runs A* path planning with threat avoidance
7. **Visualization**: Dashboard displays real-time threats and convoy status
8. **Human Authorization**: All engagement decisions require operator approval

## Security Architecture

### Defense in Depth
- **Network**: VPC with private subnets, security groups
- **Identity**: IAM roles with least privilege
- **Data**: KMS encryption for S3 and DynamoDB
- **Application**: Cognito authentication, API rate limiting

### Human-in-the-Loop Safety
- All kinetic responses require human authorization
- System provides recommendations only
- Explicit defensive-only operation mode
- Audit trail for all decisions

## Threat Model

### Maritime Threats Detected
- Small fast craft (piracy, interdiction)
- Suspicious loitering vessels
- AIS spoofing and unregistered vessels
- Drone/UAV overwater threats
- Floating mines and obstacles
- Collision risks
- Acoustic signatures (gunshots, engines)

### Detection Methods
- Computer vision (camera feeds)
- Radar contact analysis
- AIS data correlation
- Acoustic signature matching
- RF signal analysis
- Multi-sensor fusion

## Scalability

### Horizontal Scaling
- Multiple edge devices per convoy
- Auto-scaling ECS services
- Kinesis shard scaling
- DynamoDB on-demand billing

### Performance Targets
- Edge inference: <100ms latency
- Cloud fusion: <5s end-to-end
- Dashboard updates: <10s
- Path replanning: <30s

## Cost Optimization

### Development Environment
- Spot instances for SageMaker training
- Minimal instance types
- S3 lifecycle policies
- CloudWatch log retention limits

### Production Scaling
- Reserved instances for predictable workloads
- Auto-scaling based on threat levels
- Data archival to Glacier
- Cost monitoring and alerts

## Deployment Regions

### Primary: US-East-1
- Full service availability
- Lowest latency to operations center

### Secondary: EU-West-1
- International deployments
- Data sovereignty compliance

## Monitoring and Alerting

### CloudWatch Metrics
- IoT device connectivity
- ML model accuracy drift
- Kinesis throughput
- Lambda error rates
- ECS service health

### Alerts
- Device disconnections
- High threat confidence detections
- System failures
- Cost threshold breaches

## Compliance and Governance

### Data Handling
- No PII in synthetic datasets
- Evidence data encrypted at rest
- Audit logs for all actions
- Data retention policies

### Safety Requirements
- Human authorization for all engagement
- Defensive operations only
- Clear escalation procedures
- Regular safety audits