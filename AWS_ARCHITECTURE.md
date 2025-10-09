# Maritime ACPS - AWS Architecture Diagram

## Architecture Overview

This document provides the AWS architecture for the Maritime Autonomous Convoy Protection System (ACPS).

## Mermaid Diagram

```mermaid
graph TB
    subgraph Edge["Edge Layer - IoT Greengrass"]
        E1[Edge Agent 1<br/>Drone/Vessel<br/>ML Inference]
        E2[Edge Agent 2<br/>Drone/Vessel<br/>ML Inference]
        E3[Edge Agent 3<br/>Drone/Vessel<br/>ML Inference]
    end

    subgraph Cloud["AWS Cloud"]
        subgraph IoT["IoT & Ingestion"]
            IC[IoT Core<br/>MQTT Broker]
            KS[Kinesis Streams<br/>Real-time Telemetry]
        end

        subgraph Processing["Processing & Fusion"]
            LF[Lambda<br/>Threat Fusion]
            DB[(DynamoDB<br/>Event Storage)]
        end

        subgraph ML["ML Pipeline"]
            SM[SageMaker<br/>Training & Neo]
            S3M[S3<br/>Model Artifacts]
        end

        subgraph Backend["Backend Services"]
            ECS[ECS Fargate<br/>Mission Planner<br/>Path Planning]
            API[API Gateway<br/>REST API]
        end

        subgraph Frontend["Frontend & CDN"]
            CF[CloudFront<br/>CDN]
            S3F[S3<br/>Static Website]
            COG[Cognito<br/>Authentication]
        end

        subgraph Security["Security & Monitoring"]
            KMS[KMS<br/>Encryption]
            CW[CloudWatch<br/>Logs & Metrics]
        end
    end

    subgraph Users["End Users"]
        U1[Operators<br/>Dashboard]
    end

    %% Data Flow
    E1 -->|MQTT Telemetry| IC
    E2 -->|MQTT Telemetry| IC
    E3 -->|MQTT Telemetry| IC
    IC -->|Stream Data| KS
    KS -->|Process Events| LF
    LF -->|Store Events| DB
    LF -->|Trigger Replanning| ECS
    
    %% ML Pipeline
    SM -->|Deploy Models| S3M
    S3M -.->|Download Models| E1
    S3M -.->|Download Models| E2
    S3M -.->|Download Models| E3
    
    %% Backend to Frontend
    ECS --> API
    API --> CF
    S3F --> CF
    COG -->|Auth| CF
    CF --> U1
    
    %% Security
    KMS -.->|Encrypt| DB
    KMS -.->|Encrypt| S3M
    KMS -.->|Encrypt| S3F
    CW -.->|Monitor| IC
    CW -.->|Monitor| LF
    CW -.->|Monitor| ECS

    style Edge fill:#e1f5ff
    style Cloud fill:#fff4e1
    style Users fill:#e8f5e9
```

## Component Details

### 1. Edge Layer (IoT Greengrass)
- **IoT Greengrass v2**: Edge runtime for local ML inference
- **Edge Agents**: Deployed on drones/vessels with sensors
- **Local ML Models**: SageMaker Neo optimized models
- **Sensors**: Camera, radar, acoustic, RF sensors
- **Local Processing**: <100ms inference latency

### 2. IoT & Data Ingestion
- **AWS IoT Core**: MQTT broker for device connectivity
- **IoT Rules**: Route telemetry to Kinesis
- **Kinesis Data Streams**: Real-time data ingestion
- **Device Management**: Certificate-based authentication

### 3. Processing & Fusion
- **Lambda Functions**: 
  - Threat fusion from multiple sensors
  - Event processing and enrichment
  - Alert generation
- **DynamoDB**: 
  - Event storage with TTL
  - Threat history and tracking
  - Query patterns for dashboard

### 4. ML Pipeline
- **SageMaker Training**: PyTorch threat detection models
- **SageMaker Processing**: Synthetic data generation
- **SageMaker Neo**: Edge optimization for ARM/Jetson
- **S3 Model Registry**: Versioned model artifacts
- **Model Deployment**: Automated to edge devices

### 5. Backend Services
- **ECS Fargate**: 
  - Mission planner service
  - A* path planning with threat avoidance
  - Multi-route optimization
- **API Gateway**: 
  - REST API for dashboard
  - WebSocket for real-time updates
- **Auto-scaling**: Based on threat levels

### 6. Frontend & CDN
- **CloudFront**: Global CDN for low latency
- **S3 Static Hosting**: React dashboard
- **Cognito**: User authentication and authorization
- **WebSocket**: Real-time convoy tracking

### 7. Security & Monitoring
- **KMS**: Encryption at rest for all data stores
- **IAM**: Least privilege access policies
- **CloudWatch**: 
  - Logs aggregation
  - Metrics and alarms
  - Dashboard monitoring
- **VPC**: Private subnets for backend services

## Data Flow

1. **Edge Detection** (0-100ms):
   - Sensors capture maritime data
   - Local ML inference on Greengrass
   - Threat classification and scoring

2. **Cloud Ingestion** (100-500ms):
   - MQTT publish to IoT Core
   - IoT Rules route to Kinesis
   - Stream processing begins

3. **Threat Fusion** (500ms-2s):
   - Lambda processes multi-sensor data
   - Correlation and fusion algorithms
   - Store events in DynamoDB

4. **Mission Planning** (2-5s):
   - ECS service receives threat updates
   - A* path planning with threat avoidance
   - Generate alternative routes

5. **Dashboard Update** (5-10s):
   - WebSocket push to connected clients
   - Real-time map visualization
   - Alert notifications

## AWS Services Used

| Service | Purpose | Configuration |
|---------|---------|---------------|
| IoT Core | Device connectivity | MQTT, X.509 certs |
| IoT Greengrass | Edge runtime | v2, ML components |
| Kinesis Streams | Data ingestion | 2 shards, 24h retention |
| Lambda | Event processing | Python 3.11, 512MB |
| DynamoDB | Event storage | On-demand, TTL enabled |
| SageMaker | ML training | ml.p3.2xlarge (training) |
| SageMaker Neo | Edge optimization | ARM64, Jetson targets |
| ECS Fargate | Backend services | 2 vCPU, 4GB RAM |
| API Gateway | REST/WebSocket | Regional endpoint |
| S3 | Storage | Versioning, encryption |
| CloudFront | CDN | HTTPS only |
| Cognito | Authentication | User pools |
| KMS | Encryption | Customer managed keys |
| CloudWatch | Monitoring | 7-day log retention |
| VPC | Networking | Private subnets, NAT |

## Deployment Regions

- **Primary**: us-east-1 (N. Virginia)
- **Secondary**: eu-west-1 (Ireland) for international ops

## Cost Optimization

- Spot instances for SageMaker training
- S3 lifecycle policies (Glacier after 90 days)
- DynamoDB on-demand billing
- CloudWatch log retention limits
- Auto-scaling based on load

## Security Best Practices

✅ Encryption at rest (KMS)
✅ Encryption in transit (TLS 1.2+)
✅ Least privilege IAM policies
✅ VPC isolation for backend
✅ Certificate-based device auth
✅ Cognito MFA for operators
✅ CloudTrail audit logging
✅ Security group restrictions

## Performance Targets

- Edge inference: <100ms
- Cloud fusion: <5s end-to-end
- Dashboard updates: <10s
- Path replanning: <30s
- System availability: 99.9%

## Human-in-the-Loop Safety

⚠️ **CRITICAL**: All engagement decisions require human authorization
- System provides recommendations only
- Defensive operations only
- Explicit operator approval required
- Full audit trail maintained

## How to Deploy

See [deploy_instructions.md](deploy_instructions.md) for step-by-step deployment guide.

## Monitoring Dashboard

Key metrics to monitor:
- Device connectivity status
- Inference latency (edge)
- Kinesis throughput
- Lambda error rates
- DynamoDB read/write capacity
- ECS service health
- API Gateway latency
- CloudFront cache hit ratio
