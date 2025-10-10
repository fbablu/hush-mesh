#!/usr/bin/env python3
from diagrams import Diagram, Cluster, Edge
from diagrams.aws.iot import IotCore, IotGreengrass
from diagrams.aws.analytics import KinesisDataStreams
from diagrams.aws.compute import Lambda, ECS, Fargate
from diagrams.aws.database import Dynamodb
from diagrams.aws.storage import S3
from diagrams.aws.ml import Sagemaker
from diagrams.aws.network import CloudFront, APIGateway, VPC
from diagrams.aws.security import Cognito, IAM, KMS
from diagrams.aws.management import Cloudwatch
from diagrams.custom import Custom

with Diagram("Maritime ACPS - AWS Architecture", show=False, direction="TB", filename="maritime_acps_architecture"):
    
    # Edge Layer
    with Cluster("Edge Devices (IoT Greengrass)"):
        edge1 = IotGreengrass("Edge Agent 1\n(Drone/Vessel)")
        edge2 = IotGreengrass("Edge Agent 2\n(Drone/Vessel)")
        edge3 = IotGreengrass("Edge Agent 3\n(Drone/Vessel)")
        edges = [edge1, edge2, edge3]
    
    # AWS Cloud Layer
    with Cluster("AWS Cloud"):
        
        # IoT & Ingestion
        with Cluster("IoT & Data Ingestion"):
            iot_core = IotCore("IoT Core\nMQTT Broker")
            kinesis = KinesisDataStreams("Kinesis Streams\nTelemetry")
        
        # Processing Layer
        with Cluster("Processing & Fusion"):
            lambda_fusion = Lambda("Lambda\nThreat Fusion")
            dynamodb = Dynamodb("DynamoDB\nEvent Storage")
        
        # ML Pipeline
        with Cluster("ML Pipeline"):
            sagemaker = Sagemaker("SageMaker\nTraining & Neo")
            s3_models = S3("S3\nModel Artifacts")
        
        # Backend Services
        with Cluster("Backend Services"):
            ecs = ECS("ECS Fargate\nMission Planner")
            api_gw = APIGateway("API Gateway")
        
        # Frontend & CDN
        with Cluster("Frontend"):
            cloudfront = CloudFront("CloudFront")
            s3_frontend = S3("S3\nStatic Website")
            cognito = Cognito("Cognito\nAuth")
        
        # Security & Monitoring
        with Cluster("Security & Monitoring"):
            kms = KMS("KMS\nEncryption")
            cloudwatch = Cloudwatch("CloudWatch\nLogs & Metrics")
    
    # Data Flow
    edges >> Edge(label="MQTT Telemetry") >> iot_core
    iot_core >> Edge(label="Stream") >> kinesis
    kinesis >> Edge(label="Process") >> lambda_fusion
    lambda_fusion >> Edge(label="Store Events") >> dynamodb
    lambda_fusion >> Edge(label="Trigger") >> ecs
    
    # ML Pipeline Flow
    sagemaker >> Edge(label="Deploy Models") >> s3_models
    s3_models >> Edge(label="Download") >> edges
    
    # Frontend Flow
    ecs >> api_gw
    api_gw >> cloudfront
    s3_frontend >> cloudfront
    cognito >> Edge(label="Auth") >> cloudfront
    
    # Security
    kms >> Edge(label="Encrypt", style="dashed") >> [dynamodb, s3_models, s3_frontend]
    cloudwatch >> Edge(label="Monitor", style="dashed") >> [iot_core, lambda_fusion, ecs]

print("✅ Architecture diagram generated: maritime_acps_architecture.png")
