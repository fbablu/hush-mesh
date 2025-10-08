# Maritime ACPS Deployment Instructions

## Prerequisites

1. AWS CLI configured with admin permissions
2. Node.js 18+ and npm
3. Python 3.9+ with pip
4. Docker installed
5. AWS CDK v2 installed: `npm install -g aws-cdk`

## Step-by-Step Deployment

### 1. Infrastructure Setup

```bash
cd infrastructure
npm install
export AWS_REGION=us-east-1
export PROJECT_NAME=maritime-acps
cdk bootstrap
cdk deploy --parameters ProjectName=$PROJECT_NAME
```

### 2. Generate Synthetic Data

```bash
cd ../synth
pip install -r requirements.txt
python generate_maritime_data.py --scenarios piracy_ambush,swarm_interdiction --output-bucket maritime-acps-data-$AWS_ACCOUNT_ID
```

### 3. Train ML Model

```bash
cd ../ml
pip install -r requirements.txt
python train_model.py --data-bucket maritime-acps-data-$AWS_ACCOUNT_ID --model-name maritime-threat-detector
```

### 4. Deploy Edge Components

```bash
cd ../edge
# Create IoT Thing and certificates
python setup_iot_device.py --device-name convoy-edge-01
# Deploy Greengrass component
aws greengrassv2 create-deployment --cli-input-json file://deployment.json
```

### 5. Start Backend Services

```bash
cd ../backend
docker build -t maritime-planner .
# Deploy to ECS (automated via CDK)
```

### 6. Deploy Frontend

```bash
cd ../frontend
npm install
npm run build
aws s3 sync build/ s3://maritime-acps-frontend-$AWS_ACCOUNT_ID
```

### 7. Run Simulator

```bash
cd ../simulator
python maritime_sim.py --scenario piracy_ambush --duration 300
```

### 8. Access Dashboard

Open: `https://d1234567890.cloudfront.net` (from CDK output)

## Verification Checklist

- [ ] S3 buckets created and accessible
- [ ] SageMaker training job completed successfully
- [ ] IoT device connected and publishing telemetry
- [ ] Kinesis stream receiving data
- [ ] Dashboard loads and shows convoy position
- [ ] Threat detection alerts appear
- [ ] Path replanning triggers on threat detection

## Cleanup

```bash
cd infrastructure
cdk destroy
aws s3 rm s3://maritime-acps-data-$AWS_ACCOUNT_ID --recursive
aws s3 rm s3://maritime-acps-frontend-$AWS_ACCOUNT_ID --recursive
```

## Troubleshooting

- **SageMaker permissions**: Ensure SageMaker execution role has S3 access
- **IoT connectivity**: Check security groups allow MQTT (port 8883)
- **Greengrass**: Verify device has internet access for component downloads
- **Dashboard 403**: Check Cognito user pool configuration

## Cost Optimization

- Use Spot instances for SageMaker training
- Set S3 lifecycle policies for data archival
- Configure CloudWatch log retention (7 days recommended)
- Use minimal instance types for development