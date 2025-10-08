# Maritime Autonomous Convoy Protection System (ACPS)

AWS-first maritime threat detection and convoy protection system using ML, edge computing, and real-time path planning.

## Quick Start

1. **Prerequisites**: AWS CLI configured, CDK installed, Docker
2. **Deploy**: `cd infrastructure && npm install && cdk deploy`
3. **Generate Data**: `python synth/generate_maritime_data.py`
4. **Train Model**: `python ml/train_model.py`
5. **Run Demo**: `python simulator/maritime_sim.py`

## Architecture

- **Edge**: IoT Greengrass + SageMaker Neo optimized models
- **Cloud**: Kinesis + Lambda + DynamoDB + ECS Fargate
- **ML**: SageMaker training pipeline with maritime threat detection
- **UI**: React dashboard with real-time convoy tracking

## Security Notice

⚠️ **DEFENSIVE SYSTEM ONLY** - All engagement decisions require human authorization. No automated kinetic responses.

See `docs/deploy_instructions.md` for complete deployment guide.