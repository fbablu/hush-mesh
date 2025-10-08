#!/usr/bin/env python3
import boto3
import sagemaker
from sagemaker.pytorch import PyTorchModel
import os
import tarfile

def deploy_model_to_sagemaker():
    """Deploy the trained ML model to SageMaker endpoint"""
    
    # Initialize SageMaker session
    sagemaker_session = sagemaker.Session()
    role = os.environ.get('SAGEMAKER_ROLE_ARN', 'arn:aws:iam::334682688228:role/SageMakerExecutionRole')
    
    # Create model artifact (tar.gz with model and inference code)
    model_path = '/tmp/model.tar.gz'
    
    with tarfile.open(model_path, 'w:gz') as tar:
        # Add the trained model
        if os.path.exists('/workshop/ml/model.pth'):
            tar.add('/workshop/ml/model.pth', arcname='model.pth')
        
        # Add inference script
        tar.add('/workshop/ml/inference.py', arcname='code/inference.py')
        tar.add('/workshop/ml/requirements.txt', arcname='code/requirements.txt')
    
    # Upload model to S3
    s3_model_path = sagemaker_session.upload_data(
        path=model_path,
        bucket=sagemaker_session.default_bucket(),
        key_prefix='maritime-acps/model'
    )
    
    print(f"Model uploaded to: {s3_model_path}")
    
    # Create PyTorch model
    pytorch_model = PyTorchModel(
        model_data=s3_model_path,
        role=role,
        framework_version='1.12.0',
        py_version='py38',
        entry_point='inference.py',
        name='maritime-threat-detection'
    )
    
    # Deploy to endpoint
    predictor = pytorch_model.deploy(
        initial_instance_count=1,
        instance_type='ml.t2.medium',
        endpoint_name='maritime-acps-endpoint'
    )
    
    print(f"Model deployed to endpoint: {predictor.endpoint_name}")
    return predictor.endpoint_name

if __name__ == "__main__":
    endpoint_name = deploy_model_to_sagemaker()
    print(f"SageMaker endpoint ready: {endpoint_name}")