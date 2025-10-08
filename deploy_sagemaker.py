#!/usr/bin/env python3
"""
Deploy SageMaker Ship Classification for Maritime ACPS
"""

import boto3
import sagemaker
import time

def setup_sagemaker_pipeline():
    """Setup complete SageMaker pipeline"""
    
    # Install SageMaker SDK
    print("Installing SageMaker dependencies...")
    import subprocess
    subprocess.check_call(['pip', 'install', '-r', 'sagemaker_requirements.txt', '--user'])
    
    # Initialize SageMaker session
    session = sagemaker.Session()
    role = sagemaker.get_execution_role()
    bucket = session.default_bucket()
    
    print(f"SageMaker role: {role}")
    print(f"S3 bucket: {bucket}")
    
    # Create S3 bucket structure
    s3 = boto3.client('s3')
    
    try:
        s3.create_bucket(Bucket=bucket)
        print(f"Created S3 bucket: {bucket}")
    except:
        print(f"Using existing S3 bucket: {bucket}")
    
    # Upload dataset
    print("Uploading ship dataset to S3...")
    train_uri = session.upload_data(
        path='./data/Ships dataset/train',
        bucket=bucket,
        key_prefix='maritime-acps/ship-data/train'
    )
    
    print(f"Training data uploaded to: {train_uri}")
    
    return session, role, bucket, train_uri

def create_training_job():
    """Create and run SageMaker training job"""
    from sagemaker_ship_classifier import SageMakerShipClassifier
    
    classifier = SageMakerShipClassifier()
    
    # Upload data
    train_uri, test_uri = classifier.upload_data_to_s3()
    
    # Train model
    estimator = classifier.train_model(train_uri)
    
    return estimator

def main():
    print("=== Deploying SageMaker Ship Classification ===")
    
    try:
        # Setup pipeline
        session, role, bucket, train_uri = setup_sagemaker_pipeline()
        
        # Create training job
        print("Starting training job...")
        estimator = create_training_job()
        
        print("Training completed successfully!")
        print("Model ready for deployment to Maritime ACPS")
        
    except Exception as e:
        print(f"Error: {e}")
        print("Note: This requires AWS credentials and SageMaker permissions")
        print("For demo purposes, the local model can be used for development")

if __name__ == "__main__":
    main()