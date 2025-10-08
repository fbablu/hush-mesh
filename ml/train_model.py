#!/usr/bin/env python3
import argparse
import json
import boto3
import sagemaker
from sagemaker.pytorch import PyTorch
from sagemaker.tuner import HyperparameterTuner, IntegerParameter, ContinuousParameter

def create_training_job(data_bucket, model_name, use_spot=True):
    """Create SageMaker training job for maritime threat detection"""
    
    import os
    session = sagemaker.Session()
    role = os.environ['SAGEMAKER_ROLE_ARN']
    print(f"Using SageMaker role: {role}")
    
    # Training script location
    source_dir = 'training_code'
    
    # Hyperparameters
    hyperparameters = {
        'epochs': 50,
        'batch_size': 32,
        'learning_rate': 0.001,
        'sequence_length': 60,
        'num_classes': 9  # Number of threat types
    }
    
    # Training instance configuration
    instance_type = 'ml.m5.large'
    
    estimator = PyTorch(
        entry_point='train.py',
        source_dir=source_dir,
        role=role,
        instance_type=instance_type,
        instance_count=1,
        framework_version='1.12',
        py_version='py38',
        hyperparameters=hyperparameters,
        use_spot_instances=use_spot,
        max_wait=7200 if use_spot else None,
        max_run=3600,
        output_path=f's3://{data_bucket}/models/',
        code_location=f's3://{data_bucket}/code/',
        enable_sagemaker_metrics=True,
        metric_definitions=[
            {'Name': 'train:loss', 'Regex': 'Train Loss: ([0-9\\.]+)'},
            {'Name': 'validation:accuracy', 'Regex': 'Validation Accuracy: ([0-9\\.]+)'},
            {'Name': 'validation:f1', 'Regex': 'Validation F1: ([0-9\\.]+)'}
        ]
    )
    
    # Training data location
    training_input = f's3://{data_bucket}/processed/train/'
    validation_input = f's3://{data_bucket}/processed/val/'
    
    # Start training
    estimator.fit({
        'training': training_input,
        'validation': validation_input
    }, job_name=f'{model_name}-{int(time.time())}')
    
    return estimator

def create_hpo_job(data_bucket, model_name):
    """Create hyperparameter optimization job"""
    
    session = sagemaker.Session()
    role = sagemaker.get_execution_role()
    
    estimator = PyTorch(
        entry_point='train.py',
        source_dir='training_code',
        role=role,
        instance_type='ml.g4dn.xlarge',
        instance_count=1,
        framework_version='1.12',
        py_version='py38',
        use_spot_instances=True,
        max_wait=7200,
        max_run=3600
    )
    
    # Hyperparameter ranges
    hyperparameter_ranges = {
        'learning_rate': ContinuousParameter(0.0001, 0.01),
        'batch_size': IntegerParameter(16, 64),
        'sequence_length': IntegerParameter(30, 120)
    }
    
    tuner = HyperparameterTuner(
        estimator,
        objective_metric_name='validation:f1',
        hyperparameter_ranges=hyperparameter_ranges,
        metric_definitions=[
            {'Name': 'validation:f1', 'Regex': 'Validation F1: ([0-9\\.]+)'}
        ],
        max_jobs=10,
        max_parallel_jobs=2
    )
    
    training_input = f's3://{data_bucket}/processed/train/'
    validation_input = f's3://{data_bucket}/processed/val/'
    
    tuner.fit({
        'training': training_input,
        'validation': validation_input
    })
    
    return tuner

def compile_model_neo(model_artifacts_s3_path, model_name):
    """Compile model with SageMaker Neo for edge deployment"""
    
    sagemaker_client = boto3.client('sagemaker')
    
    compilation_job_name = f'{model_name}-neo-{int(time.time())}'
    
    response = sagemaker_client.create_compilation_job(
        CompilationJobName=compilation_job_name,
        RoleArn=sagemaker.get_execution_role(),
        InputConfig={
            'S3Uri': model_artifacts_s3_path,
            'DataInputConfig': '{"input0":[1,60,10]}',  # batch, sequence, features
            'Framework': 'PYTORCH'
        },
        OutputConfig={
            'S3OutputLocation': f's3://{data_bucket}/compiled-models/',
            'TargetDevice': 'jetson_xavier'
        },
        StoppingCondition={
            'MaxRuntimeInSeconds': 900
        }
    )
    
    print(f"Neo compilation job started: {compilation_job_name}")
    return compilation_job_name

def main():
    parser = argparse.ArgumentParser(description='Train maritime threat detection model')
    parser.add_argument('--data-bucket', required=True)
    parser.add_argument('--model-name', required=True)
    parser.add_argument('--hpo', action='store_true', help='Run hyperparameter optimization')
    parser.add_argument('--compile', action='store_true', help='Compile model with Neo')
    
    args = parser.parse_args()
    
    if args.hpo:
        print("Starting hyperparameter optimization...")
        tuner = create_hpo_job(args.data_bucket, args.model_name)
        print(f"HPO job started: {tuner.latest_tuning_job.job_name}")
    else:
        print("Starting training job...")
        estimator = create_training_job(args.data_bucket, args.model_name)
        print(f"Training job completed: {estimator.latest_training_job.job_name}")
        
        if args.compile:
            print("Compiling model with Neo...")
            compile_model_neo(estimator.model_data, args.model_name)

if __name__ == '__main__':
    import time
    main()