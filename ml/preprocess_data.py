#!/usr/bin/env python3
import boto3
import json
import os

def preprocess_data(bucket_name):
    """Simple preprocessing - copy raw data to processed folders"""
    s3 = boto3.client('s3')
    
    # Create processed data structure
    scenarios = ['piracy_ambush', 'swarm_interdiction']
    
    for scenario in scenarios:
        # Copy to train folder (80% split simulation)
        s3.copy_object(
            Bucket=bucket_name,
            CopySource={'Bucket': bucket_name, 'Key': f'raw/{scenario}/data.jsonl'},
            Key=f'processed/train/{scenario}.jsonl'
        )
        
        # Copy to validation folder (same data for demo)
        s3.copy_object(
            Bucket=bucket_name,
            CopySource={'Bucket': bucket_name, 'Key': f'raw/{scenario}/data.jsonl'},
            Key=f'processed/val/{scenario}.jsonl'
        )
    
    print("Data preprocessing completed")

if __name__ == '__main__':
    import sys
    bucket_name = sys.argv[1]
    preprocess_data(bucket_name)