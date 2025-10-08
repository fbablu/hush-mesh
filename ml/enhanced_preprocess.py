#!/usr/bin/env python3
import boto3
import json
import random

def preprocess_enhanced_data(bucket_name):
    """Preprocess enhanced maritime data for training"""
    s3 = boto3.client('s3')
    
    scenarios = ['piracy_ambush', 'swarm_interdiction', 'mine_field', 'submarine_contact']
    
    all_train_data = []
    all_val_data = []
    
    for scenario in scenarios:
        # Download enhanced data
        response = s3.get_object(Bucket=bucket_name, Key=f'enhanced/{scenario}/data.jsonl')
        content = response['Body'].read().decode('utf-8')
        
        frames = [json.loads(line) for line in content.strip().split('\n')]
        
        # Split 80/20 train/validation
        random.shuffle(frames)
        split_idx = int(0.8 * len(frames))
        
        train_frames = frames[:split_idx]
        val_frames = frames[split_idx:]
        
        all_train_data.extend(train_frames)
        all_val_data.extend(val_frames)
    
    # Upload combined training data
    train_content = '\n'.join([json.dumps(frame) for frame in all_train_data])
    s3.put_object(
        Bucket=bucket_name,
        Key='processed/train/enhanced_data.jsonl',
        Body=train_content,
        ContentType='application/x-ndjson'
    )
    
    # Upload combined validation data
    val_content = '\n'.join([json.dumps(frame) for frame in all_val_data])
    s3.put_object(
        Bucket=bucket_name,
        Key='processed/val/enhanced_data.jsonl',
        Body=val_content,
        ContentType='application/x-ndjson'
    )
    
    print(f"Processed {len(all_train_data)} training samples and {len(all_val_data)} validation samples")

if __name__ == '__main__':
    import sys
    bucket_name = sys.argv[1]
    preprocess_enhanced_data(bucket_name)