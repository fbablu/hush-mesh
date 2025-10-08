#!/usr/bin/env python3
"""
SageMaker Ship Classification Pipeline for Maritime ACPS
"""

import boto3
import sagemaker
from sagemaker.pytorch import PyTorch
from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.sklearn.processing import SKLearnProcessor
import os
import json

class SageMakerShipClassifier:
    def __init__(self, role_arn=None, bucket_name=None):
        self.session = sagemaker.Session()
        self.role = role_arn or sagemaker.get_execution_role()
        self.bucket = bucket_name or self.session.default_bucket()
        self.region = self.session.boto_region_name
        
    def upload_data_to_s3(self):
        """Upload ship dataset to S3"""
        print("Uploading dataset to S3...")
        
        # Upload training data
        train_data_uri = self.session.upload_data(
            path='./data/Ships dataset/train',
            bucket=self.bucket,
            key_prefix='maritime-acps/ship-data/train'
        )
        
        # Upload test data
        test_data_uri = self.session.upload_data(
            path='./data/Ships dataset/test',
            bucket=self.bucket,
            key_prefix='maritime-acps/ship-data/test'
        )
        
        print(f"Training data: {train_data_uri}")
        print(f"Test data: {test_data_uri}")
        
        return train_data_uri, test_data_uri
    
    def create_training_script(self):
        """Create PyTorch training script for SageMaker"""
        training_script = '''
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import json
import argparse

class ShipDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.images = []
        self.labels = []
        
        images_dir = os.path.join(data_dir, 'images')
        labels_dir = os.path.join(data_dir, 'labels')
        
        for img_file in os.listdir(images_dir):
            if img_file.endswith('.jpg'):
                label_file = os.path.join(labels_dir, img_file.replace('.jpg', '.txt'))
                if os.path.exists(label_file):
                    with open(label_file, 'r') as f:
                        line = f.readline().strip()
                        if line:
                            class_id = int(line.split()[0])
                            if class_id < 8:
                                self.images.append(os.path.join(images_dir, img_file))
                                self.labels.append(class_id)
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = Image.open(self.images[idx]).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

class ShipCNN(nn.Module):
    def __init__(self, num_classes=8):
        super(ShipCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((7, 7))
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(128 * 7 * 7, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

def train_model(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    train_dataset = ShipDataset(args.train_dir, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    model = ShipCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    model.train()
    for epoch in range(args.epochs):
        running_loss = 0.0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            if batch_idx % 10 == 0:
                print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}')
    
    # Save model
    torch.save(model.state_dict(), os.path.join(args.model_dir, 'model.pth'))
    
    # Save class names
    class_names = ['cargo', 'military', 'carrier', 'cruise', 'tankers', 'trawlers', 'tugboat', 'yacht']
    with open(os.path.join(args.model_dir, 'class_names.json'), 'w') as f:
        json.dump(class_names, f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--train-dir', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    
    args = parser.parse_args()
    train_model(args)
'''
        
        os.makedirs('sagemaker_scripts', exist_ok=True)
        with open('sagemaker_scripts/train.py', 'w') as f:
            f.write(training_script)
        
        return 'sagemaker_scripts/train.py'
    
    def train_model(self, train_data_uri):
        """Launch SageMaker training job"""
        print("Creating training script...")
        script_path = self.create_training_script()
        
        print("Starting SageMaker training job...")
        
        estimator = PyTorch(
            entry_point='train.py',
            source_dir='sagemaker_scripts',
            role=self.role,
            instance_type='ml.m5.large',
            instance_count=1,
            framework_version='1.12',
            py_version='py38',
            hyperparameters={
                'epochs': 15
            }
        )
        
        estimator.fit({'train': train_data_uri})
        
        return estimator
    
    def deploy_model(self, estimator):
        """Deploy model to SageMaker endpoint"""
        print("Deploying model to endpoint...")
        
        predictor = estimator.deploy(
            initial_instance_count=1,
            instance_type='ml.t2.medium',
            endpoint_name='maritime-acps-ship-classifier'
        )
        
        return predictor
    
    def compile_for_edge(self, estimator):
        """Compile model for edge deployment with SageMaker Neo"""
        print("Compiling model for edge deployment...")
        
        compilation_job_name = f"maritime-acps-neo-{int(time.time())}"
        
        compiled_model = estimator.compile_model(
            target_instance_family='jetson_nano',
            input_shape={'input0': [1, 3, 224, 224]},
            job_name=compilation_job_name,
            role=self.role,
            framework='pytorch',
            framework_version='1.12'
        )
        
        return compiled_model

def main():
    print("=== SageMaker Ship Classification Pipeline ===")
    
    # Initialize classifier
    classifier = SageMakerShipClassifier()
    
    # Upload data to S3
    train_uri, test_uri = classifier.upload_data_to_s3()
    
    # Train model
    estimator = classifier.train_model(train_uri)
    
    # Deploy endpoint
    predictor = classifier.deploy_model(estimator)
    
    # Compile for edge
    compiled_model = classifier.compile_for_edge(estimator)
    
    print("Pipeline completed!")
    print(f"Endpoint: {predictor.endpoint_name}")
    print("Model ready for Greengrass deployment")

if __name__ == "__main__":
    import time
    main()