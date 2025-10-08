import argparse
import json
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from sklearn.metrics import f1_score, accuracy_score
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MaritimeThreatModel(nn.Module):
    def __init__(self, input_size=10, hidden_size=128, num_classes=9, sequence_length=60):
        super().__init__()
        self.hidden_size = hidden_size
        self.sequence_length = sequence_length
        
        # 1D CNN for feature extraction
        self.conv1 = nn.Conv1d(input_size, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(2)
        
        # LSTM for sequence modeling
        self.lstm = nn.LSTM(128, hidden_size, batch_first=True, bidirectional=True)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )
        
    def forward(self, x):
        # x shape: (batch, sequence, features)
        x = x.transpose(1, 2)  # (batch, features, sequence)
        
        # CNN feature extraction
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.pool(x)
        
        # Back to sequence format
        x = x.transpose(1, 2)  # (batch, sequence/2, 128)
        
        # LSTM
        lstm_out, _ = self.lstm(x)
        
        # Use last output for classification
        output = self.classifier(lstm_out[:, -1, :])
        return output

class MaritimeDataset(Dataset):
    def __init__(self, data_dir, sequence_length=60):
        self.data_dir = data_dir
        self.sequence_length = sequence_length
        self.samples = self._load_samples()
        
    def _load_samples(self):
        samples = []
        for file in os.listdir(self.data_dir):
            if file.endswith('.jsonl'):
                with open(os.path.join(self.data_dir, file), 'r') as f:
                    frames = [json.loads(line) for line in f]
                    
                # Create sequences
                for i in range(0, len(frames) - self.sequence_length + 1, 10):
                    sequence = frames[i:i + self.sequence_length]
                    samples.append(sequence)
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sequence = self.samples[idx]
        
        # Extract features and labels
        features = []
        labels = []
        
        for frame in sequence:
            # Simple feature extraction
            feature_vector = [
                len(frame.get('detections', [])),
                frame.get('weather', {}).get('sea_state', 0),
                frame.get('weather', {}).get('visibility_km', 10),
                frame.get('weather', {}).get('wind_speed_knots', 0),
                frame.get('sensor_data', {}).get('radar_contacts', 0),
                frame.get('sensor_data', {}).get('acoustic_level_db', 50),
                frame.get('sensor_data', {}).get('rf_signals', 0),
                1.0 if frame.get('ground_truth_threat', False) else 0.0,
                frame.get('convoy_pose', {}).get('speed_knots', 0),
                frame.get('drone_pose', {}).get('altitude_m', 100) / 100.0
            ]
            features.append(feature_vector)
            
            # Threat type encoding
            threat_types = [
                'small_fast_craft', 'suspicious_loitering_vessel', 'unregistered_vessel',
                'ais_spoofing', 'drone_overwater', 'diver_or_swimmer',
                'floating_mine_like_object', 'collision_risk', 'acoustic_gunshot'
            ]
            
            threat_type = frame.get('threat_type')
            if threat_type in threat_types:
                label = threat_types.index(threat_type)
            else:
                label = 0  # No threat
            labels.append(label)
        
        # Use last label as sequence label
        return torch.FloatTensor(features), torch.LongTensor([labels[-1]])

def train_model(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Load datasets
    train_dataset = MaritimeDataset(args.train_dir, args.sequence_length)
    val_dataset = MaritimeDataset(args.val_dir, args.sequence_length)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Model
    model = MaritimeThreatModel(
        input_size=10,
        hidden_size=128,
        num_classes=args.num_classes,
        sequence_length=args.sequence_length
    ).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    
    # Training loop
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device).squeeze()
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_predictions = []
        val_targets = []
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device).squeeze()
                output = model(data)
                pred = output.argmax(dim=1)
                
                val_predictions.extend(pred.cpu().numpy())
                val_targets.extend(target.cpu().numpy())
        
        val_accuracy = accuracy_score(val_targets, val_predictions)
        val_f1 = f1_score(val_targets, val_predictions, average='weighted')
        
        logger.info(f'Epoch {epoch+1}/{args.epochs}')
        logger.info(f'Train Loss: {train_loss/len(train_loader):.4f}')
        logger.info(f'Validation Accuracy: {val_accuracy:.4f}')
        logger.info(f'Validation F1: {val_f1:.4f}')
    
    # Save model
    torch.save(model.state_dict(), os.path.join(args.model_dir, 'model.pth'))
    
    # Save TorchScript model for inference
    model.eval()
    example_input = torch.randn(1, args.sequence_length, 10).to(device)
    traced_model = torch.jit.trace(model, example_input)
    traced_model.save(os.path.join(args.model_dir, 'model.pt'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    # SageMaker specific arguments
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--train-dir', type=str, default=os.environ.get('SM_CHANNEL_TRAINING'))
    parser.add_argument('--val-dir', type=str, default=os.environ.get('SM_CHANNEL_VALIDATION'))
    
    # Hyperparameters
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--sequence_length', type=int, default=60)
    parser.add_argument('--num_classes', type=int, default=9)
    
    args = parser.parse_args()
    train_model(args)