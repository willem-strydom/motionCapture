# mlp_model.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import json
import numpy as np
import glob
import os
from sklearn.model_selection import train_test_split
from MocapModels import MocapModel
from simplex import optimize_mab_simplex

class TrialDataset(Dataset):
    def __init__(self, trial_files, n_frames=50, history_length=5):
        self.trials = []
        self.labels = []
        self.n_frames = n_frames
        self.history_length = history_length
        
        # Load all trial files
        for trial_file in trial_files:
            with open(trial_file, 'r') as f:
                data = json.load(f)
            
            trials = data if isinstance(data, list) else [data]
            
            # Process trials with cumulative history
            play_history = []
            win_history = []
            
            for trial in trials:
                foyer_data = trial['foyer']
                outcome = trial['outcome']
                
                if not foyer_data or not outcome:
                    continue
                
                # Extract label
                machine_name = list(outcome.keys())[0]
                label = int(machine_name[1]) - 1  # Convert 'm1' to 0, 'm2' to 1, etc.
                win = list(outcome.values())[0]
                
                # Process foyer data
                foyer_features = self.process_foyer_data(foyer_data)
                
                if foyer_features is not None:
                    # Get behavioral features
                    behavioral_features = self.encode_behavioral_features(play_history, win_history)
                    
                    # Combine features
                    combined_features = np.concatenate([foyer_features, behavioral_features])
                    
                    self.trials.append(combined_features)
                    self.labels.append(label)
                    
                    # Update history
                    play_history.append(label)
                    win_history.append(win)
    
    def process_foyer_data(self, foyer_data):
        """Downsample to fixed length and extract simple features"""
        timestamps = sorted(foyer_data.keys(), key=int)
        
        if len(timestamps) < self.n_frames:
            return None
        
        step = len(timestamps) // self.n_frames
        selected_indices = [i * step for i in range(self.n_frames)]
        
        # Just use position and angle to each machine for simplicity
        features = []
        for idx in selected_indices:
            timestamp = timestamps[idx]
            frame = foyer_data[timestamp]
            
            # Extract only essential features
            features.extend([
                frame.get('position_x', 0),
                frame.get('position_z', 0),
                frame.get('theta_1', 0),
                frame.get('theta_2', 0),
                frame.get('theta_3', 0),
                frame.get('theta_4', 0)
            ])
        
        return np.array(features)
    
    def encode_behavioral_features(self, play_history, win_history):
        """Encode play history and win rates"""
        # Last N plays (one-hot encoded)
        recent_plays = np.zeros((self.history_length, 4))
        for i, play in enumerate(play_history[-self.history_length:]):
            recent_plays[self.history_length - len(play_history) + i, play] = 1
        
        # Machine win rates
        machine_plays = np.zeros(4)
        machine_wins = np.zeros(4)
        
        for play, win in zip(play_history, win_history):
            machine_plays[play] += 1
            if win:
                machine_wins[play] += 1
        
        # Use uniform prior for unplayed machines
        win_rates = np.array([0.5] * 4)
        for i in range(4):
            if machine_plays[i] > 0:
                win_rates[i] = machine_wins[i] / machine_plays[i]
        
        # Combine all behavioral features
        return np.concatenate([recent_plays.flatten(), win_rates])
    
    def __len__(self):
        return len(self.trials)
    
    def __getitem__(self, idx):
        return torch.FloatTensor(self.trials[idx]), torch.LongTensor([self.labels[idx]])


class SimpleMLP(nn.Module):
    def __init__(self, foyer_dim, behavioral_dim, hidden_dim=32, n_machines=4):
        super(SimpleMLP, self).__init__()
        
        # Simple feature downsampling
        self.downsample = nn.Linear(foyer_dim, hidden_dim)
        
        # Simple classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim + behavioral_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_machines)
        )
    
    def forward(self, foyer_features, behavioral_features):
        # Downsample foyer features
        latent = self.downsample(foyer_features)
        
        # Combine with behavioral features
        combined = torch.cat([latent, behavioral_features], dim=1)
        
        # Classify
        output = self.classifier(combined)
        return output


class MLPMocapModel(MocapModel):
    def __init__(self, model_path=None, n_frames=4, history_length=5, burn_in=5):
        super().__init__()
        self.n_frames = n_frames
        self.history_length = history_length
        self.burn_in = burn_in
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Feature dimensions
        self.foyer_dim = n_frames * 6  # 6 features per frame
        self.behavioral_dim = history_length * 4 + 4  # one-hot history + win rates
        
        if model_path:
            self.model = SimpleMLP(
                self.foyer_dim, 
                self.behavioral_dim
            ).to(self.device)
            self.model.load_state_dict(torch.load(model_path))
            self.model.eval()
        else:
            self.model = None
    
    def adjust_winrates(self, foyer, play_history, win_history, current_winrates):
        # During burn-in, return uniform adjustments
        if len(play_history) < self.burn_in or self.model is None:
            return np.zeros(self.n_machines)
        
        # Process foyer data
        foyer_features = self.process_foyer_for_inference(foyer)
        if foyer_features is None:
            return np.zeros(self.n_machines)
        
        # Process behavioral features
        behavioral_features = self.encode_behavioral_features(play_history, win_history)
        
        # Run inference
        with torch.no_grad():
            foyer_tensor = torch.FloatTensor(foyer_features).unsqueeze(0).to(self.device)
            behavioral_tensor = torch.FloatTensor(behavioral_features).unsqueeze(0).to(self.device)
            
            output = self.model(foyer_tensor, behavioral_tensor)
            probabilities = torch.softmax(output, dim=1).cpu().numpy()[0]
        
        # Convert probabilities to win rate adjustments
        confidence = probabilities - 0.25  # Center around uniform distribution
        return optimize_mab_simplex(confidence, current_winrates, self.max_adjustment, self.house_edge)
    
    def process_foyer_for_inference(self, foyer):
        """Process foyer data matching training format"""
        timestamps = sorted(foyer.keys(), key=int)
        
        if len(timestamps) < self.n_frames:
            return None
        
        step = len(timestamps) // self.n_frames
        selected_indices = [i * step for i in range(self.n_frames)]
        
        features = []
        for idx in selected_indices:
            timestamp = timestamps[idx]
            frame = foyer[timestamp]
            
            features.extend([
                frame.get('position_x', 0),
                frame.get('position_z', 0),
                frame.get('theta_1', 0),
                frame.get('theta_2', 0),
                frame.get('theta_3', 0),
                frame.get('theta_4', 0)
            ])
        
        return np.array(features)
    
    def encode_behavioral_features(self, play_history, win_history):
        """Encode behavioral features matching training format"""
        # Recent plays
        recent_plays = np.zeros((self.history_length, 4))
        for i, play in enumerate(play_history[-self.history_length:]):
            if i < self.history_length:
                recent_plays[self.history_length - len(play_history) + i, play] = 1
        
        # Win rates
        machine_plays = np.zeros(4)
        machine_wins = np.zeros(4)
        
        for play, win in zip(play_history, win_history):
            machine_plays[play] += 1
            if win:
                machine_wins[play] += 1
        
        win_rates = np.array([0.5] * 4)
        for i in range(4):
            if machine_plays[i] > 0:
                win_rates[i] = machine_wins[i] / machine_plays[i]
        
        return np.concatenate([recent_plays.flatten(), win_rates])


def train_model(data_dirs=['./Willem', './Owen', './Ryan'], n_epochs=30, batch_size=16):
    # Collect trial files
    trial_files = []
    for directory in data_dirs:
        trial_files.extend(glob.glob(os.path.join(directory, "*.json")))
    
    # Create dataset
    dataset = TrialDataset(trial_files)
    
    # Split data
    train_indices, val_indices = train_test_split(
        range(len(dataset)), test_size=0.2, random_state=42
    )
    
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    foyer_dim = 50 * 6  # 50 frames * 6 features
    behavioral_dim = 5 * 4 + 4  # 5 recent plays + 4 win rates
    
    model = SimpleMLP(foyer_dim, behavioral_dim).to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    best_val_acc = 0
    for epoch in range(n_epochs):
        # Train
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for inputs, labels in train_loader:
            labels = labels.to(device).squeeze()
            
            # Split features
            foyer_features = inputs[:, :300].to(device)  # First 300 features
            behavioral_features = inputs[:, 300:].to(device)  # Remaining features
            
            optimizer.zero_grad()
            outputs = model(foyer_features, behavioral_features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
        
        # Validate
        model.eval()
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                labels = labels.to(device).squeeze()
                
                foyer_features = inputs[:, :300].to(device)
                behavioral_features = inputs[:, 300:].to(device)
                
                outputs = model(foyer_features, behavioral_features)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        train_acc = 100. * train_correct / train_total
        val_acc = 100. * val_correct / val_total
        
        print(f'Epoch {epoch+1}/{n_epochs}:')
        print(f'  Train Loss: {train_loss/len(train_loader):.4f}, Train Acc: {train_acc:.2f}%')
        print(f'  Val Acc: {val_acc:.2f}%')
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_mlp_model.pth')
    
    return model


if __name__ == "__main__":
    model = train_model()