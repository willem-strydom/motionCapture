import torch
import torch.nn as nn
import torch.nn.functional as F
import json
from sklearn.preprocessing import LabelEncoder
import numpy as np


def load_and_process_data(json_path):
    # Load JSON data
    with open(json_path, 'r') as f:
        trials = json.load(f)

    # Initialize containers
    X = []
    y = []
    
    # Process each trial
    j = -1
    for trial in trials:
        j += 1
        if j % 2 == 1:
            continue
        
        # Extract label from outcome
        label = list(trial['outcome'].keys())[0]
        
        # Process skeleton data
        foyer_data = trial['foyer']
        
        # Sort timestamps numerically
        timestamps = sorted(foyer_data.keys(), key=lambda x: int(x))
        
        # Get all unique rigid IDs from first timestamp (assuming consistent joints)
        first_ts = timestamps[0]
        rigid_ids = sorted(
            [list(entry.keys())[0] for entry in foyer_data[first_ts]],
            key=lambda x: int(x)
        )
        
        # Create feature matrix for this trial
        trial_features = []
        print(len(timestamps))
        if len(timestamps) >= 650:
            truncated_timestamps = timestamps[-650:]
            for ts in truncated_timestamps:
                # Create dictionary of rigid body data for this timestep
                ts_data = {list(entry.keys())[0]: entry[list(entry.keys())[0]] 
                        for entry in foyer_data[ts]}
                
                # Build feature vector maintaining rigid ID order
                ts_vector = []
                i = 0
                for rid in rigid_ids:
                    if i < 10:
                        if rid in ts_data:
                            # Flatten position + rotation (7 values per rigid body)
                            ts_vector.extend(ts_data[rid][0] + ts_data[rid][1])
                        else:
                            # Handle missing data (pad with zeros if needed)
                            ts_vector.extend([0.0]*7)
                        i += 1
                while i < 10:
                    ts_vector.extend([0.0]*7)
                    i += 1
                trial_features.append(ts_vector)
            
            X.append(np.array(trial_features))
            y.append(label)

    # Convert to numpy arrays
    #print(len(X[6][5]))
    X = np.array(X)
    
    # Encode labels
    le = LabelEncoder()
    y = le.fit_transform(y)
    
    return X, y, le

class SmallCNN(nn.Module):
    def __init__(self, input_length, num_features, conv_channels=5, kernel_size=1, 
                 hidden_size=32, num_classes=4):
        super(SmallCNN, self).__init__()
        conv_output_length = input_length - kernel_size + 1
        pool_output_length = conv_output_length // 2
        
        # Layers
        self.conv1 = nn.Conv1d(in_channels=num_features, 
                               out_channels=conv_channels, 
                               kernel_size=kernel_size)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(conv_channels * pool_output_length, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        return F.softmax(x, dim=1)

# be sure to dynamically change input lenge in the future
def create_model(input_length=50, num_features=1):
    model = SmallCNN(input_length=input_length, 
                     num_features=num_features, 
                     conv_channels=5, 
                     kernel_size=1, 
                     hidden_size=8, #8 latent features
                     num_classes=4)
    return model


# model = create_model(input_length=length, num_features=pick_however_many_you_want)