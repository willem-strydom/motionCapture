import json
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout

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

def build_model(input_shape, num_classes):

    model = Sequential([
        Conv1D(64, 3, activation='relu', input_shape=input_shape),
        MaxPooling1D(2),
        Dropout(0.3),
        LSTM(128, return_sequences=True),
        LSTM(64),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )
    return model

# Main workflow
if __name__ == "__main__":
    # Load and process data
    X, y, label_encoder = load_and_process_data('trial_77374.json')
    
    # Verify shapes
    print(f"Data shape: {X.shape}")  # Should be (200, 1200, num_features)
    print(f"Labels shape: {y.shape}")  # Should be (200,)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, random_state=42
    )
    
    # Build and train model
    model = build_model(
        input_shape=(X.shape[1], X.shape[2]),
        num_classes=len(label_encoder.classes_)
    )
    
    history = model.fit(
        X_train, y_train,
        epochs=100,
        batch_size=16,
        validation_data=(X_test, y_test),
        verbose=1
    )
    
    # Evaluate
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nTest accuracy: {accuracy:.2f}")