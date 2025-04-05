import json
import math
from FeatureExtractor import IntegralLineOfSight, WindowedControl, RegressionHouse, BayesianHouse
def correct_angles_with_new_machine_centers(filename, correct_machine_positions):
    """
    Process a JSON file to recompute the theta angles using correct machine center positions.
    
    Parameters:
    filename (str): Path to the JSON file to process
    correct_machine_positions (dict): Dictionary mapping machine names (m1, m2, m3, m4) to their
                                     correct (x, z) positions
    """
    
    # Load the data from the file
    with open(filename, 'r') as file:
        data = json.load(file)
    
    # Process each trial in the data
    if isinstance(data, list):
        for trial in data:
            process_trial(trial, correct_machine_positions)
    else:
        # Single trial case
        process_trial(data, correct_machine_positions)
    
    # Write the corrected data back to the file
    with open(filename, 'w') as file:
        json.dump(data, file, indent=4)
    
    print(f"Processed and updated file: {filename}")

def process_file(filename, mocap, behavioral,play_history,win_history):
    """
    Process a JSON file to extract model performance info
    
    Parameters:
    filename (str): Path to the JSON file to process
    model (implements .adjust_winrates(foyer,play_history,win_history,current_winrates) and returns 1xmachine of adjustments)
    """
    
    # Load the data from the file
    with open(filename, 'r') as file:
        data = json.load(file)
    
    # Process each trial in the data
    mocapImpact = 0
    behavioralImpact = 0
    if isinstance(data, list):
        for trial in data:
            mocapInstanceImpact, instanceDecision, instanceOutcome = get_mocap_impact(mocap, trial)
            behavioralInstanceImpact,_,_ = get_behavioral_impact(behavioral,trial,play_history,win_history)
            play_history.append(instanceDecision)
            win_history.append(instanceOutcome)
            behavioralImpact += behavioralInstanceImpact
            mocapImpact += mocapInstanceImpact
    return [mocapImpact, mocapImpact < 0],[behavioralImpact,behavioralImpact < 0],play_history,win_history

def get_mocap_impact(model,trial):
    i = 0
    existingWinrates = []
    for machine in iter(trial['pre_win_rates'].items()):
        if machine[0] == next(iter(trial['outcome'].keys())):
            playerDecision = i
        existingWinrates.append(machine[1])
        i += 1
    adjustments = model.adjust_winrates(trial['foyer'],None,None,existingWinrates)
    return adjustments[playerDecision],playerDecision,next(iter(trial['outcome'].values()))

def get_behavioral_impact(model,trial,play_history,win_history):
    i = 0
    existingWinrates = []
    for machine in iter(trial['pre_win_rates'].items()):
        if machine[0] == next(iter(trial['outcome'].keys())):
            playerDecision = i
        existingWinrates.append(machine[1])
        i += 1
    adjustments = model.adjust_winrates(play_history,win_history,existingWinrates)
    return adjustments[playerDecision],playerDecision,next(iter(trial['outcome'].values()))

def process_trial(trial, correct_positions):
    """Process a single trial's data to correct theta angles."""
    # Process foyer data
    if "foyer" in trial:
        for timestamp, frame in trial["foyer"].items():
            recompute_frame_angles(frame, correct_positions)
    
    # Process walk data if present
    if "walk" in trial:
        for timestamp, frame in trial["walk"].items():
            recompute_frame_angles(frame, correct_positions)

def recompute_frame_angles(frame, correct_positions):
    """
    Recompute the theta angles for a single frame using correct machine positions.
    
    Parameters:
    frame (dict): A single frame of mocap data
    incorrect_positions (dict): Dictionary mapping machine names to incorrect (x,z) positions
    correct_positions (dict): Dictionary mapping machine names to correct (x,z) positions
    """
    # Extract player position and facing angle (theta)
    player_pos = [
        frame.get("position_x", 0),
        frame.get("position_y", 0),
        frame.get("position_z", 0)
    ]
    player_theta = frame.get("theta", 0)
    
    # Create theta vector using the same approach as original code
    theta_vec = [math.cos(player_theta), math.sin(player_theta)]
    
    # Calculate angles to each machine using correct positions
    for i, machine_name in enumerate(["m1", "m2", "m3", "m4"], 1):
        if machine_name in correct_positions:
            mx, mz = correct_positions[machine_name]
            
            # Calculate vector from player to machine
            dx = mx - player_pos[0]
            dz = mz - player_pos[2]
            machine_vec = [dx, dz]
            
            # Calculate angle using same method as original code
            norm_product = math.hypot(*theta_vec) * math.hypot(dx, dz)
            
            if norm_product > 0:
                dot_product = theta_vec[0] * machine_vec[0] + theta_vec[1] * machine_vec[1]
                angle = math.acos(dot_product / norm_product)
                angle_degrees = math.degrees(angle)
            else:
                angle_degrees = 0.0
            
            # Update the theta value in the frame
            frame[f"theta_{i}"] = angle_degrees

if __name__ == "__main__":
    import os
    import glob
    
    correct_positions = {
        "m1": (0.914461, -0.416378),
        "m2": (0.930728, -0.117966),
        "m3": (0.945728,  0.181054),
        "m4": (0.975101,  0.483173)
    }
    
    def process_directory(directory_path, mocap, behavioral):
        """
        Process all JSON files in the specified directory.
        
        Parameters:
        directory_path (str): Path to the directory containing JSON files
        correct_positions (dict): Dictionary mapping machine names to correct positions
        """
        # Get all JSON files in the directory
        json_files = glob.glob(os.path.join(directory_path, "*.json"))
        
        print(f"Found {len(json_files)} JSON files in {directory_path}")
        
        # Process each JSON file
        mocapSuccess = 0
        mocapImpact = 0
        behavioralSuccess = 0
        behavioralImpact = 0
        i = 0
        play_history = []
        win_history = []
        for file_path in json_files:
            #try:
                #print(f"Processing {file_path}...")
                #correct_angles_with_new_machine_centers(file_path, correct_positions)
                mocapOutcome,behavioralOutcome,play_history,win_history = process_file(file_path,mocap,behavioral,play_history,win_history)
                mocapSuccess += mocapOutcome[1]
                mocapImpact += mocapOutcome[0]
                behavioralSuccess += behavioralOutcome[1]
                behavioralImpact += behavioralOutcome[0]
                i += 1
                #print(f"Successfully processed {file_path}")
            #except Exception as e:
                #print(f"Error processing {file_path}: {str(e)}")
        return [mocapImpact,mocapSuccess,i],[behavioralImpact,behavioralSuccess,i]
    
    # Example usage:
    mocap = IntegralLineOfSight()
    behavioral = BayesianHouse()
    for directory in ['./Willem', './Owen', './Ryan']:
        mocapPerformance,behavioralPerformance = process_directory(directory, mocap,behavioral)
        print(f"Mocap model had total impact of{mocapPerformance[0]}, and made a benificial prediction {mocapPerformance[1]} times out of {mocapPerformance[2]}, with an accuracy of {mocapPerformance[1] / mocapPerformance[2]}%")
        print(f"Behvaioral model had total impact of{behavioralPerformance[0]}, and made a benificial prediction {behavioralPerformance[1]} times out of {behavioralPerformance[2]}, with an accuracy of {behavioralPerformance[1] / behavioralPerformance[2]}%")