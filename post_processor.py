import json
import math

def correct_angles_with_new_machine_centers(filename, correct_machine_positions):
    """
    Process a JSON file to recompute the theta angles using correct machine center positions.
    
    Parameters:
    filename (str): Path to the JSON file to process
    correct_machine_positions (dict): Dictionary mapping machine names (m1, m2, m3, m4) to their
                                     correct (x, z) positions
    """
    # The incorrect machine positions used in the original script
    incorrect_positions = {
        "m1": (1, 1),
        "m2": (2, 2),
        "m3": (3, 3),
        "m4": (4, 4)
    }
    
    # Load the data from the file
    with open(filename, 'r') as file:
        data = json.load(file)
    
    # Process each trial in the data
    if isinstance(data, list):
        for trial in data:
            process_trial(trial, incorrect_positions, correct_machine_positions)
    else:
        # Single trial case
        process_trial(data, incorrect_positions, correct_machine_positions)
    
    # Write the corrected data back to the file
    with open(filename, 'w') as file:
        json.dump(data, file, indent=4)
    
    print(f"Processed and updated file: {filename}")

def process_trial(trial, incorrect_positions, correct_positions):
    """Process a single trial's data to correct theta angles."""
    # Process foyer data
    if "foyer" in trial:
        for timestamp, frame in trial["foyer"].items():
            recompute_frame_angles(frame, incorrect_positions, correct_positions)
    
    # Process walk data if present
    if "walk" in trial:
        for timestamp, frame in trial["walk"].items():
            recompute_frame_angles(frame, incorrect_positions, correct_positions)

def recompute_frame_angles(frame, incorrect_positions, correct_positions):
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

# We no longer need these functions since we're using the same approach as the original code
# They are kept here as comments for reference

# def quaternion_to_forward(qw, qx, qy, qz):
#     """
#     Convert a quaternion to a forward vector.
#     
#     Parameters:
#     qw, qx, qy, qz: Quaternion components
#     
#     Returns:
#     Tuple (x, y, z) representing the forward vector
#     """
#     # Using quaternion to rotate the forward vector
#     vx = 1 - 2 * (qy * qy + qz * qz)
#     vy = 2 * (qx * qy - qw * qz)
#     vz = 2 * (qx * qz + qw * qy)
#     
#     return [vx, vz]  # Return 2D vector for XZ plane

# def compute_signed_angle(vec1, vec2):
#     """
#     Calculate the signed angle between two 2D vectors.
#     Positive angle means vec2 is to the right of vec1, negative means to the left.
#     
#     Parameters:
#     vec1, vec2: Two 2D vectors [x, z]
#     
#     Returns:
#     The signed angle in radians
#     """
#     # Implementation details removed as we're not using this function anymore

# Example usage:
if __name__ == "__main__":
    import os
    import glob
    
    # Define the correct machine positions
    correct_positions = {
        "m1": (1.5, 2.0),  # Example corrected positions
        "m2": (3.0, 1.0),
        "m3": (4.5, 2.5),
        "m4": (3.0, 4.0)
    }
    
    # Process a single file
    # correct_angles_with_new_machine_centers("player_history.json", correct_positions)
    
    # Process all JSON files in a directory
    def process_directory(directory_path, correct_positions):
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
        for file_path in json_files:
            try:
                print(f"Processing {file_path}...")
                correct_angles_with_new_machine_centers(file_path, correct_positions)
                print(f"Successfully processed {file_path}")
            except Exception as e:
                print(f"Error processing {file_path}: {str(e)}")
    
    # Example usage:
    for directory in ['./Willem', './Owen', './Ryan']:
        process_directory(directory, correct_positions)