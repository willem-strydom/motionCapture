import csv
import tkinter as tk
from collections import defaultdict
import MocapVisualizer
import time
from threading import Thread

def parse_mocap_csv(filename):
    """Parse motion capture CSV file into structured data format."""
    
    with open(filename, 'r') as f:
        lines = list(csv.reader(f))

    # Find data header line starting with "Frame"
    data_header_idx = None
    for i, line in enumerate(lines):
        if line and line[0] == 'Frame':
            data_header_idx = i
            break
    if data_header_idx is None:
        raise ValueError("Could not find data header line starting with 'Frame'")

    # Extract relevant header lines
    type_line = lines[data_header_idx - 4]
    name_line = lines[data_header_idx - 3]
    data_columns = lines[data_header_idx]

    # Build rigid body structure mapping
    rigid_bodies = {}
    i = 2  # Start after Frame and Time columns
    
    while i < len(data_columns):
        col_type = type_line[i]
        col_name = name_line[i]

        if col_type == 'Rigid Body' and col_name.startswith('Rigid Body'):
            rb_name = col_name
            
            if rb_name not in rigid_bodies:
                # Check for rotation components (next 4 columns)
                rotation_cols = []
                for j in range(4):
                    if i + j >= len(data_columns):
                        break
                    if (type_line[i + j] == 'Rigid Body' and 
                        name_line[i + j] == rb_name):
                        rotation_cols.append(i + j)
                
                if len(rotation_cols) == 4:
                    # Check for position components (next 3 after rotation)
                    position_cols = []
                    for j in range(4, 7):
                        idx = i + j
                        if idx >= len(data_columns):
                            break
                        if (type_line[idx] == 'Rigid Body' and 
                            name_line[idx] == rb_name):
                            position_cols.append(idx)
                    
                    if len(position_cols) == 3:
                        rigid_bodies[rb_name] = {
                            'rotation_cols': rotation_cols,
                            'position_cols': position_cols
                        }
                        # Skip processed columns
                        i += 6  # 4 rotation + 3 position - 1 (since we increment in loop)
                # Move to next column if we didn't find valid components
                if rb_name not in rigid_bodies:
                    i += 1
            else:
                i += 1
        else:
            i += 1

    # Parse frame data
    frames = []
    for line in lines[data_header_idx + 1:]:
        if not line or len(line) < len(data_columns):
            continue  # Skip incomplete lines

        frame_data = {
            'frame': int(line[0]),
            'time': float(line[1]),
            'rigid_bodies': defaultdict(dict)
        }

        for rb_name, cols in rigid_bodies.items():
            try:
                rotation = [float(line[i]) for i in cols['rotation_cols']]
                position = [float(line[i]) for i in cols['position_cols']]
            except (ValueError, IndexError):
                # Handle missing/invalid data
                rotation = [0.0] * 4
                position = [0.0] * 3

            frame_data['rigid_bodies'][rb_name] = {
                'rotation': {
                    'x': rotation[0],
                    'y': rotation[1],
                    'z': rotation[2],
                    'w': rotation[3]
                },
                'position': {
                    'x': position[0],
                    'y': position[1],
                    'z': position[2]
                }
            }
        
        frames.append(frame_data)
    
    return frames

def simulate_realtime_stream(visualizer, data, speed=1.0):
    """Simulate real-time streaming from CSV data"""
    prev_time = data[0]['time']
    
    for idx, frame in enumerate(data):
        if idx == 0:
            continue  # Skip first frame as reference
        
        # Calculate time difference between frames
        time_diff = (frame['time'] - prev_time) / speed
        time.sleep(time_diff)
        
        # Add frame to visualizer
        visualizer.add_frame(frame)
        prev_time = frame['time']

if __name__ == "__main__":
    # Load your CSV data
    data = parse_mocap_csv("4482-2-0209-1.csv")
    
    # Create GUI and visualizer
    root = tk.Tk()
    visualizer = MocapVisualizer.MocapVisualizer(root)
    
    # Start simulation in a separate thread
    Thread(target=simulate_realtime_stream, args=(visualizer, data), daemon=True).start()
    
    # Start GUI main loop
    root.mainloop()