import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from tkinter import scrolledtext
from collections import defaultdict

class MocapVisualizer:

    def __init__(self, root):
        self.root = root
        self.data = []  # Dynamic data buffer
        self.current_frame_idx = 0
        self.playing = False
        self.frame_times = []
        self.bounding_boxes = []
        self.collision_states = {}
        self.playback_speed = 1.0  # Default speed multiplier

        # Initialize axis limits with extreme values
        self.x_limits = [float('inf'), -float('inf')]
        self.y_limits = [float('inf'), -float('inf')]

        # GUI Setup
        root.title("Real-Time Motion Capture Visualizer")
        root.geometry("1400x900")

        # Configure main layout
        main_pane = ttk.PanedWindow(root, orient=tk.VERTICAL)
        main_pane.pack(fill=tk.BOTH, expand=True)

        # Console panel
        self.console = scrolledtext.ScrolledText(main_pane, wrap=tk.WORD, height=8)
        self.console.configure(state='disabled')
        main_pane.add(self.console)

        # Visualization canvas
        self.fig = Figure(figsize=(10, 6))
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=main_pane)
        main_pane.add(self.canvas.get_tk_widget())

        # Control panel
        control_frame = ttk.Frame(root)
        control_frame.pack(pady=10, fill=tk.X)

        # Frame navigation
        self.current_frame = ttk.Spinbox(
            control_frame, from_=0, to=0, width=8,
            command=self.jump_to_frame
        )
        self.current_frame.pack(side=tk.LEFT, padx=5)
        self.current_frame.bind("<Return>", self.jump_to_frame)

        # Playback controls
        self.play_btn = ttk.Button(control_frame, text="▶ Play", command=self.toggle_play)
        self.play_btn.pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="⏹ Stop", command=self.stop).pack(side=tk.LEFT, padx=5)

        # Playback speed
        self.speed_var = tk.StringVar(value="1.0x")
        speed_menu = ttk.Combobox(
            control_frame, textvariable=self.speed_var,
            values=["0.25x", "0.5x", "1.0x", "2.0x", "4.0x"],
            state="readonly", width=6
        )
        speed_menu.pack(side=tk.LEFT, padx=5)
        speed_menu.bind("<<ComboboxSelected>>", self.update_speed)

        # Bounding box input fields
        bbox_input = ttk.Frame(control_frame)
        bbox_input.pack(side=tk.LEFT, padx=5)

        ttk.Label(bbox_input, text="Box Name:").grid(row=0, column=0, sticky=tk.W)
        self.bbox_name = ttk.Entry(bbox_input, width=12)
        self.bbox_name.grid(row=1, column=0, padx=2)
        
        ttk.Label(bbox_input, text="X:").grid(row=0, column=1)
        self.bbox_x = ttk.Entry(bbox_input, width=8)
        self.bbox_x.grid(row=1, column=1, padx=2)
        
        ttk.Label(bbox_input, text="Y:").grid(row=0, column=2)
        self.bbox_y = ttk.Entry(bbox_input, width=8)
        self.bbox_y.grid(row=1, column=2, padx=2)
        
        ttk.Label(bbox_input, text="Width:").grid(row=0, column=3)
        self.bbox_width = ttk.Entry(bbox_input, width=8)
        self.bbox_width.grid(row=1, column=3, padx=2)
        
        ttk.Label(bbox_input, text="Height:").grid(row=0, column=4)
        self.bbox_height = ttk.Entry(bbox_input, width=8)
        self.bbox_height.grid(row=1, column=4, padx=2)
        
        ttk.Button(bbox_input, text="Add Box", command=self.add_bounding_box).grid(row=1, column=5, padx=5)
        
        # Bounding box list
        bbox_list = ttk.Frame(control_frame)
        bbox_list.pack(side=tk.LEFT, padx=5)
        
        self.bbox_listbox = tk.Listbox(bbox_list, width=20, height=4)
        self.bbox_listbox.pack(side=tk.LEFT)
        ttk.Button(bbox_list, text="Remove", command=self.remove_selected_box).pack(side=tk.LEFT, padx=5)

        # Initial axis setup
        self.ax.set_xlabel('X Position')
        self.ax.set_ylabel('Y Position')
        self.canvas.draw()
        self.toggle_play()

    def add_frame(self, frame):
        """Add new frame to data buffer and update visualization"""
        self.data.append(frame)
        self.frame_times.append(frame['time'])
        
        # Update axis limits
        frame_added = False
        for rb in frame['rigid_bodies'].values():
            x = rb['position']['x']
            y = rb['position']['y']
            
            if x < self.x_limits[0]:
                self.x_limits[0] = x
                frame_added = True
            if x > self.x_limits[1]:
                self.x_limits[1] = x
                frame_added = True
            if y < self.y_limits[0]:
                self.y_limits[0] = y
                frame_added = True
            if y > self.y_limits[1]:
                self.y_limits[1] = y
                frame_added = True
        
        # Update plot limits if needed
        if frame_added:
            x_pad = (self.x_limits[1] - self.x_limits[0]) * 0.1
            y_pad = (self.y_limits[1] - self.y_limits[0]) * 0.1
            self.ax.set_xlim(self.x_limits[0]-x_pad, self.x_limits[1]+x_pad)
            self.ax.set_ylim(self.y_limits[0]-y_pad, self.y_limits[1]+y_pad)
        
        # Update UI elements
        self.current_frame.config(to=len(self.data)-1)
        self.check_collisions(frame)
        
        # Auto-scroll if playing
        if self.playing:
            self.current_frame_idx = len(self.data) - 1
            self.draw_frame(frame)

    def toggle_play(self):
        """Toggle playback state"""
        self.playing = not self.playing
        if self.playing:
            self.play_btn.config(text="⏸ Pause")
            self.animate()
        else:
            self.play_btn.config(text="▶ Play")

    def animate(self):
        """Drive frame animation using timestamp-based scheduling"""
        if not self.playing or self.current_frame_idx >= len(self.data)-1:
            return

        # Process next frame
        self.current_frame_idx += 1
        self.draw_frame(self.data[self.current_frame_idx])
        self.current_frame.delete(0, tk.END)
        self.current_frame.insert(0, str(self.current_frame_idx))

        # Calculate delay to next frame
        try:
            if self.current_frame_idx < len(self.data)-1:
                current_time = self.data[self.current_frame_idx]['time']
                next_time = self.data[self.current_frame_idx+1]['time']
                delay = (next_time - current_time) * 1000 / self.playback_speed
            else:
                delay = 100  # Default check interval
        except IndexError:
            delay = 100

        self.root.after(int(delay), self.animate)

    def update_speed(self, event=None):
        """Handle playback speed changes"""
        speed_map = {
            "0.25x": 0.25,
            "0.5x": 0.5,
            "1.0x": 1.0,
            "2.0x": 2.0,
            "4.0x": 4.0
        }
        self.playback_speed = speed_map[self.speed_var.get()]

    def calculate_limits(self):
        """Calculate axis limits based on all data"""
        for frame in self.data:
            for rb in frame['rigid_bodies'].values():
                x = rb['position']['x']
                y = rb['position']['y']
                
                self.x_limits[0] = min(self.x_limits[0], x)
                self.x_limits[1] = max(self.x_limits[1], x)
                self.y_limits[0] = min(self.y_limits[0], y)
                self.y_limits[1] = max(self.y_limits[1], y)
        
        # Add 10% padding
        x_pad = (self.x_limits[1] - self.x_limits[0]) * 0.1
        y_pad = (self.y_limits[1] - self.y_limits[0]) * 0.1
        self.ax.set_xlim(self.x_limits[0]-x_pad, self.x_limits[1]+x_pad)
        self.ax.set_ylim(self.y_limits[0]-y_pad, self.y_limits[1]+y_pad)
        self.ax.set_xlabel('X Position')
        self.ax.set_ylabel('Y Position')

    def is_inside_box(self, x, y, box):
        """Check if a point is inside a bounding box"""
        return (box['x'] <= x <= box['x'] + box['width']) and (box['y'] <= y <= box['y'] + box['height'])
    
    def check_collisions(self, frame):
        """Check and report collisions (modified for GUI console)"""
        current_time = frame['time']
        frame_num = frame['frame']
        
        for rb_name, rb_data in frame['rigid_bodies'].items():
            x = rb_data['position']['x']
            y = rb_data['position']['y']
            
            for box in self.bounding_boxes:
                key = (rb_name, box['name'])
                was_inside = self.collision_states.get(key, False)
                is_inside = self.is_inside_box(x, y, box)
                
                if is_inside and not was_inside:
                    self.log_message(
                        f"[Frame {frame_num}] {rb_name} entered {box['name']} "
                        f"at ({x:.2f}, {y:.2f})"
                    )
                elif not is_inside and was_inside:
                    self.log_message(
                        f"[Frame {frame_num}] {rb_name} exited {box['name']} "
                        f"at ({x:.2f}, {y:.2f})"
                    )
                
                self.collision_states[key] = is_inside

    def add_bounding_box(self):
        """Add a new bounding box from input values"""
        try:
            name = self.bbox_name.get() or f"Box {len(self.bounding_boxes)+1}"
            x = float(self.bbox_x.get())
            y = float(self.bbox_y.get())
            width = float(self.bbox_width.get())
            height = float(self.bbox_height.get())
            
            self.bounding_boxes.append({
                'name': name,
                'x': x,
                'y': y,
                'width': width,
                'height': height
            })
            self.bbox_listbox.insert(tk.END, name)
            self.draw_frame(self.data[self.current_frame_idx])
        except ValueError:
            pass

    def remove_selected_box(self):
        """Remove selected bounding box from list"""
        selection = self.bbox_listbox.curselection()
        if selection:
            index = selection[0]
            self.bbox_listbox.delete(index)
            del self.bounding_boxes[index]
            self.draw_frame(self.data[self.current_frame_idx])

    def draw_frame(self, frame):
        """Draw a single frame's data"""
        self.ax.clear()
        
        # Draw bounding boxes
        for box in self.bounding_boxes:
            rect = Rectangle(
                (box['x'], box['y']),
                box['width'],
                box['height'],
                linewidth=1,
                edgecolor='r',
                facecolor='none',
                label=box['name']
            )
            self.ax.add_patch(rect)
            # Add box name label
            self.ax.text(box['x'], box['y'], box['name'],
                       fontsize=8, color='red', ha='left', va='bottom')
        
        # Draw rigid bodies
        for rb_name, rb_data in frame['rigid_bodies'].items():
            x = rb_data['position']['x']
            y = rb_data['position']['y']
            self.ax.scatter(x, y, label=rb_name)
            self.ax.text(x, y, rb_name, fontsize=8, ha='right', va='bottom')
        
        # Maintain axis limits and labels
        self.ax.set_xlim(self.x_limits[0], self.x_limits[1])
        self.ax.set_ylim(self.y_limits[0], self.y_limits[1])
        self.ax.set_title(f"Frame: {frame['frame']} - Time: {frame['time']:.3f}s - Speed: {self.playback_speed}x")
        self.ax.legend()
        self.canvas.draw()

    def jump_to_frame(self, event=None):
        """Handle manual frame navigation"""
        try:
            new_idx = int(self.current_frame.get())
            if 0 <= new_idx < len(self.data):
                self.current_frame_idx = new_idx
                self.playing = False
                self.play_btn.config(text="▶ Play")
                self.draw_frame(self.data[new_idx])
        except ValueError:
            pass

    def log_message(self, message):
        """Update console with new messages"""
        self.console.configure(state='normal')
        self.console.insert(tk.END, message + "\n")
        self.console.configure(state='disabled')
        self.console.see(tk.END)
        
        # Trim console buffer
        if int(self.console.index('end-1c').split('.')[0]) > 1000:
            self.console.delete(1.0, 2.0)

    def stop(self):
        """Reset to initial state"""
        self.playing = False
        self.current_frame_idx = 0
        self.play_btn.config(text="▶ Play")
        if self.data:
            self.draw_frame(self.data[0])