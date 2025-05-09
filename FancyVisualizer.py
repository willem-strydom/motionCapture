import dearpygui.dearpygui as dpg
from NatNetClient import NatNetClient
import threading
from multiprocessing import Process
import textwrap
from time import time,sleep
from FeatureExtractor import Machine, Game, Player, Trial, IntegralLineOfSight, BayesianHouse
import math
import requests

class MachineManager:
    def __init__(self, logger, game_manager, plot_manager):
        self.game_manager = game_manager
        self.plot_manager = plot_manager
        self.current_selected = None
        self.logger = logger
        self.plot_tags = {}  # To store plot elements for each machine

    def create_controls(self, parent):
        with dpg.collapsing_header(label="Machine Manager", parent=parent):
            with dpg.group():
                dpg.add_input_text(tag="machine_name", hint="Name", width=120)
                dpg.add_input_float(tag="win_chance", width=80, default_value=0.5)
                
                coord_labels = [
                    ("Bottom_Left_X", "Bottom Left X"), ("Bottom_Left_Z", "Bottom Left Z"),
                    ("Bottom_Right_X", "Bottom Right X"), ("Bottom_Right_Z", "Bottom Right Z"),
                    ("Upper_Left_X", "Upper Left X"), ("Upper_Left_Z", "Upper Left Z"),
                    ("Upper_Right_X", "Upper Right X"), ("Upper_Right_Z", "Upper Right Z")
                ]
                
                for tag, label in coord_labels:
                    dpg.add_input_float(label=label, tag=tag, width=100)

                with dpg.group(horizontal=True):
                    dpg.add_button(label="Add", callback=self.add_machine)
                    dpg.add_button(label="Update", callback=self.update_machine)
                    dpg.add_button(label="Delete", callback=self.delete_machine)

                with dpg.child_window(height=200):
                    dpg.add_listbox(tag="machine_list", items=[], callback=self.select_machine)
            
            with dpg.group():
                dpg.add_button(label="Send Awareness", callback=self.game_manager.connection_manager.send_awareness)
            with dpg.group():
                dpg.add_input_text(tag="outcome_machine", label="Machine", width=120)
                dpg.add_checkbox(tag="outcome_success", label="Success", default_value=True)
                dpg.add_button(label="Send Outcome", callback=self.send_outcome)
            with dpg.group():
                dpg.add_input_text(tag="x", label="x", width=120)
                dpg.add_input_text(tag="z", label="z", width=120)
                dpg.add_input_text(tag="theta", label="theta", width=120)
                dpg.add_button(label="Send Position", callback=self.send_position)
            with dpg.group():
                dpg.add_input_text(tag="adjm1", label="adjm1", width=120)
                dpg.add_input_text(tag="adjm2", label="adjm2", width=120)
                dpg.add_input_text(tag="adjm3", label="adjm3", width=120)
                dpg.add_input_text(tag="adjm4", label="adjm4", width=120)
                dpg.add_button(label="Send Position", callback=self.send_adjustments)

    def add_machine(self):
        name = dpg.get_value("machine_name")
        win_chance = dpg.get_value("win_chance")
        coords = [dpg.get_value(tag) for tag in [
            "Bottom_Left_X", "Bottom_Left_Z",
            "Bottom_Right_X", "Bottom_Right_Z",
            "Upper_Left_X", "Upper_Left_Z",
            "Upper_Right_X", "Upper_Right_Z"
        ]]
        
        new_machine = Machine(
            name=name,
            winChance=win_chance,
            bottomLeft=(coords[0], coords[1]),
            bottomRight=(coords[2], coords[3]),
            upperLeft=(coords[4], coords[5]),
            upperRight=(coords[6], coords[7])
        )
        
        self.game_manager.game.add_machine(new_machine)
        self.update_machine_list()
        self.plot_manager.draw_machine_boundary(new_machine)
        self.logger.log_event(f"Added machine: {name}")

    def update_machine(self):
        if self.current_selected:
            try:
                machine = self.game_manager.game.machines[self.current_selected]
                old_win_chance = machine.winChance
                machine.set_win_chance(dpg.get_value("win_chance"))
                
                # Get new coordinates
                coords = [dpg.get_value(tag) for tag in [
                    "Bottom_Left_X", "Bottom_Left_Z",
                    "Bottom_Right_X", "Bottom_Right_Z",
                    "Upper_Left_X", "Upper_Left_Z",
                    "Upper_Right_X", "Upper_Right_Z"
                ]]
                
                # Update machine coordinates
                machine.bottomLeft = (coords[0], coords[1])
                machine.bottomRight = (coords[2], coords[3])
                machine.upperLeft = (coords[4], coords[5])
                machine.upperRight = (coords[6], coords[7])
                
                # Update the visual representation
                self.plot_manager.remove_machine_boundary(machine.name)
                self.plot_manager.draw_machine_boundary(machine)
                
                self.game_manager.game.machines[self.current_selected] = machine
                self.logger.log_event(f"Updated {machine.name} win chance to {machine.winChance} and coordinates")
            except KeyError:
                self.logger.log_event("No machine selected for update")

    def delete_machine(self):
        if self.current_selected:
            self.plot_manager.remove_machine_boundary(self.current_selected)
            del self.game_manager.game.machines[self.current_selected]
            self.update_machine_list()
            self.logger.log_event(f"Deleted machine: {self.current_selected}")
            self.current_selected = None

    def select_machine(self, sender, app_data):
        self.current_selected = app_data
        machine = self.game_manager.game.machines[app_data]
        dpg.set_value("machine_name", machine.name)
        dpg.set_value("win_chance", machine.winChance)
        
        coord_values = [
            machine.bottomLeft[0], machine.bottomLeft[1],
            machine.bottomRight[0], machine.bottomRight[1],
            machine.upperLeft[0], machine.upperLeft[1],
            machine.upperRight[0], machine.upperRight[1]
        ]
        
        for i, tag in enumerate([
            "Bottom_Left_X", "Bottom_Left_Z",
            "Bottom_Right_X", "Bottom_Right_Z",
            "Upper_Left_X", "Upper_Left_Z",
            "Upper_Right_X", "Upper_Right_Z"
        ]):
            dpg.set_value(tag, coord_values[i])

    def update_machine_list(self):
        dpg.configure_item("machine_list", items=list(self.game_manager.game.machines.keys()))

    def send_outcome(self, sender, app_data):
        machine = dpg.get_value("outcome_machine")
        success = dpg.get_value("outcome_success")
        self.game_manager.connection_manager.send_outcome(machine,success)

    def send_position(self, sender, app_data):
        x = dpg.get_value("x")
        z = dpg.get_value("z")
        theta = dpg.get_value("theta")
        self.game_manager.connection_manager.send_position(x,z,theta)

    def send_adjustments(self,sender,app_data):
        m1 = dpg.get_value("adjm1")
        m2 = dpg.get_value("adjm2")
        m3 = dpg.get_value("adjm3")
        m4 = dpg.get_value("adjm4")
        self.game_manager.connection_manager.send_adjustment(self.game_manager.game.get_winrates(),[float(m1),float(m2),float(m3),float(m4)])

class ConnectionManager:
    def __init__(self, logger,game_manager):
        self.connected = False
        self.thread = None
        self.logger = logger
        self.btn = None
        self.clientAddress =  "192.168.1.127"
        self.serverAddress = "10.229.139.24"
        self.url = "http://localhost:3000/"
        self.streaming_client = None
        self.game_manager = game_manager
        self.max_retries = 5
        
    def create_controls(self, parent):
        with dpg.group(parent=parent) as group:
            # Create themes first
            with dpg.theme() as red_theme:
                with dpg.theme_component(dpg.mvButton):
                    dpg.add_theme_color(dpg.mvThemeCol_Button, (200, 50, 50))
                    
            with dpg.theme() as green_theme:
                with dpg.theme_component(dpg.mvButton):
                    dpg.add_theme_color(dpg.mvThemeCol_Button, (50, 200, 50))

            # Create button with themes
            self.connection = dpg.add_button(
                label="Disconnected",
                width=200,
                callback=self.toggle_connection
            )
            dpg.bind_item_theme(self.connection, red_theme)

                        # Create button with themes
            self.trial = dpg.add_button(
                label="Not in Trial",
                width=200,
                callback=self.toggle_trial
            )
            dpg.bind_item_theme(self.trial, red_theme)
            
            # Store themes as attributes
            self.red_theme = red_theme
            self.green_theme = green_theme

    def toggle_trial(self):
        if not self.game_manager.game.inTrial:
            self.game_manager.game.start_next_trial()
            dpg.set_item_label(self.trial, "in Trial")
            dpg.bind_item_theme(self.trial, self.green_theme)
            self.send_awareness()
            self.logger.log_event("Started new trial")
        else:
            self.game_manager.game.save_current_trial()
            self.game_manager.game.inTrial = False
            dpg.set_item_label(self.trial, "Not in Trial")
            dpg.bind_item_theme(self.trial, self.red_theme)
            self.logger.log_event("Saved last trial to disk")

    def send_awareness(self):
        requests.post(self.url+"participant", json={"event": "awareness"})

    def send_outcome(self, machine, success):
        requests.post(self.url+"participant", json={"event": "outcome","machine":machine,"success":int(success)})

    def send_position(self,x,z,theta):
        requests.post(self.url+"spectator",json={"event":"playerPosition","x":float(x),"z":float(z),"theta":float(theta)})

    def send_adjustment(self,winrates,adjustments):
        requests.post(self.url+"spectator",json={"event":"machineAdjustment","winrates":winrates,"adjustments":adjustments})

    def toggle_connection(self):
        if not self.connected:
            self.streaming_client = NatNetClient()
            self.streaming_client.set_client_address(self.clientAddress)
            self.streaming_client.set_server_address(self.serverAddress)
            self.streaming_client.set_use_multicast(True)

            # Configure the streaming client to call our data handlers.
            self.streaming_client.new_frame_listener = self.game_manager.receive_new_frame
            
            # This will run perpetually, and operate on a separate thread.
            try:
                self.connected = self.streaming_client.run()
                if not self.connected:
                    self.logger.log_event("ERROR: Could not start streaming client.")
                    self.streaming_client = None
                else:
                    sleep(0.1)
                    if self.streaming_client.is_alive() is False:
                        self.connected = False
                        tries = 1
                        while (tries < self.max_retries and not self.streaming_client.is_alive()):
                            self.connected = self.streaming_client.run()
                            tries += 1
                            sleep(0.1)
                        if not self.streaming_client.is_alive():
                            self.logger.log_event(f"ERROR: Could not connect to streaming client, attempted {self.max_retries} times")
                        else:
                            dpg.set_item_label(self.connection, f"Connected!")
                            dpg.bind_item_theme(self.connection, self.green_theme)
                            self.streaming_client.set_print_level(0)
                            self.logger.log_event(f"Connected to server (after {tries} failed attempts)")
                    else:
                        dpg.set_item_label(self.connection, "Connected!")
                        dpg.bind_item_theme(self.connection, self.green_theme)
                        self.streaming_client.set_print_level(0)
                        self.logger.log_event("Connected to server")
            except Exception as e:
                self.logger.log_event(f"ERROR: starting streaming client threw: {e}")
        else:
            self.streaming_client.shutdown()
            self.streaming_client = None
            dpg.set_item_label(self.connection, "Disconnected")
            dpg.bind_item_theme(self.connection, self.red_theme)
            self.logger.log_event("Disconnected from server")
            self.connected = False

class PlotManager:
    def __init__(self, logger):
        self.plot_tags = {}
        self.logger = logger

    def create_plot(self, parent,sheight):
        with dpg.plot(label="MoCap Arena", parent=parent, height=sheight, width=-1, tag="main_plot"):
            dpg.add_plot_axis(dpg.mvYAxis, label="X", tag="x_axis", invert=False)
            dpg.add_plot_axis(dpg.mvXAxis, label="Z", tag="z_axis", invert=False)
            
            dpg.set_axis_limits("x_axis", -2, 1.5)
            dpg.set_axis_limits("z_axis", -1.6, 1.6)
            
            # Red dot
            with dpg.theme() as red_theme:
                with dpg.theme_component(dpg.mvScatterSeries):
                    dpg.add_theme_color(dpg.mvPlotCol_Line, (255, 0, 0))
                    dpg.add_theme_style(dpg.mvPlotStyleVar_Marker, dpg.mvPlotMarker_Circle)
            dpg.add_scatter_series([2], [2], parent="x_axis", tag="red_dot")
            dpg.bind_item_theme("red_dot", red_theme)
    
    def draw_machine_boundary(self, machine):
        """Draw a bounding box with machine name annotation"""
        import random
        r = random.randint(50, 200)
        g = random.randint(50, 200)
        b = random.randint(50, 200)
        
        name_safe = machine.name.replace(" ", "_")
        boundary_tag = f"boundary_{name_safe}"
        annotation_tag = f"annotation_{name_safe}"
        
        # Get coordinates
        bl = machine.bottomLeft
        br = machine.bottomRight
        ul = machine.upperLeft
        ur = machine.upperRight
        
        # Store tags for later deletion
        self.plot_tags[machine.name] = (boundary_tag, annotation_tag)

        # Draw outline (boundary)
        with dpg.theme() as line_theme:
            with dpg.theme_component(dpg.mvLineSeries):
                dpg.add_theme_color(dpg.mvPlotCol_Line, (r, g, b, 255))
                dpg.add_theme_style(dpg.mvPlotStyleVar_LineWeight, 2.0)
        
        # Create closed boundary
        x_points = [bl[0], br[0], ur[0], ul[0], bl[0]]
        z_points = [bl[1], br[1], ur[1], ul[1], bl[1]]
        dpg.add_line_series(z_points, x_points, parent="x_axis", tag=boundary_tag)
        dpg.bind_item_theme(boundary_tag, line_theme)
        
        # Calculate center position for annotation
        avg_x = sum(point[0] for point in [bl, br, ur, ul]) / 4
        avg_z = sum(point[1] for point in [bl, br, ur, ul]) / 4
        
        # Add text annotation directly to the plot (not to axis)
        dpg.add_plot_annotation(
            label=machine.name,
            default_value=(avg_z,avg_x),
            parent="main_plot",  # Changed to plot's tag
            tag=annotation_tag,
            color=(r, g, b, 255),
            offset=(5, 5)
        )
        
        self.logger.log_event(f"Drew boundary for machine: {machine.name}")

    def remove_machine_boundary(self, machine_name):
        """Remove the boundary and annotation for a machine"""
        if machine_name in self.plot_tags:
            boundary_tag, annotation_tag = self.plot_tags[machine_name]
            try:
                dpg.delete_item(boundary_tag)
                dpg.delete_item(annotation_tag)
                del self.plot_tags[machine_name]
                self.logger.log_event(f"Removed boundary for machine: {machine_name}")
            except Exception as e:
                self.logger.log_event(f"Error removing boundary: {str(e)}")

    def draw_foyer_line(self,right_point, left_point):
        """Renders the foyer dision line"""
        z_points = [left_point[1],right_point[1]]
        x_points = [left_point[0],right_point[0]]
        dpg.add_line_series(z_points, x_points, parent="x_axis", tag="foyer_line")

    def update_direction_arrow(self, base_z, base_x, theta, arrow_length=0.1):
        """
        Draw or update an arrow indicating the direction.
        World coordinates: (position_x, position_z)
        Plot coordinates: (base_z, base_x)
        Assuming theta is measured in world coordinates where 0 rad points to positive x.
        Arrow tip is calculated by:
            tip_x (world) = position_x + cos(theta) * arrow_length
            tip_z (world) = position_z + sin(theta) * arrow_length
        Mapped to plot: (tip_z, tip_x)
        """
        tip_world_x = base_x + math.cos(theta) * arrow_length
        tip_world_z = base_z + math.sin(theta) * arrow_length

        # Create data for the line series: from base to tip in plot coordinates
        x_vals = [base_z, tip_world_z]
        y_vals = [base_x, tip_world_x]

        if dpg.does_item_exist("direction_arrow"):
            dpg.set_value("direction_arrow", [x_vals, y_vals])
        else:
            # Add arrow as a line series with an arrow-like theme
            with dpg.theme() as arrow_theme:
                with dpg.theme_component(dpg.mvLineSeries):
                    dpg.add_theme_color(dpg.mvPlotCol_Line, (0, 0, 255, 255))
                    dpg.add_theme_style(dpg.mvPlotStyleVar_LineWeight, 3.0)
            dpg.add_line_series(x_vals, y_vals, parent="x_axis", tag="direction_arrow")
            dpg.bind_item_theme("direction_arrow", arrow_theme)
        #self.logger.log_event(f"Updated direction arrow on plot for pitch:{theta}")
        # ^ useful for getting offset. might move to dedicated "pre-trial calibration" pipeline

class AppLogger:
    def __init__(self):
        self.log_content = []

    def create_logger(self, parent):
        dpg.add_input_text(
            multiline=True,
            readonly=True,
            tracked=True,
            tag="log_output",
            width=-1,
            height=-1
        )

    def log_event(self, message):
        timestamp = f"[{time()}]"
        wrapped = textwrap.fill(f"{timestamp} {message}", width=100)
        self.log_content.append(wrapped)
        dpg.set_value("log_output", "\n".join(self.log_content[-100:]))

class GameManager:
    def __init__(self, logger, machine_manager, plot_manager,connection_manager):
        self.logger = logger
        self.machine_manager = machine_manager
        self.plot_manager = plot_manager
        self.connection_manager = connection_manager
        self.game = Game(2,0.1,BayesianHouse(),IntegralLineOfSight())

    def configure_mocap_enviorement(self):
        self.game.set_foyer_line([[-0.869957,-1.361096],[-0.796456,1.429263]])
        self.plot_manager.draw_foyer_line([-0.869957,-1.361096],[-0.796456,1.429263])
        m1 = Machine("m1",0.5,[0.914461,-0.416378],[0.897491,-0.716809],[0.615168,-0.399924],[0.600168,-0.699141])
        m2 = Machine("m2",0.5,[0.930728,-0.117966],[0.914461,-0.416378],[0.634181,-0.100121],[0.615168,-0.399924])
        m3 = Machine("m3",0.5,[0.945728,0.181054],[0.930728,-0.117966],[0.652421,0.196054],[0.634181,-0.100121])
        m4 = Machine("m4",0.5,[0.975101,0.483173],[0.945728,0.181054],[0.669682,0.497284],[0.652421,0.196054])
        self.game.add_machine(m1)
        self.plot_manager.draw_machine_boundary(m1)
        self.game.add_machine(m2)
        self.plot_manager.draw_machine_boundary(m2)
        self.game.add_machine(m3)
        self.plot_manager.draw_machine_boundary(m3)
        self.game.add_machine(m4)
        self.plot_manager.draw_machine_boundary(m4)
        self.machine_manager.update_machine_list()

    def receive_new_frame(self, data_dict, mocap_data):
        # Process game logic
        self.game.receive_new_frame(data_dict, mocap_data)
        # Update rigid body visualization        
        self.update_rigid_body_visual(self.game.get_last_processed_frame())
        if not self.game.behindFoyer:
            self.logger.log_event("Left Foyer!")
            self.game.behindFoyer = True
        if self.game.playMachine is not None and self.game.inTrial:
            current_trial = self.game.trials[-1]
            if current_trial.get_outcome() is not None:
                machine_name = next(iter(current_trial.get_outcome().keys()))
                win_loss = next(iter(current_trial.get_outcome().values()))
                self.logger.log_event(f"Collision detected with {machine_name} and outcome was {win_loss}")
                self.connection_manager.send_outcome(machine_name,win_loss)
                self.connection_manager.toggle_trial()
                #self.game.inTrial = False
        
    def update_rigid_body_visual(self, processed_data):
        # Low poll rate update for spectator website
        if processed_data:
            if processed_data['time'] % 200 == 0:
                self.connection_manager.send_position(processed_data['position_x'],processed_data['position_z'], processed_data['theta'])
            # Update red dot (plot expects [position_z] and [position_x])
            dpg.set_value("red_dot", [[processed_data['position_z']], [processed_data['position_x']]])
            # Update direction arrow using processed theta.
            # Mapping: world (x, z) -> plot (position_z, position_x)
            self.plot_manager.update_direction_arrow(
                base_z=processed_data['position_z'],
                base_x=processed_data['position_x'],
                theta=processed_data["theta"]
            )

class MainApp:
    def __init__(self):
        dpg.create_context()
        self.logger = AppLogger()
        self.plot_manager = PlotManager(self.logger)
        self.game_manager = GameManager(self.logger, None, self.plot_manager,None)
        self.machine_manager = MachineManager(self.logger, self.game_manager, self.plot_manager)
        self.connection_manager = ConnectionManager(self.logger, self.game_manager)
        
        self.game_manager.machine_manager = self.machine_manager
        self.game_manager.connection_manager = self.connection_manager
        self.create_layout()
        self.game_manager.configure_mocap_enviorement()
        dpg.setup_dearpygui()
        dpg.show_viewport()

    def create_layout(self):
        dpg.create_viewport(title="Machine Manager", width=1200, height=800)
        
        with dpg.window(tag="Primary Window"):
            main_group = dpg.add_group(horizontal=True)
            
            # Left panel
            with dpg.child_window(width=400, parent=main_group) as left_panel:
                self.connection_manager.create_controls(left_panel)
                self.machine_manager.create_controls(left_panel)
            
            # Right panel
            with dpg.child_window(width=-1, parent=main_group) as right_panel:
                self.plot_manager.create_plot(right_panel, 600)
                self.logger.create_logger(right_panel)

        dpg.set_primary_window("Primary Window", True)

    def run(self):
        dpg.start_dearpygui()
        dpg.destroy_context()

if __name__ == "__main__":
    gui = MainApp()
    gui.run()