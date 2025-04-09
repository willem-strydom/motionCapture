import MoCapData
import random
import json
import matlab.engine
import os
import math
import numpy as np
from MocapModels import IntegralLineOfSight
from BehavioralModels import BayesianHouse
import requests

class Machine:

    def __init__(self,name,winChance,bottomLeft,bottomRight,upperLeft,upperRight):
        self.name = name
        self.bottomLeft = bottomLeft
        self.bottomRight = bottomRight
        self.upperLeft = upperLeft
        self.upperRight = upperRight
        self.winChance = winChance

    def set_win_chance(self,updated_rate):
        self.winChance = updated_rate

    def find_center(self):
        xCenter = (self.bottomLeft[0]+self.bottomRight[0]+self.upperLeft[0]+self.upperRight[0])/4
        zCenter = (self.bottomLeft[1]+self.bottomRight[1]+self.upperLeft[1]+self.upperRight[1])/4
        return (xCenter,zCenter)

    def get_name(self):
        return self.name

    def get_win_chance(self):
        return self.winChance
    
    def check_collision(self,playerXY):
        # assuming x decreases as you go closer and z decreases as you go left
        above_bottomLeft = playerXY[0] <= self.bottomLeft[0] and playerXY[2] <= self.bottomLeft[1]
        above_bottomRight = playerXY[0] <= self.bottomRight[0] and playerXY[2] >= self.bottomRight[1]
        below_upperLeft = playerXY[0] >= self.upperLeft[0] and playerXY[2] <= self.upperLeft[1]
        below_upperRight = playerXY[0] >= self.upperLeft[0] and playerXY[2] >= self.upperRight[1]
        return above_bottomLeft and above_bottomRight and below_upperLeft and below_upperRight

class Player:

    def __init__(self):
        self.history = []
        self.play_history = []
        self.win_history = []

    def add_to_history(self,trial):
        self.history.append(trial)
        self.play_history.append(next(iter(trial.get_outcome().keys())))
        self.win_history.append(next(iter(trial.get_outcome().values())))

    def get_play_history(self):
        return self.play_history
    
    def get_win_history(self):
        return self.win_history

    def write_history_to_file(self, filename="player_history.json"):
        history_data = []
        for trial in self.history:
            trial_data = {
                "mocap_prediction": trial.get_mocap_prediction(),
                "behavioral_prediction": trial.get_behavioral_prediction(),
                "outcome": trial.get_outcome(),
                "pre_win_rates": trial.get_pre_win_rates(),
                "post_mocap_win_rates": trial.get_post_mocap_win_rates(),
                "post_behavioral_win_rates": trial.get_post_behavioral_win_rates(),
                "foyer": trial.get_foyer(),
                "walk": trial.get_walk()
            }
            history_data.append(trial_data)
        
        with open(filename, "w") as file:
            json.dump(history_data, file, indent=4)

        print(f"Player history successfully written to {filename}")

    def write_trial_to_file(self,trial, filename="player_history.json"):
        history_data = []

        trial_data = {
            "mocap_prediction": trial.get_mocap_prediction(),
            "behavioral_prediction": trial.get_behavioral_prediction(),
            "outcome": trial.get_outcome(),
            "pre_win_rates": trial.get_pre_win_rates(),
            "post_mocap_win_rates": trial.get_post_mocap_win_rates(),
            "post_behavioral_win_rates": trial.get_post_behavioral_win_rates(),
            "foyer": trial.get_foyer(),
            "walk": trial.get_walk()
        }
        history_data.append(trial_data)
        
        with open(filename, "w") as file:
            json.dump(history_data, file, indent=4)

        print(f"Trial successfully written to {filename}")

class Trial:

    def __init__(self,machines):
        self.foyer = {}
        self.walk = {}
        self.prePredictionWinRates = {}
        self.outcome = None
        self.mocap_prediction = None
        self.behavioral_prediction = None
        for name,machine in machines.items():
            self.prePredictionWinRates[name] = machine.get_win_chance()
        self.postMocapPredictionWinRates = {}
        self.postBehavioralPredictionWinRates = {}
        
    def evaluate(self,choice):
        rollValue = random.random()
        self.outcome = {}
        for (name,winrate),idx in zip(self.postMocapPredictionWinRates.items(),range(len(self.postMocapPredictionWinRates))):
            if idx == choice:
                self.outcome = {name:bool(rollValue < winrate)}
        return self.outcome

    def update_foyer(self,timestamp,rigidBodies):
        self.foyer.update({timestamp:rigidBodies})

    def update_walk(self,timestamp,rigidBodies):
        self.walk.update({timestamp:rigidBodies})

    def check_out_of_foyer(self,timestamp,foyer_line):
        # foyer_line should be [(x,z),(x,z)]
        processed_data = self.foyer.get(timestamp)
        current_position = [processed_data['position_x'],processed_data['position_y'], processed_data['position_z']]
        # assuming x decreases as you get closer and z increases as you go left
        over_right = current_position[0] >= foyer_line[0][0] and current_position[2] >= foyer_line[0][1]
        over_left = current_position[0] >= foyer_line[1][0] and current_position[2] <= foyer_line[1][1]
        return over_left or over_right
            
    def set_mocap_prediction(self,prediction,adjustments):
        self.mocap_prediction = prediction
        for (name,winRate),delta in zip(self.prePredictionWinRates.items(),adjustments):
            new_rate = winRate + delta
            self.postMocapPredictionWinRates[name] = new_rate

    def set_behavioral_prediction(self,prediction,adjustments):
        self.behavioral_prediction = prediction
        for (name,winRate),delta in zip(self.postMocapPredictionWinRates.items(),adjustments):
            new_rate = winRate + delta
            self.postBehavioralPredictionWinRates[name] = new_rate
    
    def get_mocap_prediction(self):
        return self.mocap_prediction
    
    def get_behavioral_prediction(self):
        return self.behavioral_prediction
    
    def get_outcome(self):
        return self.outcome
    
    def get_pre_win_rates(self):
        return self.prePredictionWinRates
    
    def get_post_mocap_win_rates(self):
        return self.postMocapPredictionWinRates
    
    def get_post_behavioral_win_rates(self):
        return self.postBehavioralPredictionWinRates
    
    def get_foyer(self):
        return self.foyer
    
    def get_walk(self):
        return self.walk
    
    def get_prev_foyer_frame(self):
        if not self.foyer: 
            return None
        #print(self.foyer)
        return self.foyer.get(next(reversed(self.foyer)))

    def get_specific_walk_pos(self,timestamp):
        processed_data = self.walk.get(timestamp)
        current_position = [processed_data['position_x'],processed_data['position_y'], processed_data['position_z']]
        return current_position
    
    def get_name(self):
        if (len(self.foyer)!=0):
            return next(iter(self.foyer.keys()))
        else:
            return np.random.randint(1,10)

class Game:

    def __init__(self,B,M,behavioralModel,mocapModel):
        self.machines = {}
        self.trials = []
        self.player = Player()
        self.maxProbabilityAdjustment = M
        self.houseAdvantage = B
        self.foyer_line = None
        self.inTrial = False
        self.behindFoyer = True
        self.playMachine = None
        self.behavioralModel = behavioralModel
        self.mocapModel = mocapModel
        #self.eng = matlab.engine.start_matlab()     # initialize matlab connection
        #self.eng.addpath(os.getcwd(),nargout=1)       # add current working director to matlab path to access local functions
        self.lastProcessedFrame = None
        self.featureExtractor = FeatureExtractor()
        #print(self.eng.which('process_frame', nargout=1))
        #print(os.getcwd())

    def shut_down(self):
        #self.eng.rmpath(os.getcwd(),nargout=0)
        #self.eng.quit()
        pass
    
    def adjust_probabilities(self,delta_list):
        for machine, delta in zip(self.machines.values(), delta_list):
            new_rate = machine.get_win_chance() + delta
            machine.set_win_chance(new_rate)

    def get_winrates(self):
        winrates = []
        for name,machine in iter(self.machines.items()):
            winrates.append(float(machine.get_win_chance()))
        return winrates

    def add_machine(self,machine):
        self.machines[machine.get_name()] = machine
        self.featureExtractor.add_machine(machine)

    def set_foyer_line(self,foyer_line):
        # [Left (x,z), Right (x,z)]
        self.foyer_line = foyer_line

    def start_next_trial(self):
        self.trials.append(Trial(self.machines))
        self.inTrial = True
        self.behindFoyer = True
        self.playMachine = None
        
    def save_current_trial(self):
        trial = self.trials[-1]
        self.player.write_trial_to_file(trial,f"trial_{trial.get_name()}.json")

    def get_last_processed_frame(self):
        return self.lastProcessedFrame

    def receive_new_frame(self, data_dict, mocap_data):
        timestamp = data_dict["frame_number"]
        streamed_data = self.featureExtractor.parse_mocap_data(mocap_data)
        if (len(self.trials)>0):
            trial = self.trials[-1] # most recent trial
            processed_data = self.featureExtractor.process_streamed_data(timestamp=timestamp,streamed_data=streamed_data,previous_data=trial.get_prev_foyer_frame())
        else:
            trial = None
            processed_data = self.featureExtractor.process_streamed_data(timestamp=timestamp,streamed_data=streamed_data,previous_data=None)
        self.lastProcessedFrame = processed_data
        if not self.inTrial:
            return
        if not trial.get_walk(): # we are still in foyer
            trial.update_foyer(timestamp,processed_data)
            if trial.check_out_of_foyer(timestamp,self.foyer_line): # we have JUST left foyer
                self.behindFoyer = False
                adjustments = self.mocapModel.adjust_winrates(trial.get_foyer(),self.player.get_play_history(),self.player.get_win_history(),self.get_winrates())
                #print(f"trial:{trial.get_foyer().keys()}")
                #print(f"adjustments: {adjustments}")
                adjustments = list(adjustments)
                self.adjust_probabilities(adjustments)
                min = 0
                for i in range(len(adjustments)):
                    if (adjustments[i] < adjustments[min]):
                        min = i
                trial.set_mocap_prediction(min,adjustments)
                requests.post("http://localhost:3000/spectator",json={"event":"machineAdjustment","winrates":self.get_winrates(),"adjustments":adjustments})
                # update walk history so we can later identify the "crossing point" by matching timestamps between arrays
                trial.update_walk(timestamp,processed_data)
        else:
            trial.update_walk(timestamp,processed_data)
            playerXYZ = trial.get_specific_walk_pos(timestamp)
            playMachine = None
            for (name,machine),idx in zip(self.machines.items(),range(len(self.machines.items()))):
                if machine.check_collision(playerXYZ) and playMachine == None:
                    playMachine = idx
            if playMachine != None:
                trial.evaluate(playMachine)
                playHistory = self.player.get_play_history()
                winHistory = self.player.get_win_history()
                outcome = trial.get_outcome()
                #print(outcome)
                playHistory.append(playMachine)
                winHistory.append(next(iter(outcome.values())))
                winrateAdjustments = self.behavioralModel.adjust_winrates(playHistory,winHistory,self.get_winrates())
                self.adjust_probabilities(winrateAdjustments)
                
                try:
                    requests.post("http://localhost:3000/spectator",json={"event":"machineAdjustment","winrates":self.get_winrates(),"adjustments":winrateAdjustments.tolist()})
                except Exception as e:
                    print(f"error: {e}")
                    print(f"winrates:{self.get_winrates()}, adjustments:{winrateAdjustments}")

                min = 0
                for i in range(len(winrateAdjustments)):
                    if (winrateAdjustments[i] < winrateAdjustments[min]):
                        min = i
                trial.set_behavioral_prediction(min,winrateAdjustments)
                self.player.add_to_history(trial)
                self.playMachine = playMachine
                #self.inTrial = False
    
class FeatureExtractor:
    def __init__(self):
        self.MACHINE_CENTERS = {}
    
    def add_machine(self,machine):
        self.MACHINE_CENTERS.update({machine.get_name():machine.find_center()})

    def quaternion_to_forward(self,qw, qx, qy, qz):
        """
        Rotate the default forward vector (1, 0, 0) by the quaternion.
        The formula below assumes the quaternion is normalized.
        """
        # Using quaternion-vector multiplication:
        # v' = q * v * q_conjugate, with v = (0, 1, 0, 0) for forward vector (1,0,0)
        # But for efficiency, the resulting forward vector can be computed as:
        vx = 1- 2 * (qy * qy + qz * qz)
        vy = 2 * (qx * qy - qw * qz)
        vz = 2 * (qx * qz + qw * qy)
        return vx, vy, vz

    def get_view_angle(self,qw, qx, qy, qz):
        # Get the forward vector after rotation
        vx, vy, vz = self.quaternion_to_forward(qw, qx, qy, qz)
        
        # Compute the angle in the x-z plane. Note: adjust arguments if your coordinate system differs.
        angle = math.atan2(vz, vx)  # returns angle in [-pi, pi]
        
        # Convert to 0 - 2pi
        if angle < 0:
            angle += 2 * math.pi
        # correct the angle with a mesured offset (whatever it reports when you are looking at positive x), for my tests it was -1.8 (offset is negated)
        # we discovered if you create the rigid body with the participant looking forward, this is not needed.
        corrected_angle = (-angle) % (2 * math.pi)
        return corrected_angle

    def compute_theta_dot(self,theta, prev_theta, dt):
        delta = (theta - prev_theta) % (2 * math.pi)
        if delta > math.pi:
            delta -= 2 * math.pi
        return delta / dt

    def compute_velocity(self,pos, prev_pos, dt):
        return [(pos[i] - prev_pos[i]) / dt for i in range(len(pos))]

    def compute_acceleration(self,vel, prev_vel, dt):
        return [(vel[i] - prev_vel[i]) / dt for i in range(len(vel))]

    def compute_theta_to_machines(self,position, theta_angle):
        theta_vec = [math.cos(theta_angle), math.sin(theta_angle)]
        results = {}

        for name, (mx, mz) in self.MACHINE_CENTERS.items():
            dx = mx - position[0]
            dz = mz - position[2]
            machine_vec = [dx, dz]

            norm_product = math.hypot(*theta_vec) * math.hypot(dx, dz)
            if norm_product > 0:
                dot_product = theta_vec[0] * machine_vec[0] + theta_vec[1] * machine_vec[1]
                angle = math.acos(dot_product / norm_product)
                results[name] = math.degrees(angle)
            else:
                results[name] = 0.0

        return results
    
    def parse_mocap_data(self,mocap_data):
        rigid_data = mocap_data.get_rigid_data()
        rigid_list = rigid_data.get_rigid_list()
        rigid_body_list = None
        if rigid_data.get_rigid_body_count() > 0:
            rigid_body_list = {
                            'position_x':rigid_list[0].get_position()[0],
                            'position_y':rigid_list[0].get_position()[1],
                            'position_z':rigid_list[0].get_position()[2],
                            'rotation_x':rigid_list[0].get_rotation()[0],
                            'rotation_y':rigid_list[0].get_rotation()[1],
                            'rotation_z':rigid_list[0].get_rotation()[2],
                            'rotation_w':rigid_list[0].get_rotation()[3]
                            }
                
        return rigid_body_list  

    def process_streamed_data(self,streamed_data, previous_data, timestamp):
        if not streamed_data:
            return
        streamed_data.update({'time': timestamp})

        # Extract position and quaternion
        pos = [streamed_data['position_x'], streamed_data['position_y'], streamed_data['position_z']]
        quat = [streamed_data['rotation_w'], streamed_data['rotation_x'],
                streamed_data['rotation_y'], streamed_data['rotation_z']]

        # θ (yaw angle in radians)
        theta = self.get_view_angle(*quat)
        streamed_data.update({'theta': theta})

        # Velocity & Acceleration
        if previous_data:
            dt = timestamp - previous_data['time']
            prev_pos = [previous_data['position_x'], previous_data['position_y'], previous_data['position_z']]
            vel = self.compute_velocity(pos, prev_pos, dt)
            streamed_data.update({
                'velocity_x': vel[0],
                'velocity_z': vel[2]
            })

            if 'velocity_x' in previous_data and 'velocity_z' in previous_data:
                prev_vel = [previous_data['velocity_x'], 0, previous_data['velocity_z']]
                acc = self.compute_acceleration(vel, prev_vel, dt)
                streamed_data.update({
                    'acceleration_x': acc[0],
                    'acceleration_z': acc[2]
                })

            if 'theta' in previous_data:
                theta_dot = self.compute_theta_dot(theta, previous_data['theta'], dt)
                streamed_data.update({'theta_dot': theta_dot})

                if 'theta_dot' in previous_data:
                    theta_dot_dot = (theta_dot - previous_data['theta_dot']) / dt
                    streamed_data.update({'theta_dot_dot': theta_dot_dot})

        # θ_1 through θ_4: angle to each machine
        angle_dict = self.compute_theta_to_machines(pos, theta)
        for i, name in enumerate(self.MACHINE_CENTERS.keys(), 1):
            streamed_data.update({f'theta_{i}': angle_dict.get(name, 0)})

        return streamed_data

if __name__ == "__main__":
    game = Game(2,0.1,BayesianHouse(),IntegralLineOfSight())
    m1 = Machine("m1",0.5,[0.914461,-0.416378],[0.897491,-0.716809],[0.615168,-0.399924],[0.600168,-0.699141])
    m2 = Machine("m2",0.5,[0.930728,-0.117966],[0.914461,-0.416378],[0.634181,-0.100121],[0.615168,-0.399924])
    m3 = Machine("m3",0.5,[0.945728,0.181054],[0.930728,-0.117966],[0.652421,0.196054],[0.634181,-0.100121])
    m4 = Machine("m4",0.5,[0.975101,0.483173],[0.945728,0.181054],[0.669682,0.497284],[0.652421,0.196054])
    game.add_machine(m1)
    game.add_machine(m2)
    game.add_machine(m3)
    game.add_machine(m4)
    game.start_next_trial()
    game.trials[-1].set_mocap_prediction(2,[0,0.1,-0.1,0])
    game.trials[-1].evaluate(2)
    game.trials[-1].set_behavioral_prediction(3,[0,0.1,-0.1,0])
    game.save_current_trial()
    #print(game.trials[-1].get_outcome())