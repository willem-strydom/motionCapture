import MoCapData
import random
import json
import matlab.engine
import os
import math
import numpy as np
from scipy.stats import beta

MACHINE_CENTERS = {
    "machine1": (0.914461, -0.416378),
    "machine2": (0.930728, -0.117966),
    "machine3": (0.945728,  0.181054),
    "machine4": (0.975101,  0.483173)
}

def compute_theta_dot(theta, prev_theta, dt):
    delta = (theta - prev_theta) % (2 * math.pi)
    if delta > math.pi:
        delta -= 2 * math.pi
    return delta / dt

def compute_velocity(pos, prev_pos, dt):
    return [(pos[i] - prev_pos[i]) / dt for i in range(len(pos))]

def compute_acceleration(vel, prev_vel, dt):
    return [(vel[i] - prev_vel[i]) / dt for i in range(len(vel))]

def compute_theta_to_machines(position, theta_angle):
    theta_vec = [math.cos(theta_angle), math.sin(theta_angle)]
    results = {}

    for name, (mx, mz) in MACHINE_CENTERS.items():
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

    def get_name(self):
        return self.name

    def get_win_chance(self):
        return self.winChance
    
    def check_collision(self,playerXY):
        # assuming x decreases as you go deeper and z decreases as you go right
        above_bottomLeft = playerXY[0] <= self.bottomLeft[0] and playerXY[2] <= self.bottomLeft[1]
        above_bottomRight = playerXY[0] <= self.bottomRight[0] and playerXY[2] >= self.bottomRight[1]
        below_upperLeft = playerXY[0] >= self.upperLeft[0] and playerXY[2] <= self.upperLeft[1]
        below_upperRight = playerXY[0] >= self.upperLeft[0] and playerXY[2] >= self.upperRight[1]
        return above_bottomLeft and above_bottomRight and below_upperLeft and below_upperRight

class Player:

    def __init__(self):
        self.history = []
        self.play_history = []

    def add_to_history(self,trial):
        self.history.append(trial)
        self.play_history.append(trial.get_outcome)

    def get_play_history(self):
        return self.play_history.keys()
    
    def get_win_history(self):
        return self.play_history.values()

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
                self.outcome = {name:(rollValue < winrate)}
            else:
                print(idx)
        return self.outcome

    def update_foyer(self,timestamp,rigidBodies):
        self.foyer.update({timestamp:rigidBodies})

    def update_walk(self,timestamp,rigidBodies):
        self.walk.update({timestamp:rigidBodies})

    def check_out_of_foyer(self,timestamp,foyer_line):
        # foyer_line should be [(x,z),(x,z)]
        processed_data = self.foyer.get(timestamp)
        #print(processed_data)
        #print(self.get_foyer())
        # assuming parent rigid body is first
        current_position = [processed_data['position_x'],processed_data['position_y'], processed_data['position_z']]
        # assuming x decreases as you go down and z decreases as you go left
        over_right = current_position[0] >= foyer_line[0][0] and current_position[2] >= foyer_line[0][1]
        over_left = current_position[0] >= foyer_line[1][0] and current_position[2] <= foyer_line[1][1]
        return over_left or over_right
            
    def set_mocap_prediction(self,prediction,adjustments):
        self.mocap_prediction = prediction
        print(self.prePredictionWinRates)
        for (name,winRate),delta in zip(self.prePredictionWinRates.items(),adjustments):
            new_rate = winRate + delta
            self.postMocapPredictionWinRates[name] = new_rate

    def set_behavioral_prediction(self,prediction,adjustments):
        self.behavioral_prediction = prediction
        print(self.prePredictionWinRates)
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
        self.for_training = False                    # defaulting to True to minimize impact on GUI
        self.behavioralModel = behavioralModel
        self.mocapModel = mocapModel
        self.eng = matlab.engine.start_matlab()     # initialize matlab connection
        self.eng.addpath(os.getcwd(),nargout=1)       # add current working director to matlab path to access local functions
        self.lastProcessedFrame = None
        #print(self.eng.which('process_frame', nargout=1))
        #print(os.getcwd())

    def shut_down(self):
        self.eng.rmpath(os.getcwd(),nargout=0)
        self.eng.quit()

    def old_adjust_probabilities(self,prediction,delta_list):
        if self.for_training:
            return
        self.prediction = prediction
        pre_sum = 0
        post_sum = 0
        for name,winRate in delta_list.items():
            pre_sum += self.machines[name].get_win_chance()
            new_rate = winRate + self.machines[name].get_win_chance()
            self.machines[name].set_win_chance(new_rate)
            post_sum += new_rate
        if pre_sum != post_sum:
            self.machines[prediction].set_win_chance(self.machines[prediction].get_win_chance()+ pre_sum - post_sum)
    
    def adjust_probabilities(self,delta_list):
        for machine, delta in zip(self.machines.values(), delta_list):
            new_rate = machine.get_win_chance() + delta
            machine.set_win_chance(new_rate)

    def get_winrates(self):
        winrates = []
        for name,machine in iter(self.machines.items()):
            winrates.append(machine.get_win_chance())
        return winrates

    def determine_adjustment(self,predicted_choice):
        if self.for_training: # give the players a changing enviorment to react to, without any house model assumptions
            training_adjustments = [0,0,0,0]
        naiveDelta = self.maxProbabilityAdjustment / (len(self.machines)-1) # dont play w 1 machine, will divide by 0
        adjustments = {}
        sum = 0
        i = 0
        for name,machine in self.machines.items():
            if self.for_training:
                adjustments.update({name:training_adjustments[i]})
                i += 1
            elif not name == predicted_choice:
                adjustments.update({name:naiveDelta})
                sum += naiveDelta

        if not self.for_training:
            adjustments.update({predicted_choice: -sum})
        return adjustments

    def add_machine(self,machine):
        self.machines[machine.get_name()] = machine

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
        #print(f"reciefed frame {timestamp}")
        #print(mocap_data)
        streamed_data = self.parse_mocap_data(mocap_data)
        if (len(self.trials)>0):
            trial = self.trials[-1] # most recent trial
            processed_data = self.process_streamed_data(timestamp=timestamp,streamed_data=streamed_data,previous_data=trial.get_prev_foyer_frame())
        else:
            trial = None
            processed_data = self.process_streamed_data(timestamp=timestamp,streamed_data=streamed_data,previous_data=None)
        self.lastProcessedFrame = processed_data
        if not self.inTrial:
            return
        #print(processed_data)
        if not trial.get_walk(): # we are still in foyer
            trial.update_foyer(timestamp,processed_data)
            if trial.check_out_of_foyer(timestamp,self.foyer_line): # we have JUST left foyer
                self.behindFoyer = False
                # placeholder pipeline for predicting the machine to be played
                adjustments = self.mocapModel.adjust_winrates(trial.get_foyer(),self.player.get_play_history,self.player.get_win_history,self.get_winrates)
                self.adjust_probabilities(adjustments)
                min = 0
                for i in range(len(adjustments)):
                    if (adjustments[i] < adjustments[min]):
                        min = i
                trial.set_mocap_prediction(min,adjustments)
                # update walk history so we can later identify the "crossing point" by matching timestamps between arrays
                trial.update_walk(timestamp,processed_data)
        else:
            trial.update_walk(timestamp,processed_data)
            playerXYZ = trial.get_specific_walk_pos(timestamp)
            playMachine = None
            for name,machine,idx in zip(self.machines.items(),range(len(self.machines.items()))):
                if machine.check_collision(playerXYZ) and playMachine == None:
                    playMachine = idx
            if playMachine != None:
                self.playingMachine = playMachine
                trial.evaluate(playMachine)
                playHistory = self.player.get_play_history
                playHistory.append(playMachine)
                winHistory = self.player.get_win_history
                winHistory.append(iter(next(trial.get_outcome().values())))
                winrateAdjustments = self.behavioralModel.adjust_winrates(playHistory,winHistory,self.get_winrates)
                self.adjust_probabilities(winrateAdjustments)
                min = 0
                for i in range(len(adjustments)):
                    if (adjustments[i] < adjustments[min]):
                        min = i
                trial.set_behavioral_prediction(min,adjustments)
                self.player.add_to_history(trial)
                self.inTrial = False

    def parse_mocap_skeleton_data(self,mocap_data):
        skeleton_data = mocap_data.get_skeleton_data()
        skeleton_list = skeleton_data.get_skeleton_list()
        skeleton = None
        rigid_body_list = None
        #print(skeleton_data.get_skeleton_count())
        #print(skeleton_list)
        if skeleton_data.get_skeleton_count() == 1:
            skeleton = skeleton_list[0]
            rigid_body_list = []
            for rigid_body in skeleton.get_rigid_body_list():
                #print(rigid_body)
                if rigid_body.is_valid():
                    #print('valid')
                    important_data = {}
                    important_data.update({rigid_body.get_id():[rigid_body.get_position(),rigid_body.get_rotation()]})
                    rigid_body_list.append(important_data)
                #else:
                    #print(rigid_body.get_as_string())
        else:
            print(f"expected 1 rigid body, got {skeleton_data.get_skeleton_count()}")
                
        return rigid_body_list
    
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
        corrected_angle = (-angle) % (2 * math.pi)
        return corrected_angle
    
    def parse_mocap_data(self,mocap_data):
        rigid_data = mocap_data.get_rigid_data()
        rigid_list = rigid_data.get_rigid_list()
        rigid_body_list = None
        #print(skeleton_data.get_skeleton_count())
        #print(skeleton_list)
        if rigid_data.get_rigid_body_count() > 0:
            #headband = {rigid_list[0].get_id():[rigid_list[0].get_position(),rigid_list[0].get_rotation()]}
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
    
    def old_process_streamed_data(self,streamed_data,previous_data,timestamp): # uses MATLAB, kinda large processing overhead...
        streamed_data.update({'time':timestamp})
        #print(streamed_data)
        matlab_array = self.eng.struct(self.dict_to_struct(self.eng,streamed_data))
        if previous_data == None:
            previous_data = self.eng.struct()
        else:
            #print(f"previous_data = {previous_data}")
            previous_data = self.eng.struct(previous_data)
        processed_data = self.eng.process_frame(matlab_array,previous_data,nargout=1)
        processed_dict = dict(processed_data)
        return processed_dict
    

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
            vel = compute_velocity(pos, prev_pos, dt)
            streamed_data.update({
                'velocity_x': vel[0],
                'velocity_z': vel[2]
            })

            if 'velocity_x' in previous_data and 'velocity_z' in previous_data:
                prev_vel = [previous_data['velocity_x'], 0, previous_data['velocity_z']]
                acc = compute_acceleration(vel, prev_vel, dt)
                streamed_data.update({
                    'acceleration_x': acc[0],
                    'acceleration_z': acc[2]
                })

            if 'theta' in previous_data:
                theta_dot = compute_theta_dot(theta, previous_data['theta'], dt)
                streamed_data.update({'theta_dot': theta_dot})

                if 'theta_dot' in previous_data:
                    theta_dot_dot = (theta_dot - previous_data['theta_dot']) / dt
                    streamed_data.update({'theta_dot_dot': theta_dot_dot})

        # θ_1 through θ_4: angle to each machine
        angle_dict = compute_theta_to_machines(pos, theta)
        for i, name in enumerate(["machine1", "machine2", "machine3", "machine4"], 1):
            streamed_data.update({f'theta_{i}': angle_dict.get(name, 0)})

        return streamed_data
    
    def dict_to_struct(self,eng, data):
        """
        Safely create a MATLAB struct from a Python dict.
        """
        fields = []
        values = []

        for k, v in data.items():
            fields.append(k)
            if isinstance(v, (int, float)):
                values.append(matlab.double([v]))
            else:
                values.append(v)

        # Now dynamically call struct('field1', value1, 'field2', value2, ...)
        args = []
        for f, v in zip(fields, values):
            args.append(f)
            args.append(v)

        matlab_struct = eng.struct(*args)
        return matlab_struct

class MocapModel:
    def __init__(self):
        self.n_machines = 4
        self.max_adjustment = 0.1
        self.house_edge = 2

    def adjust_winrates(self,foyer,play_history,win_history,current_winrates):
        raise NotImplementedError("MocapModel is abstract.")
    
class IntegralLineOfSight(MocapModel):
    def __init__(self):
        super().__init__()

    def adjust_winrates(self,foyer,play_history,win_history,current_winrates):
        confidence = [0,0,0,0]
        timeOffset = int(next(iter(foyer.keys()))) # each timestep is labeled with its timestamp, so offset is first sample

        for timestamp,sample in iter(foyer.items()):
            offsetTime = int(timestamp)-timeOffset
            confidence[0] += offsetTime * sample['theta_1']
            confidence[1] += offsetTime * sample['theta_2']
            confidence[2] += offsetTime * sample['theta_3']
            confidence[3] += offsetTime * sample['theta_4']
        # pairs the confidence with machine index and sorts them. this lets us operate on their relative rankings
        confidence_sorted, indices_sorted = zip(*sorted(zip(confidence, [0,1,2,3]))) 
        confidence_sorted = list(confidence_sorted)
        indices_sorted = list(indices_sorted)
        adjustments = [0,0,0,0]
        orderedSchema = [-self.max_adjustment, -0.5*self.max_adjustment, 0.5*self.max_adjustment, self.max_adjustment]
        for i in range(len(indices_sorted)):
            adjustments[indices_sorted[i]] = orderedSchema[i]
        return adjustments

    def old_adjust_winrates(self,foyer,play_history,win_history,current_winrates):
        confidence = [0,0,0,0]
        timeOffset = int(next(iter(foyer.keys()))) # each timestep is labeled with its timestamp, so offset is first sample

        for timestamp,sample in iter(foyer.items()):
            offsetTime = int(timestamp)-timeOffset
            confidence[0] += offsetTime * sample['theta_1']
            confidence[1] += offsetTime * sample['theta_2']
            confidence[2] += offsetTime * sample['theta_3']
            confidence[3] += offsetTime * sample['theta_4']
        confidence = confidence / np.sum(confidence) # normalize
        confidence = confidence - np.mean(confidence) # 0 center
        max_adj = []
        min_adj = []
        proposedAdjustments = confidence * self.max_adjustment
        for winRate in current_winrates:
            max_adj.append(np.minimum(self.max_adjustment, 1 - winRate))
            min_adj.append(np.maximum(-self.max_adjustment, -winRate))
        clippedAdjustments = np.clip(proposedAdjustments, min_adj, max_adj) # ensure each adjustment will respect the winrate bounds of 0,1 and the max adjustment
        # clipped could still be non-zero summed...
        error = np.sum(clippedAdjustments)
        idealCorrection = np.ones(clippedAdjustments.size) * -error / self.n_machines
        newClipped = clippedAdjustments
        while (error > 1e-8):
            for i in range(clippedAdjustments.size):
                if (min_adj[i] < clippedAdjustments[i]+idealCorrection[i] < max_adj[i]):
                    newClipped[i] += idealCorrection[i]
                else:
                    correctedCorrection = max_adj[i] - clippedAdjustments[i]+idealCorrection[i] if (max_adj[i] - clippedAdjustments[i]+idealCorrection[i] < 0) else min_adj[i]-clippedAdjustments[i]+idealCorrection[i]
                    if (i != self.n_machines-1): 
                        newClipped[i] += idealCorrection[i] + correctedCorrection
                        for j in range(clippedAdjustments.size - (i+1)):
                            idealCorrection[i+j] += -correctedCorrection/(clippedAdjustments.size - (i+1))
                    else: 
                        idealCorrection[i] += correctedCorrection
                        for j in range(clippedAdjustments.size):
                            if (j != i): idealCorrection[j] += -correctedCorrection/(clippedAdjustments.size - 2)
                        newClipped = clippedAdjustments
                        i = clippedAdjustments.size
                        continue
            error = np.sum(newClipped)
        
        return newClipped
    
class BehavioralModel:
    def __init__(self):
        self.n_machines = 4
        self.max_adjustment = 0.1
        self.house_edge = 2

    def adjust_winrates(self, play_history, win_history, current_winrates):
        raise NotImplementedError("BehavioralModel is abstract.")
    
class WindowedControl(BehavioralModel):
    def __init__(self):
        super().__init__()
        # Dummy hidden state for simulation purposes
        self.hidden_state = np.zeros(self.n_machines)

    def adjust_winrates(self,play_history, win_history, current_winrates):
        # Use recent performance per machine to drive adjustment.
        adjustments = np.zeros(self.n_machines)
        for i in range(self.n_machines):
            indices = [j for j, p in enumerate(play_history) if p == i+1]
            if indices:
                recent = indices[-5:] if len(indices) >= 5 else indices
                win_rate = np.mean([1 if win_history[j] else 0 for j in recent])
            else:
                win_rate = 0.5
            # “Optimism” rule: if win_rate is high, lower win chance; if low, boost it.
            if win_rate > 0.5:
                adjustments[i] = -self.max_adjustment * 2*(win_rate - 0.5)
            else:
                adjustments[i] = self.max_adjustment * (0.5 - win_rate)

        adjustments -= np.mean(adjustments)
        new_winrates = current_winrates + adjustments
        new_winrates = np.clip(new_winrates, 0, 1)
        new_winrates = new_winrates / np.sum(new_winrates) * self.house_edge
        adjustments = np.clip(new_winrates-current_winrates,-self.max_adjustment,self.max_adjustment)
        # ^ questionable clamping...
        return adjustments
    
class BayesianHouse(BehavioralModel):
    def __init__(self):
        super().__init__()
        # Initialize Beta distribution parameters for each machine
        self.alphas = np.ones(self.n_machines)
        self.betas = np.ones(self.n_machines)

    def adjust_winrates(self, play_history, win_history, current_winrates):
        # Update Beta distribution parameters based on play and win history
        for i in range(self.n_machines):
            plays = [j for j, p in enumerate(play_history) if p == i + 1]
            wins = sum([1 for j in plays if win_history[j]])
            losses = len(plays) - wins
            self.alphas[i] = wins + 1
            self.betas[i] = losses + 1

        # Calculate the expected win probability for each machine
        expected_win_probs = [beta.mean(a, b) for a, b in zip(self.alphas, self.betas)]

        confidence_sorted, indices_sorted = zip(*sorted(zip(expected_win_probs, [0,1,2,3]))) 
        confidence_sorted = list(confidence_sorted)
        indices_sorted = list(indices_sorted)
        adjustments = [0,0,0,0]
        orderedSchema = [-self.max_adjustment, -0.5*self.max_adjustment, 0.5*self.max_adjustment, self.max_adjustment]
        for i in range(len(indices_sorted)):
            adjustments[indices_sorted[i]] = orderedSchema[i]
        return adjustments

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