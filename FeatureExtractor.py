import MoCapData
import random
import json
import matlab.engine
import os
import math

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

    def add_to_history(self,trial):
        self.history.append(trial)

    def write_history_to_file(self, filename="player_history.json"):
        history_data = []
        for trial in self.history:
            trial_data = {
                "prediction": trial.get_prediction(),
                "outcome": trial.get_outcome(),
                "pre_win_rates": trial.get_pre_win_rates(),
                "post_win_rates": trial.get_post_win_rates(),
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
            "prediction": trial.get_prediction(),
            "outcome": trial.get_outcome(),
            "pre_win_rates": trial.get_pre_win_rates(),
            "post_win_rates": trial.get_post_win_rates(),
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
        self.prediction = None
        for name,machine in machines.items():
            self.prePredictionWinRates[name] = machine.get_win_chance()
        self.postPredictionWinRates = {}
        
    def evaluate(self,choice):
        rollValue = random.random()
        self.outcome = {choice:(rollValue < self.postPredictionWinRates.get(choice))}
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
            
    def set_prediction(self,prediction,postPredictionWinRates):
        self.prediction = prediction
        pre_sum = 0
        post_sum = 0
        for name,winRate in self.prePredictionWinRates.items():
            pre_sum += winRate
            new_rate = winRate + postPredictionWinRates[name]
            self.postPredictionWinRates[name] = new_rate
            post_sum += new_rate
        if pre_sum != post_sum:
            self.postPredictionWinRates[prediction] += pre_sum - post_sum
    def get_prediction(self):
        return self.prediction
    
    def get_outcome(self):
        return self.outcome
    
    def get_pre_win_rates(self):
        return self.prePredictionWinRates
    
    def get_post_win_rates(self):
        return self.postPredictionWinRates
    
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
        return next(iter(self.foyer.keys()))

class Game:

    def __init__(self,B,M):
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
        self.eng = matlab.engine.start_matlab()     # initialize matlab connection
        self.eng.addpath(os.getcwd(),nargout=1)       # add current working director to matlab path to access local functions
        #print(self.eng.which('process_frame', nargout=1))
        #print(os.getcwd())

    def shut_down(self):
        self.eng.rmpath(os.getcwd(),nargout=0)
        self.eng.quit()

    def adjust_probabilities(self,prediction,delta_list):
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
            

    def determine_adjustment(self,predicted_choice):
        if self.for_training: # give the players a changing enviorment to react to, without any house model assumptions

            while True:
                a = random.uniform(-self.maxProbabilityAdjustment, self.maxProbabilityAdjustment)
                b = random.uniform(-self.maxProbabilityAdjustment, self.maxProbabilityAdjustment)
                c = random.uniform(-self.maxProbabilityAdjustment, self.maxProbabilityAdjustment)
                d = -(a + b + c)
                if -self.maxProbabilityAdjustment <= d <= self.maxProbabilityAdjustment:
                    training_adjustments = [a, b, c, d]
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

    def receive_new_frame(self, data_dict, mocap_data):
        if not self.inTrial:
            return
        trial = self.trials[-1] # most recent trial
        timestamp = data_dict["frame_number"]
        #print(f"reciefed frame {timestamp}")
        #print(mocap_data)
        streamed_data = self.parse_mocap_data(mocap_data)
        processed_data = self.process_streamed_data(timestamp=timestamp,streamed_data=streamed_data,previous_data=trial.get_prev_foyer_frame())
        #print(processed_data)
        if not trial.get_walk(): # we are still in foyer
            trial.update_foyer(timestamp,processed_data)
            if trial.check_out_of_foyer(timestamp,self.foyer_line): # we have JUST left foyer
                self.behindFoyer = False
                # placeholder pipeline for predicting the machine to be played
                prediction = LSTM.predict(trial.get_foyer(),self.machines)
                # placeholder pipeline for determining how to adjust the machine probabilities based on prediction
                adjustments = self.determine_adjustment(prediction)
                trial.set_prediction(prediction,adjustments)
                self.adjust_probabilities(prediction,adjustments)
                # update walk history so we can later identify the "crossing point" by matching timestamps between arrays
                trial.update_walk(timestamp,processed_data)
        else:
            trial.update_walk(timestamp,processed_data)
            playerXYZ = trial.get_specific_walk_pos(timestamp)
            playMachine = None
            for name,machine in self.machines.items():
                if machine.check_collision(playerXYZ) and playMachine == None:
                    playMachine = name
            if playMachine != None:
                self.playingMachine = playMachine
                trial.evaluate(playMachine)
                self.player.add_to_history(trial)
                #self.inTrial = False

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
    
    def process_streamed_data(self,streamed_data,previous_data,timestamp):
        theta = self.get_view_angle(streamed_data["rotation_w"],streamed_data["rotation_x"],streamed_data["rotation_y"],streamed_data["rotation_z"])
        #streamed_data.update({'time':timestamp})
        streamed_data.update({'theta':theta})
        return streamed_data
    
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
        corrected_angle = (-angle + 1.8) % (2 * math.pi)
        return corrected_angle
    
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

class LSTM:

    def predict(foyer_data,machines):
        return next(iter(machines.keys())) # placeholder, returns first machine that was added
    

if __name__ == "__main__":
    game = Game(2,0.1)