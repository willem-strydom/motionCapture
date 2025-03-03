import MoCapData
import random
import json

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
        roll = random.random
        self.outcome = {choice:(roll < self.postPredictionWinRates.get(choice))}
        return self.outcome

    def update_foyer(self,timestamp,rigidBodies):
        self.foyer.update({timestamp:rigidBodies})

    def update_walk(self,timestamp,rigidBodies):
        self.walk.update({timestamp:rigidBodies})

    def check_out_of_foyer(self,timestamp,foyer_line):
        # foyer_line should be [(x,z),(x,z)]
        processed_data = self.foyer.get(timestamp)
        # assuming parent rigid body is first
        current_position = next(iter(processed_data[0].values()))[0]
        # assuming x decreases as you go left and z decreases as you go "up"
        over_right = current_position[0] <= foyer_line[0][0] and current_position[2] >= foyer_line[0][1]
        over_left = current_position[0] <= foyer_line[1][0] and current_position[2] <= foyer_line[1][1]
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
    
    def get_specific_walk_pos(self,timestamp):
        return next(iter(self.walk.get(timestamp)[0].values()))[0]

class Game:
    
    def __init__(self,B,M):
        self.machines = {}
        self.trials = []
        self.player = Player()
        self.maxProbabilityAdjustment = M
        self.houseAdvantage = B
        self.foyer_line = None
        self.inTrial = False

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
        naiveDelta = self.maxProbabilityAdjustment / (len(self.machines)-1) # dont play w 1 machine, will divide by 0
        adjustments = {}
        sum = 0
        for name,machine in self.machines.items():
            if not name == predicted_choice:
                adjustments.update({name:naiveDelta})
                sum += naiveDelta

        adjustments.update({predicted_choice: -sum})
        return adjustments

    def add_machine(self,machine):
        print(machine.get_name())
        print(machine.get_win_chance())
        self.machines[machine.get_name()] = machine

    def set_foyer_line(self,foyer_line):
        # [Left (x,z), Right (x,z)]
        self.foyer_line = foyer_line

    def start_next_trial(self):
        self.trials.append(Trial(self.machines))
        self.inTrial = True

    def save_current_trial(self):
        self.player.add_to_history(self.trials[-1])
        self.player.write_history_to_file("trial.json")

    def receive_new_frame(self, data_dict, mocap_data):
        if not self.inTrial:
            return
        trial = self.trials[-1] # most recent trial
        timestamp = data_dict["frame_number"]
        processed_data = self.parse_mocap_data(mocap_data)
        if not trial.get_walk(): # we are still in foyer
            trial.update_foyer(timestamp,processed_data)
            if trial.check_out_of_foyer(timestamp,self.foyer_line): # we have JUST left foyer
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
                trial.evaluate(playMachine)
                self.player.add_to_history(trial)
                self.inTrial = False

    def parse_mocap_data(self,mocap_data):
        skeleton_data = mocap_data.get_skeleton_data()
        skeleton_list = skeleton_data.get_skeleton_list()
        skeleton = None
        rigid_body_list = None
        if skeleton_data.get_skeleton_count() == 1:
            skeleton = skeleton_list[0]
            rigid_body_list = []
            for rigid_body in skeleton.get_rigid_body_list():
                if rigid_body.is_valid():
                    important_data = {}
                    important_data.update({rigid_body.get_id():[rigid_body.get_position(),rigid_body.get_rotation()]})
                    rigid_body_list.append(important_data)
        return rigid_body_list
    
class LSTM:

    def predict(foyer_data,machines):
        return next(iter(machines.keys())) # placeholder, returns first machine that was added