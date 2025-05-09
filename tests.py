import unittest
import json
import os
from MoCapData import generate_mocap_data, generate_position_srand
from FeatureExtractor import Machine, Player, Trial, Game
from MocapModels import MocapModel, IntegralLineOfSight
from BehavioralModels import BehavioralModel, BayesianHouse
import glob

class TestMachine(unittest.TestCase):
    def setUp(self):
        self.machine = Machine(
            name="TestMachine",
            winChance=0.4,
            bottomLeft=(20, 15),
            bottomRight=(20, 5),
            upperLeft=(10, 15),
            upperRight=(10, 5)
        )

    def test_collision_detection(self):
        # Test inside bounds
       # machine1 = Machine("M1",0.5,[-0.1,2.8],[-0.05,-0.16],[-1.8,2.9],[-1.8,0.1])

        self.assertTrue(self.machine.check_collision((15, 0, 10)))
        # Test outside bounds
        self.assertFalse(self.machine.check_collision((5, 0, 3)))
        # Test edge cases
        self.assertTrue(self.machine.check_collision((10, 0, 5)))
        self.assertTrue(self.machine.check_collision((20, 0, 15)))

    def test_win_chance_updates(self):
        initial_chance = self.machine.get_win_chance()
        self.machine.set_win_chance(0.1)
        self.assertEqual(self.machine.get_win_chance(), 0.1)

class TestPlayer(unittest.TestCase):
    def setUp(self):
        self.player = Player()
        self.test_file = "test_history.json"

    def tearDown(self):
        if os.path.exists(self.test_file):
            os.remove(self.test_file)

    def test_history_storage(self):
        machines = {"Machine1": Machine("Machine1", 0.3, (0,0), (0,0), (0,0), (0,0))}
        trial = Trial(machines)
        #self.player.add_to_history(trial)
        #self.assertEqual(len(self.player.history), 1)

    def test_json_output(self):
        machines = {"Machine1": Machine("Machine1", 0.3, (0,0), (0,0), (0,0), (0,0))}
        trial = Trial(machines)
        #self.player.add_to_history(trial)
        #self.player.write_history_to_file(self.test_file)
        
        #with open(self.test_file) as f:
            #data = json.load(f)
            #self.assertEqual(len(data), 1)

class TestTrial(unittest.TestCase):
    def setUp(self):
        self.machines = {
            "MachineA": Machine("MachineA", 0.4, (0,0), (0,0), (0,0), (0,0)),
            "MachineB": Machine("MachineB", 0.3, (0,0), (0,0), (0,0), (0,0))
        }
        self.trial = Trial(self.machines)

    def test_foyer_updates(self):
        test_data = {"rb1": [(1,2,3), (0,0,0,1)]}
        self.trial.update_foyer(123, test_data)
        self.assertIn(123, self.trial.foyer)

    def test_prediction_mechanism(self):
        # Initial setup
        initial_total = sum(self.trial.prePredictionWinRates.values())
        adjustments = [0.1,-0.1]

        # Apply prediction and adjustments
        self.trial.set_mocap_prediction("MachineA", adjustments)

        # Verify exact sum preservation
        post_total = sum(self.trial.postMocapPredictionWinRates.values())
        self.assertEqual(post_total, initial_total, 
                        "Total probabilities should remain exactly equal to house advantage")
        
        # Verify individual machine changes using exact comparisons
        self.assertAlmostEqual(self.trial.postMocapPredictionWinRates["MachineA"], 
                        self.trial.prePredictionWinRates["MachineA"] + adjustments[0])
        self.assertAlmostEqual(self.trial.postMocapPredictionWinRates["MachineB"], 
                        self.trial.prePredictionWinRates["MachineB"] + adjustments[1])

class TestGame(unittest.TestCase):
    def setUp(self):
        self.game = Game(2,0.1,BayesianHouse(),IntegralLineOfSight())
        m1 = Machine("m1",0.5,[0.914461,-0.416378],[0.897491,-0.716809],[0.615168,-0.399924],[0.600168,-0.699141])
        m2 = Machine("m2",0.5,[0.930728,-0.117966],[0.914461,-0.416378],[0.634181,-0.100121],[0.615168,-0.399924])
        m3 = Machine("m3",0.5,[0.945728,0.181054],[0.930728,-0.117966],[0.652421,0.196054],[0.634181,-0.100121])
        m4 = Machine("m4",0.5,[0.975101,0.483173],[0.945728,0.181054],[0.669682,0.497284],[0.652421,0.196054])
        self.game.add_machine(m1)
        self.game.add_machine(m2)
        self.game.add_machine(m3)
        self.game.add_machine(m4)
        self.game.set_foyer_line([[-0.869957,-1.361096],[-0.796456,1.429263]])
   
    def test_machine_management(self):
        self.assertEqual(len(self.game.machines), 4)

    def test_probability_adjustment(self):
        initial_houseAdvantage = 0
        for name,machine in self.game.machines.items():
            initial_houseAdvantage += machine.get_win_chance()
        self.game.adjust_probabilities([0,0.1,-0.1,0])
        actual_houseAdvantage = 0
        for name,machine in self.game.machines.items():
            actual_houseAdvantage += machine.get_win_chance()
        self.assertEqual(actual_houseAdvantage,initial_houseAdvantage, "Total probabilities should remain exactly equal to house advantage")

    def test_frame_processing(self):
        # Generate test mocap data
        mocap_data = generate_mocap_data(frame_num=1)

        # Simulate frame data dictionary
        data_dict = {
            "frame_number": 1,
            "other_info": "test"
        }
        
        self.game.start_next_trial()
        self.game.receive_new_frame(data_dict, mocap_data)

        # Verify data parsing
        current_trial = self.game.trials[-1]
        self.assertGreater(len(current_trial.foyer), 0)

    def test_data_parsing(self):
        self.game.start_next_trial()
        # before foyer
        mocap_data = generate_mocap_data(frame_num=0,x=-1,z=0)
        self.game.receive_new_frame({"frame_number":0},mocap_data)
        # past foyer
        mocap_data = generate_mocap_data(frame_num=1,x=-0.5,z=0)
        self.game.receive_new_frame({"frame_number":1},mocap_data)
        # in machine 3 (index 2)
        mocap_data = generate_mocap_data(frame_num=2,x=0.75,z=0)
        self.game.receive_new_frame({"frame_number":2},mocap_data)
        # assert a machine play was detected
        self.assertIsNotNone(self.game.playMachine)

    def test_real_data_parsing(self):
        # Load the data from the file
        json_files = glob.glob(os.path.join(".\Owen", "*.json"))
        for file_path in json_files:
            with open(file_path, 'r') as file:
                data = json.load(file)
            
            if isinstance(data, list):
                for trial in data:
                    self.game.start_next_trial()
                    official_outcome = trial['outcome']
                    keys = official_outcome.keys()
                    values = official_outcome.values()
                    for (name,machine),idx in zip(self.game.machines.items(),range(len(self.game.machines))):
                        if name == next(iter(keys)):
                            outcome = idx
                    foyer = trial['foyer']
                    walk = trial['walk']
                    for timestamp,sample in iter(foyer.items()):
                        time = int(timestamp)
                        x = sample['position_x']
                        z = sample['position_z']
                        w = sample['rotation_w']
                        rx = sample['rotation_x']
                        ry = sample['rotation_y']
                        rz = sample['rotation_z']
                        mocap_data = generate_mocap_data(frame_num=time,x=x,z=z,w=w,rx=rx,ry=ry,rz=rz)
                        self.game.receive_new_frame({"frame_number":time},mocap_data)
                    self.assertIsNotNone(self.game.trials[-1].get_walk())
                    for timestamp,sample in iter(walk.items()):
                        time = int(timestamp)+1
                        x = sample['position_x']
                        z = sample['position_z']
                        w = sample['rotation_w']
                        rx = sample['rotation_x']
                        ry = sample['rotation_y']
                        rz = sample['rotation_z']
                        mocap_data = generate_mocap_data(frame_num=time,x=x,z=z,w=w,rx=rx,ry=ry,rz=rz)
                        self.game.receive_new_frame({"frame_number":time},mocap_data)
                self.assertEqual(self.game.playMachine,outcome)
                #self.assertFalse(self.game.inTrial)
                #print(f"recorded: {official_outcome}, observed: {self.game.playMachine}")
                #print(f"pre_winrates = {self.game.trials[-1].get_pre_win_rates()}")
                #print(f"post_mocap_winrates = {self.game.trials[-1].get_post_mocap_win_rates()}")
                #print(f"post_behavioral_winrates = {self.game.trials[-1].get_post_behavioral_win_rates()}")



    def test_collision_detection_with_real_data(self):
        machine = Machine("Test", 0.4, (10,5), (10,15), (20,5), (20,15))
        # Generate position from mocap data generator
        test_position = generate_position_srand(0, 0)
        # Convert to proper format (x,z,y?)
        # Note: Coordinate system conversion might be needed here
        collision = machine.check_collision((test_position[0], 0, test_position[2]))
        self.assertIsInstance(collision, bool)

if __name__ == '__main__':
    unittest.main()