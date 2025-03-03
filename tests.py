import unittest
from unittest.mock import patch, MagicMock, mock_open
from FeatureExtractor import Machine, Player, Trial, Game, LSTM
from MoCapData import (
    generate_mocap_data,
    generate_rigid_body,
    generate_skeleton_data,
    generate_prefix_data,
    generate_suffix_data,
    RigidBody
)

class TestGameIntegration(unittest.TestCase):
    def setUp(self):
        self.game = Game()
        self.machine = Machine(
            "TestMachine", 0.5,
            bottomLeft=(0,0), bottomRight=(0,10),
            upperLeft=(10,0), upperRight=(10,10)
        )
        self.game.add_machine(self.machine)
        self.game.set_foyer_line([(0,0), (0,10)])

    def create_custom_mocap(self, frame_num=0, position=(0,0,0)):
        """Create mocap data with custom position using official generators"""
        mocap_data = generate_mocap_data(frame_num)
        
        # Create custom rigid body with specified position
        custom_body = generate_rigid_body(
            body_num=999,  # Unique ID for test body
            frame_num=frame_num
        )
        custom_body.pos = position
        custom_body.tracking_valid = True

        # Create skeleton data with custom body
        skeleton = next(iter(mocap_data.skeleton_data.skeleton_list))
        skeleton.rigid_body_list = [custom_body]
        
        return mocap_data

    @patch.object(LSTM, 'predict')
    def test_full_play_cycle(self, mock_predict):
        mock_predict.return_value = "TestMachine"
        
        # Start trial
        self.game.start_next_trial()
        
        # Frame 1: In foyer
        mocap_foyer = self.create_custom_mocap(
            frame_num=1,
            position=(0, 5, 0)  # On foyer line
        )
        self.game.receive_new_frame({"frame_number": 1}, mocap_foyer)
        
        # Verify prediction made
        trial = self.game.trials[-1]
        self.assertEqual(trial.prediction, "TestMachine")
        
        # Frame 2: Enter machine area
        mocap_machine = self.create_custom_mocap(
            frame_num=2,
            position=(5, 5, 0)  # Inside machine
        )
        self.game.receive_new_frame({"frame_number": 2}, mocap_machine)
        
        # Verify outcome recorded
        self.assertIsNotNone(trial.outcome)
        self.assertIn("TestMachine", trial.outcome)

    def test_mocap_data_parsing(self):
        # Generate official test data
        mocap_data = generate_mocap_data(0)
        processed = self.game.parse_mocap_data(mocap_data)
        
        # Verify structure parsing
        self.assertTrue(len(processed) > 0)
        self.assertIsInstance(processed[0], dict)
        
        # Verify rigid body extraction
        skeleton_data = mocap_data.get_skeleton_data()
        expected_body = skeleton_data.get_skeleton_list()[0].get_rigid_body_list()[0]
        body_id = expected_body.get_id()
        self.assertIn(body_id, processed[0])

    def test_collision_detection_with_generated_data(self):
        # Generate rigid body at edge case position
        edge_body = generate_rigid_body(
            body_num=0,
            frame_num=0
        )
        edge_body.pos = [10, 10, 0]  # Machine upperRight
        
        # Test collision detection
        self.assertTrue(self.machine.check_collision(edge_body.pos[:2]))

    def test_foyer_detection(self):
        # Generate frame with prefix data
        mocap_data = generate_mocap_data(1)
        prefix_data = generate_prefix_data(1)
        mocap_data.set_prefix_data(prefix_data)
        
        # Add custom position
        skeleton = mocap_data.get_skeleton_data().get_skeleton_list()[0]
        skeleton.rigid_body_list[0].pos = [0, 5, 0]  # On foyer line
        
        # Process frame
        self.game.start_next_trial()
        self.game.receive_new_frame({"frame_number": 1}, mocap_data)
        
        # Verify foyer detection
        trial = self.game.trials[-1]
        self.assertGreater(len(trial.foyer), 0)

class TestMocapDataStructures(unittest.TestCase):
    def test_generated_data_consistency(self):
        # Test official generator functions
        frame_num = 42
        mocap_data = generate_mocap_data(frame_num)
        
        # Verify frame number in prefix
        self.assertEqual(mocap_data.prefix_data.frame_number, frame_num)
        
        # Verify skeleton data
        self.assertGreaterEqual(
            mocap_data.skeleton_data.get_skeleton_count(),
            3
        )
        
        # Verify rigid body data
        skeleton = mocap_data.skeleton_data.get_skeleton_list()[0]
        self.assertGreaterEqual(
            len(skeleton.get_rigid_body_list()),
            2
        )

    def test_rigid_body_generation(self):
        body = generate_rigid_body(0, 0)
        self.assertIsInstance(body, RigidBody)
        self.assertTrue(body.tracking_valid)
        
        # Verify position generation
        pos = body.get_position()
        self.assertEqual(len(pos), 3)
        self.assertTrue(all(isinstance(v, float) for v in pos))

class TestTrialProcessing(unittest.TestCase):
    def setUp(self):
        self.machines = {
            "M1": Machine("M1", 0.4, (0,0), (0,0), (0,0), (0,0)),
            "M2": Machine("M2", 0.6, (0,0), (0,0), (0,0), (0,0))
        }
        
    def test_trial_initialization(self):
        mocap_data = generate_mocap_data(0)
        trial = Trial(self.machines)
        
        # Verify win rate capture
        self.assertEqual(trial.prePredictionWinRates["M1"], 0.4)
        self.assertEqual(trial.prePredictionWinRates["M2"], 0.6)

    @patch("random.random")
    def test_outcome_evaluation(self, mock_random):
        mock_random.return_value = 0.5  # Mid value
        trial = Trial(self.machines)
        trial.postPredictionWinRates = {"M1": 0.5}
        
        outcome = trial.evaluate("M1")
        self.assertTrue(outcome["M1"])

if __name__ == "__main__":
    unittest.main(failfast=True)