import numpy as np
import random
import numpy as np
from sklearn.cluster import KMeans
from statsmodels.tsa.arima.model import ARIMA
import tensorflow as tf
from tensorflow.keras import layers

################################
# Part 3: House Simulations    #
################################

class BaseHouse:
    def __init__(self, parameters):
        self.parameters = parameters
        self.n_machines = 4
        self.max_adjustment = parameters.get('max_adjustment', 0.1)

    def adjust_winrates(self, play_history, win_history, current_winrates):
        raise NotImplementedError("BaseHouse is abstract.")


class HeuristicAdjustmentHouse(BaseHouse):
    def __init__(self, parameters):
        super().__init__(parameters)
        # Dummy hidden state for simulation purposes
        self.hidden_state = np.zeros(self.n_machines)

    def adjust_winrates(self, play_history, win_history, current_winrates):
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
                adjustments[i] = -self.max_adjustment * (win_rate - 0.5)
            else:
                adjustments[i] = self.max_adjustment * (0.5 - win_rate)
        return np.clip(adjustments, -self.max_adjustment, self.max_adjustment)


class RegressionHouse(BaseHouse):
    def __init__(self, parameters):
        super().__init__(parameters)

    def adjust_winrates(self, play_history, win_history, current_winrates):
        # Simulate a regression: compare observed win rates with a target (0.5) and adjust.
        adjustments = np.zeros(self.n_machines)
        for i in range(self.n_machines):
            indices = [j for j, p in enumerate(play_history) if p == i+1]
            if indices:
                win_rate = np.mean([1 if win_history[j] else 0 for j in indices])
            else:
                win_rate = 0.5
            error = win_rate - 0.5
            adjustments[i] = -self.max_adjustment * error
        return np.clip(adjustments, -self.max_adjustment, self.max_adjustment)


class VARHouse(BaseHouse):
    def __init__(self, parameters):
        super().__init__(parameters)
        self.adjustment_history = []

    def adjust_winrates(self, play_history, win_history, current_winrates):
        # For VAR simulation, use the average of past adjustments to forecast a counter adjustment.
        if self.adjustment_history:
            past = np.array(self.adjustment_history)
            avg_adjustment = np.mean(past, axis=0)
            adjustments = -avg_adjustment
        else:
            adjustments = np.zeros(self.n_machines)
        adjustments = np.clip(adjustments, -self.max_adjustment, self.max_adjustment)
        self.adjustment_history.append(adjustments)
        return adjustments


class StatespaceHouse(BaseHouse):
    def __init__(self, parameters):
        super().__init__(parameters)
        self.winrate_history = []  # store past winrate vectors

    def adjust_winrates(self, play_history, win_history, current_winrates):
        # Simple moving average based adjustment: drive current winrates toward the historical mean.
        if self.winrate_history:
            moving_avg = np.mean(self.winrate_history, axis=0)
            adjustments = (moving_avg - current_winrates) * self.max_adjustment
        else:
            adjustments = np.zeros(self.n_machines)
        adjustments = np.clip(adjustments, -self.max_adjustment, self.max_adjustment)
        self.winrate_history.append(current_winrates.copy())
        return adjustments



### New House Models ###########################################################

import numpy as np
import warnings
from statsmodels.tsa.arima.model import ARIMA

class ARIMAHouse:
    def __init__(self, params):
        # Maximum adjustment per machine
        self.max_adjustment = params.get('max_adjustment', 0.1)
        self.n_machines = 4
        # Store winrate history per machine as a list for each machine.
        self.winrate_histories = [[] for _ in range(self.n_machines)]
        # Minimum data points required to fit ARIMA.
        self.min_points = 3

    def adjust_winrates(self, play_history, win_history, current_winrates):
        """
        For each machine, append the current winrate to its history.
        If sufficient history exists, fit an ARIMA model (order=(1,0,0)) and forecast
        the next winrate. The adjustment is computed as the difference between the forecasted
        winrate and the current winrate, then clipped to [-max_adjustment, max_adjustment].
        """
        adjustments = np.zeros(self.n_machines)
        for i in range(self.n_machines):
            # Update the winrate history for machine i.
            self.winrate_histories[i].append(current_winrates[i])
            history = self.winrate_histories[i]
            
            # Forecast only if we have enough data points.
            if len(history) >= self.min_points:
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        model = ARIMA(history, order=(1, 0, 0))
                        model_fit = model.fit()
                        forecast = model_fit.forecast(steps=1)
                        predicted = forecast[0]
                except Exception:
                    # On failure, default to no change.
                    predicted = current_winrates[i]
            else:
                # Insufficient data: no forecast, no adjustment.
                predicted = current_winrates[i]
            
            # Ensure the predicted winrate remains within [0, 1].
            predicted = np.clip(predicted, 0, 1)
            # Compute adjustment as the difference, then clip to maximum allowed.
            adjustment = np.clip(predicted - current_winrates[i], -self.max_adjustment, self.max_adjustment)
            adjustments[i] = adjustment

        return adjustments


import numpy as np

class MarkovChainHouse:
    def __init__(self, params):
        # Use the provided max_adjustment; default to 0.1 if not given.
        self.max_adjustment = params.get('max_adjustment', 0.1)
        # Use provided states if available, otherwise default to three states.
        self.states = params.get('states', ['increase', 'stay', 'decrease'])
        # Default transition matrix for a 3-state Markov chain.
        default_transition_matrix = np.array([
            [0.7, 0.2, 0.1],  # from 'increase'
            [0.2, 0.6, 0.2],  # from 'stay'
            [0.1, 0.2, 0.7]   # from 'decrease'
        ])
        self.transition_matrix = params.get('transition_matrix', default_transition_matrix)
        self.n_machines = 4
        # Initialize the current state for each machine as 'stay'
        self.current_states = ['stay'] * self.n_machines
        # Mapping for state names to indices for transition matrix lookup.
        self.state_to_index = {state: idx for idx, state in enumerate(self.states)}

    def adjust_winrates(self, play_history, win_history, current_winrates):
        """
        For each machine, update its state via a Markov transition and return
        an adjustment vector where 'increase' gives +max_adjustment, 'decrease' gives -max_adjustment,
        and 'stay' gives 0.
        """
        adjustments = np.zeros(self.n_machines)
        for i in range(self.n_machines):
            current_state = self.current_states[i]
            current_index = self.state_to_index[current_state]
            # Sample the next state based on the current state's probabilities.
            next_state_index = np.random.choice(len(self.states), p=self.transition_matrix[current_index])
            next_state = self.states[next_state_index]
            self.current_states[i] = next_state
            # Map the state to an adjustment value.
            if next_state == 'increase':
                adjustments[i] = self.max_adjustment
            elif next_state == 'decrease':
                adjustments[i] = -self.max_adjustment
            else:
                adjustments[i] = 0.0
        return adjustments


import numpy as np
from sklearn.linear_model import SGDRegressor

class RLHouse(BaseHouse):
    def __init__(self, parameters):
        super().__init__(parameters)
        self.model = SGDRegressor(learning_rate='constant', eta0=0.01)
        self.model.partial_fit([self._feature_vector([])], [0])  # Initialize model
        self.max_adjustment = parameters.get('max_adjustment', 0.1)
        self.discount_factor = parameters.get('discount_factor', 0.95)

    def _feature_vector(self, play_history):
        # Example feature vector: counts of plays for each machine
        return np.array([play_history.count(i + 1) for i in range(self.n_machines)])

    def adjust_winrates(self, play_history, win_history, current_winrates):
        if not play_history:
            return np.zeros(self.n_machines)

        # Update model with the latest play
        state = self._feature_vector(play_history[:-1])
        next_state = self._feature_vector(play_history)
        reward = win_history[-1] * 2  # Win yields 2 coins
        future_reward = self.model.predict([next_state])[0]
        target = reward + self.discount_factor * future_reward
        self.model.partial_fit([state], [target])

        # Determine adjustments based on predicted rewards
        predicted_rewards = [self.model.predict([self._feature_vector(play_history + [i + 1])])[0] for i in range(self.n_machines)]
        adjustments = -self.max_adjustment * (predicted_rewards - np.mean(predicted_rewards))

        # Ensure adjustments are within the allowed range
        return np.clip(adjustments, -self.max_adjustment, self.max_adjustment)


import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense

class LSTMHouse:
    def __init__(self, config):
        """
        Initialize the LSTMHouse model.
        
        config: dict containing configuration parameters.
            Expected keys:
              - 'max_adjustment': float, maximum adjustment for winrates.
              - 'input_shape': tuple, shape of input data for LSTM (time_steps, features).
                             (Default: (10, 1))
              - 'units': int, number of LSTM units (Default: 50)
              - 'window_size': int, number of past observations for online training (Default: 10)
        """
        self.max_adjustment = config.get('max_adjustment', 0.1)
        self.input_shape = config.get('input_shape', (10, 1))
        self.units = config.get('units', 50)
        self.window_size = config.get('window_size', 10)
        
        # Build the LSTM model with an explicit Input layer.
        self.model = Sequential()
        self.model.add(Input(shape=self.input_shape))
        self.model.add(LSTM(self.units))
        self.model.add(Dense(1))
        self.model.compile(optimizer='adam', loss='mean_squared_error')
        
        # Buffer for online training: stores recent average winrate values.
        self.winrate_buffer = []
        
    def online_train(self, current_avg):
        """
        Append the current average winrate to the buffer and, if the buffer is full,
        update the model using the most recent window of data.
        """
        # Append the current average winrate (wrapped as a list for proper shape).
        self.winrate_buffer.append([current_avg])
        
        # Train only if we have enough data in the buffer.
        if len(self.winrate_buffer) >= self.window_size:
            # Prepare training data: shape (1, window_size, 1)
            window_data = np.array(self.winrate_buffer[-self.window_size:]).reshape(1, self.window_size, 1)
            # Define a simple target based on the deviation from 0.5.
            target = np.tanh(0.5 - current_avg)
            y_train = np.array([[target]])
            # Perform an online update.
            self.model.train_on_batch(window_data, y_train)
    
    def predict_adjustment(self):
        """
        Predict an adjustment value using the latest window of data.
        If insufficient data is available, return 0.
        """
        if len(self.winrate_buffer) < self.window_size:
            return 0.0
        window_data = np.array(self.winrate_buffer[-self.window_size:]).reshape(1, self.window_size, 1)
        prediction = self.model.predict(window_data, verbose=0)[0, 0]
        return prediction
    
    def adjust_winrates(self, play_history, win_history, current_winrates):
        """
        Use the online-trained LSTM model to predict an adjustment for the current winrates.
        The adjustment is computed by scaling the model’s prediction by max_adjustment,
        and is applied uniformly across all machines.
        """
        # Compute the current average winrate.
        current_avg = np.mean(current_winrates)
        # Update the model with the current average winrate.
        self.online_train(current_avg)
        # Get the model prediction.
        pred = self.predict_adjustment()
        # Scale the prediction to obtain an adjustment value.
        adjustment_value = self.max_adjustment * pred
        # Create an adjustment vector matching the shape of current_winrates.
        adjustments = np.full(current_winrates.shape, adjustment_value)
        return adjustments

import numpy as np
import random
import math

class Exp3House:
    def __init__(self, params):
        # Maximum adjustment per machine (e.g., 0.1)
        self.max_adjustment = params.get('max_adjustment', 0.1)
        self.n_machines = 4
        # Define discrete adjustment actions for each machine:
        # Decrease by max_adjustment, no change, or increase by max_adjustment.
        self.adjustments = np.array([-self.max_adjustment, 0.0, self.max_adjustment])
        self.n_actions = len(self.adjustments)
        # Exploration parameter for Exp3 (gamma). Default 0.1.
        self.gamma = params.get('gamma', 0.1)
        # Initialize weights for each machine (each with three actions).
        self.weights = np.ones((self.n_machines, self.n_actions))

    def adjust_winrates(self, play_history, win_history, current_winrates):
        """
        For each machine, compute a probability distribution over adjustment actions
        using the Exp3 rule and sample an action. Then, for the machine that was played in the last turn,
        update its weights using a reward signal:
          - Reward = 1 if the player lost (good for the house)
          - Reward = 0 if the player won (bad for the house)
        Returns an adjustment vector (length = n_machines) with values in {-max_adjustment, 0, +max_adjustment}.
        """
        adjustments_out = np.zeros(self.n_machines)
        chosen_actions = np.zeros(self.n_machines, dtype=int)
        probabilities = np.zeros((self.n_machines, self.n_actions))
        
        # For each machine, compute the probabilities and sample an action.
        for i in range(self.n_machines):
            total_weight = np.sum(self.weights[i])
            probabilities[i] = (1 - self.gamma) * (self.weights[i] / total_weight) + self.gamma / self.n_actions
            chosen_action = np.random.choice(self.n_actions, p=probabilities[i])
            chosen_actions[i] = chosen_action
            adjustments_out[i] = self.adjustments[chosen_action]
        
        # Update the weights for the machine that was played last.
        if play_history:
            played_machine = play_history[-1] - 1  # Convert to 0-index.
            # Define adversary reward: if player lost, reward = 1; if won, reward = 0.
            reward = 1 if not win_history[-1] else 0
            action_chosen = chosen_actions[played_machine]
            p = probabilities[played_machine][action_chosen]
            estimated_reward = reward / p if p > 0 else 0
            # Update weight using the Exp3 update rule.
            self.weights[played_machine][action_chosen] *= math.exp(self.gamma * estimated_reward / self.n_actions)
        
        return adjustments_out

import numpy as np
import random

class FPLHouse:
    def __init__(self, params):
        self.max_adjustment = params.get('max_adjustment', 0.1)
        self.n_machines = 4
        # Define discrete adjustment actions: decrease, no change, increase.
        self.actions = np.array([-self.max_adjustment, 0.0, self.max_adjustment])
        self.n_actions = len(self.actions)
        # Learning rate parameter for FPL.
        self.eta = params.get('eta', 0.1)
        # Cumulative rewards for each machine-action pair.
        self.cumulative_rewards = np.zeros((self.n_machines, self.n_actions))
        # Track the last action taken for each machine.
        self.last_actions = np.zeros(self.n_machines, dtype=int)

    def adjust_winrates(self, play_history, win_history, current_winrates):
        """
        Updates cumulative rewards based on the outcome of the player's last play.
        For each machine, perturbs the cumulative rewards and selects the action with the highest perturbed reward.
        Returns an adjustment vector corresponding to the chosen actions.
        """
        # Update cumulative rewards based on the last play.
        if play_history:
            played_machine = play_history[-1] - 1  # 0-indexed machine id.
            outcome = win_history[-1]
            # Reward is -1 if player won (bad for the house) and +1 if lost (good for the house).
            reward = -1 if outcome else 1
            # Update cumulative reward for the played machine's last action.
            last_action = self.last_actions[played_machine]
            self.cumulative_rewards[played_machine, last_action] += reward
        
        # For each machine, select an action by perturbing cumulative rewards.
        adjustments = np.zeros(self.n_machines)
        for i in range(self.n_machines):
            # Perturb cumulative rewards with exponential noise.
            perturbation = np.random.exponential(scale=1.0/self.eta, size=self.n_actions)
            perturbed_rewards = self.cumulative_rewards[i] + perturbation
            # Select the action with the highest perturbed reward.
            action = np.argmax(perturbed_rewards)
            self.last_actions[i] = action
            adjustments[i] = self.actions[action]
        
        return adjustments

import numpy as np
from scipy.stats import beta

class BayesianHouse(BaseHouse):
    def __init__(self, parameters):
        super().__init__(parameters)
        # Initialize Beta distribution parameters for each machine
        self.alphas = np.ones(self.n_machines)
        self.betas = np.ones(self.n_machines)
        self.max_adjustment = parameters.get('max_adjustment', 0.1)

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

        # Adjust win rates: decrease if expected win probability is high, increase if low
        adjustments = np.zeros(self.n_machines)
        for i in range(self.n_machines):
            if expected_win_probs[i] > 0.5:
                adjustments[i] = -self.max_adjustment * (expected_win_probs[i] - 0.5)
            else:
                adjustments[i] = self.max_adjustment * (0.5 - expected_win_probs[i])

        # Ensure adjustments are within the allowed range
        return np.clip(adjustments, -self.max_adjustment, self.max_adjustment)

import numpy as np

class BOBWHouse(BaseHouse):
    def __init__(self, parameters):
        super().__init__(parameters)
        self.total_plays = np.zeros(self.n_machines)
        self.total_rewards = np.zeros(self.n_machines)
        self.gamma = parameters.get('gamma', 0.07)  # Exploration parameter for EXP3
        self.weights = np.ones(self.n_machines)
        self.max_adjustment = parameters.get('max_adjustment', 0.1)
        self.threshold = parameters.get('threshold', 0.1)  # Threshold to detect adversarial behavior

    def adjust_winrates(self, play_history, win_history, current_winrates):
        n = len(play_history)
        if n == 0:
            return np.zeros(self.n_machines)

        # Update total plays and rewards
        for i in range(self.n_machines):
            plays = [j for j, p in enumerate(play_history) if p == i + 1]
            self.total_plays[i] = len(plays)
            self.total_rewards[i] = sum([win_history[j] for j in plays])

        # Calculate average rewards
        average_rewards = np.divide(self.total_rewards, self.total_plays, out=np.zeros_like(self.total_rewards), where=self.total_plays > 0)

        # Detect adversarial behavior based on reward variance
        reward_variance = np.var(average_rewards)
        if reward_variance > self.threshold:
            # Adversarial setting: Use EXP3
            probabilities = (1 - self.gamma) * (self.weights / np.sum(self.weights)) + (self.gamma / self.n_machines)
            chosen_machine = np.random.choice(np.arange(1, self.n_machines + 1), p=probabilities)
            reward = win_history[-1] if play_history[-1] == chosen_machine else 0
            estimated_reward = reward / probabilities[chosen_machine - 1]
            self.weights[chosen_machine - 1] *= np.exp(self.gamma * estimated_reward / self.n_machines)
            adjustments = -self.max_adjustment * (probabilities - 1 / self.n_machines)
        else:
            # Stochastic setting: Use UCB1
            total_counts = np.sum(self.total_plays)
            ucb_values = average_rewards + np.sqrt((2 * np.log(total_counts)) / (self.total_plays + 1e-5))
            best_machine = np.argmax(ucb_values)
            adjustments = -self.max_adjustment * (ucb_values - np.mean(ucb_values))

        # Ensure adjustments are within the allowed range
        return np.clip(adjustments, -self.max_adjustment, self.max_adjustment)

import numpy as np

class ExpectiminimaxHouse(BaseHouse):
    def __init__(self, parameters):
        super().__init__(parameters)
        self.max_depth = parameters.get('max_depth', 3)  # Depth of the search tree
        self.max_adjustment = parameters.get('max_adjustment', 0.1)  # Maximum adjustment per machine

    def adjust_winrates(self, play_history, win_history, current_winrates):
        # Initialize adjustments
        adjustments = np.zeros(self.n_machines)
        # Evaluate each machine's adjustment using Expectiminimax
        for i in range(self.n_machines):
            adjustments[i] = self.expectiminimax(play_history, win_history, current_winrates, i, self.max_depth, True)
        # Normalize adjustments to ensure the sum of win rates remains constant
        adjustments = self.normalize_adjustments(adjustments, current_winrates)
        return adjustments

    def expectiminimax(self, play_history, win_history, current_winrates, machine_index, depth, is_maximizing_player):
        if depth == 0 or len(play_history) == 0:
            return self.evaluate_state(play_history, win_history, current_winrates)
        if is_maximizing_player:
            max_eval = -np.inf
            for i in range(self.n_machines):
                eval = self.expectiminimax(play_history, win_history, current_winrates, i, depth - 1, False)
                max_eval = max(max_eval, eval)
            return max_eval
        else:
            expected_value = 0
            for outcome in [0, 1]:  # 0 for loss, 1 for win
                probability = current_winrates[machine_index] if outcome == 1 else 1 - current_winrates[machine_index]
                new_play_history = play_history + [machine_index + 1]
                new_win_history = win_history + [outcome]
                eval = self.expectiminimax(new_play_history, new_win_history, current_winrates, machine_index, depth - 1, True)
                expected_value += probability * eval
            return expected_value

    def evaluate_state(self, play_history, win_history, current_winrates):
        # Simple evaluation: negative of player's average winnings
        if len(win_history) == 0:
            return 0
        return -np.mean(win_history) * 2  # Each win gives the player 2 coins

    def normalize_adjustments(self, adjustments, current_winrates):
        # Apply adjustments to current win rates
        new_winrates = current_winrates + adjustments
        # Ensure win rates are within [0, 1]
        new_winrates = np.clip(new_winrates, 0, 1)
        # Normalize to maintain the sum of win rates
        total_winrate = np.sum(new_winrates)
        if total_winrate == 0:
            return adjustments  # Avoid division by zero
        scaling_factor = 2 / total_winrate
        normalized_winrates = new_winrates * scaling_factor
        # Calculate final adjustments
        final_adjustments = normalized_winrates - current_winrates
        # Clip adjustments to the maximum allowed adjustment
        final_adjustments = np.clip(final_adjustments, -self.max_adjustment, self.max_adjustment)
        return final_adjustments
