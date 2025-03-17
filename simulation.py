import numpy as np
import random

###########################################
# Part 1: Efficient Game Simulation Code  #
###########################################

def game_simulation(PlayerFunc, HouseFunc, MachineSum=2.0, MaxAdjustment=0.5, numberOfTurns=100):
    """
    Runs the game simulation for a given number of turns.
    
    Parameters:
      - PlayerFunc: a callable that returns a new player instance
      - HouseFunc: a callable that returns a new house instance
      - MachineSum: total sum of winrates (default 2.0)
      - MaxAdjustment: maximum adjustment per machine (default 0.1)
      - numberOfTurns: number of turns to simulate
      
    Returns:
      A dictionary with history of winrates, house adjustments, play history, win history, and total reward.
    """
    n_machines = 4
    winrates_history = []
    adjustments_history = []
    play_history = []
    win_history = []
    current_winrates = np.full(n_machines, MachineSum / n_machines)  # start evenly
    player = PlayerFunc()   # new player instance
    house = HouseFunc({'max_adjustment': MaxAdjustment})
    total_reward = 0

    for t in range(numberOfTurns):
        # Player predicts which machine to play
        chosen_machine = player.choose_machine(play_history, win_history)
        play_history.append(chosen_machine)
        # Roll the chosen machine: if random number < current winrate, win!
        machine_index = chosen_machine - 1
        if random.random() < current_winrates[machine_index]:
            outcome = True
            reward = 2
        else:
            outcome = False
            reward = 0
        win_history.append(outcome)
        total_reward += reward

        # Record current winrates
        winrates_history.append(current_winrates.copy())
        # House predicts adjustments based on current history and winrates
        adjustment = house.adjust_winrates(play_history, win_history, current_winrates)
        adjustments_history.append(adjustment.copy())
        # Update winrates: add adjustment, clip to [0,1], then normalize to sum = MachineSum
        new_winrates = current_winrates + adjustment
        new_winrates = np.clip(new_winrates, 0, 1)
        new_winrates = new_winrates / np.sum(new_winrates) * MachineSum
        current_winrates = new_winrates

    return {
        'winrates_history': winrates_history,
        'adjustments_history': adjustments_history,
        'play_history': play_history,
        'win_history': win_history,
        'total_reward': total_reward
    }

###############################
# Part 2: Player Simulations  #
###############################

class BasePlayer:
    def __init__(self, parameters):
        self.parameters = parameters
        self.n_machines = 4

    def choose_machine(self, play_history, win_history):
        raise NotImplementedError("BasePlayer is abstract.")


class EpsilonGreedyPlayer(BasePlayer):
    def __init__(self, parameters):
        super().__init__(parameters)
        self.epsilon = parameters.get('epsilon', 0.1)
        self.counts = np.zeros(self.n_machines)
        self.values = np.ones(self.n_machines)  # start optimistic

    def choose_machine(self, play_history, win_history):
        # Recompute empirical means from history
        if play_history:
            for i in range(self.n_machines):
                indices = [j for j, p in enumerate(play_history) if p == i+1]
                if indices:
                    # Note: reward is 2 for win, 0 for loss.
                    wins = sum([1 for j in indices if win_history[j]])
                    self.values[i] = (wins * 2) / len(indices)
                    self.counts[i] = len(indices)
        # With probability epsilon, choose a random machine.
        if random.random() < self.epsilon:
            return random.randint(1, self.n_machines)
        else:
            return int(np.argmax(self.values)) + 1


class KalmanFilterPlayer(BasePlayer):
    def __init__(self, parameters):
        super().__init__(parameters)
        # Q: estimated reward (win probability * 2); P: uncertainty
        self.Q = np.full(self.n_machines, 0.5)  
        self.P = np.full(self.n_machines, 1.0)
        self.R = parameters.get('R', 0.1)

    def choose_machine(self, play_history, win_history):
        # Update only for the last played machine.
        if play_history:
            last_machine = play_history[-1] - 1
            last_outcome = 1 if win_history[-1] else 0
            K = self.P[last_machine] / (self.P[last_machine] + self.R)
            self.Q[last_machine] = self.Q[last_machine] + K * (last_outcome - self.Q[last_machine])
            self.P[last_machine] = (1 - K) * self.P[last_machine]
        # Choose machine with highest estimated Q.
        return int(np.argmax(self.Q)) + 1


class UCBPlayer(BasePlayer):
    def __init__(self, parameters):
        super().__init__(parameters)
        self.counts = np.zeros(self.n_machines)
        self.values = np.ones(self.n_machines)
        self.c = parameters.get('c', 0.1)
        self.t = 0

    def choose_machine(self, play_history, win_history):
        self.t += 1
        if play_history:
            for i in range(self.n_machines):
                indices = [j for j, p in enumerate(play_history) if p == i+1]
                if indices:
                    wins = sum([1 for j in indices if win_history[j]])
                    self.values[i] = (wins * 2) / len(indices)
                    self.counts[i] = len(indices)
        ucb_values = np.zeros(self.n_machines)
        for i in range(self.n_machines):
            if self.counts[i] == 0:
                ucb_values[i] = float('inf')
            else:
                ucb_values[i] = self.values[i] + self.c * np.sqrt(np.log(self.t) / self.counts[i])
        return int(np.argmax(ucb_values)) + 1
    
class RandomPlayer(BasePlayer):
    def __init__(self, parameters):
        super().__init__(parameters)
        self.counts = np.zeros(self.n_machines)
        self.values = np.ones(self.n_machines)

    def choose_machine(self, play_history, win_history):
        return random.randint(1, 4)


class ThompsonSamplingPlayer(BasePlayer):
    def __init__(self, parameters):
        super().__init__(parameters)
        # Using a Beta prior for each machine; initialize with 1,1
        self.alphas = np.ones(self.n_machines)
        self.betas = np.ones(self.n_machines)

    def choose_machine(self, play_history, win_history):
        if play_history:
            for i in range(self.n_machines):
                indices = [j for j, p in enumerate(play_history) if p == i+1]
                if indices:
                    wins = sum([1 for j in indices if win_history[j]])
                    losses = len(indices) - wins
                    self.alphas[i] = wins + 1
                    self.betas[i] = losses + 1
        samples = [np.random.beta(self.alphas[i], self.betas[i]) for i in range(self.n_machines)]
        return int(np.argmax(samples)) + 1

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


class LSTMHouse(BaseHouse):
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


####################################################
# Part 4: Simulation of Simulations (Meta-Sim)     #
####################################################

def simulation_of_simulations(maxNumberTurns=100, maxNumberOfSimulations=10, 
                              player_means=None, player_variances=None, HouseModel=None):
    """
    Run statistical simulations for all 4 player models (each with randomized initial parameters)
    against a chosen House model.
    
    Returns a list of tuples:
      (sim_result, initial parameters used, player model name)
    """
    results = []
    player_models = {
        'EpsilonGreedy': EpsilonGreedyPlayer,
        'KalmanFilter': KalmanFilterPlayer,
        'UCB': UCBPlayer,
        'ThompsonSampling': ThompsonSamplingPlayer,
        'PurelyRandom' : RandomPlayer
    }
    # Ensure a House model is provided.
    if HouseModel is None:
        raise ValueError("A HouseModel must be provided.")

    # Loop over each player model and simulation instance.
    for model_name, PlayerClass in player_models.items():
        for sim in range(maxNumberOfSimulations):
            # Randomize initial parameters if provided.
            params = {}
            if player_means and model_name in player_means:
                for param, mean in player_means[model_name].items():
                    variance = player_variances[model_name].get(param, 0.01)
                    params[param] = random.gauss(mean, variance)
            # Create a lambda that returns an instance of the player with these parameters.
            player_func = lambda params=params: PlayerClass(params)
            sim_result = game_simulation(player_func, lambda kwargs: HouseModel(kwargs), 
                                         MachineSum=2.0, MaxAdjustment=0.1, numberOfTurns=maxNumberTurns)
            results.append((sim_result, params, model_name))
    return results


def run_simulations_for_house(HouseModel, maxNumberTurns, maxNumberOfSimulations, 
                              player_means, player_variances):
    """
    Run meta-simulations for a given House model, print details for each simulation, and
    print the average win rate (fraction of wins) for each player type.
    """
    sim_results = simulation_of_simulations(maxNumberTurns=maxNumberTurns, 
                                            maxNumberOfSimulations=maxNumberOfSimulations,
                                            player_means=player_means, 
                                            player_variances=player_variances, 
                                            HouseModel=HouseModel)
    
    # Dictionary to collect win rates per player model.
    win_rates_by_model = {}

    for sim_result, init_params, model in sim_results:
        total_reward = sim_result['total_reward']
        # Calculate win rate as the fraction of wins over total plays.
        win_rate = sum(sim_result['win_history']) / len(sim_result['win_history'])
        #print(f"Player model: {model}, Initial parameters: {init_params}, Total Reward: {total_reward}")
        
        if model not in win_rates_by_model:
            win_rates_by_model[model] = []
        win_rates_by_model[model].append(win_rate)

    print("\nAverage Win Rates per Player Type:")
    for model, rates in win_rates_by_model.items():
        avg_rate = sum(rates) / len(rates)
        print(f"{model}: {avg_rate:.3f}")


#####################
# Main Code Block   #
#####################

if __name__ == '__main__':
    # Define means and variances for player parameters for each model.
    player_means = {
        'EpsilonGreedy': {'epsilon': 0.3},
        'KalmanFilter': {'R': 0.3},
        'UCB': {'c': 0.3},
        'ThompsonSampling': {},  # No extra parameter in this simple version
        'PurelyRandom': {}
    }
    player_variances = {
        'EpsilonGreedy': {'epsilon': 0.1},
        'KalmanFilter': {'R': 0.1},
        'UCB': {'c': 0.1},
        'ThompsonSampling': {},
        'PurelyRandom': {}
    }
    
    # Run meta-simulations.
    print("House Model: RegressionHouse")
    sim_results = run_simulations_for_house(maxNumberTurns=1000, maxNumberOfSimulations=20,
                                              player_means=player_means, player_variances=player_variances,HouseModel=RegressionHouse)

    #sim_results = simulation_of_simulations(maxNumberTurns=1000, maxNumberOfSimulations=10,
    #                                          player_means=player_means, player_variances=player_variances,HouseModel=StatespaceHouse)

    #sim_results = simulation_of_simulations(maxNumberTurns=1000, maxNumberOfSimulations=10,
    #                                          player_means=player_means, player_variances=player_variances,HouseModel=VARHouse)
    print("House Model: LSTMHouse")
    sim_results = run_simulations_for_house(maxNumberTurns=1000, maxNumberOfSimulations=20,
                                              player_means=player_means, player_variances=player_variances,HouseModel=LSTMHouse)