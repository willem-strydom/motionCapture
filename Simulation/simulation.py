import numpy as np
import random
from Houses import LSTMHouse, RegressionHouse, VARHouse, StatespaceHouse, RLHouse, ARIMAHouse, MarkovChainHouse, HeuristicAdjustmentHouse, Exp3House, FPLHouse, ExpectiminimaxHouse, BayesianHouse, BOBWHouse
from Players import EpsilonGreedyPlayer, KalmanFilterPlayer, UCBPlayer, RandomPlayer, ThompsonSamplingPlayer
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
        adjustment -= np.mean(adjustment)
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
        'EpsilonGreedy': {'epsilon': 0.1},
        'KalmanFilter': {'R': 0.1},
        'UCB': {'c': 0.1},
        'ThompsonSampling': {},  # No extra parameter in this simple version
        'PurelyRandom': {}
    }
    player_variances = {
        'EpsilonGreedy': {'epsilon': 0.02},
        'KalmanFilter': {'R': 0.02},
        'UCB': {'c': 0.02},
        'ThompsonSampling': {},
        'PurelyRandom': {}
    }
    
    # Run meta-simulations.
    print("House Model: RegressionHouse")
    sim_results = run_simulations_for_house(maxNumberTurns=200, maxNumberOfSimulations=5,
                                              player_means=player_means, player_variances=player_variances,HouseModel=RegressionHouse)

    sim_results = simulation_of_simulations(maxNumberTurns=200, maxNumberOfSimulations=5,
                                              player_means=player_means, player_variances=player_variances,HouseModel=StatespaceHouse)

    sim_results = simulation_of_simulations(maxNumberTurns=200, maxNumberOfSimulations=5,
                                              player_means=player_means, player_variances=player_variances,HouseModel=VARHouse)
    print("House Model: LSTMHouse")
    sim_results = run_simulations_for_house(maxNumberTurns=200, maxNumberOfSimulations=5,
                                              player_means=player_means, player_variances=player_variances,HouseModel=LSTMHouse)
    print("House Model: HeuristicAdjustmentHouse")
    sim_results = run_simulations_for_house(maxNumberTurns=200, maxNumberOfSimulations=5,
                                              player_means=player_means, player_variances=player_variances,HouseModel=HeuristicAdjustmentHouse)
    # LSTMHouse, RegressionHouse, VARHouse, StatespaceHouse, RLHouse, ARIMAHouse, MarkovChainHouse, ClusteringHouse, NeuralNetworkHouse
    #print("House: NeuralNetworkHouse")
    #sim_results = run_simulations_for_house(maxNumberTurns=100, maxNumberOfSimulations=5, player_means=player_means, player_variances=player_variances,HouseModel=NeuralNetworkHouse)
    #print("House: ClusteringHouse")
    #sim_results = run_simulations_for_house(maxNumberTurns=100, maxNumberOfSimulations=5, player_means=player_means, player_variances=player_variances,HouseModel=ClusteringHouse)
    print("House: MarkovChainHouse")
    sim_results = run_simulations_for_house(maxNumberTurns=200, maxNumberOfSimulations=5, player_means=player_means, player_variances=player_variances,HouseModel=MarkovChainHouse)
    print("House: ARIMAHouse")
    sim_results = run_simulations_for_house(maxNumberTurns=200, maxNumberOfSimulations=5, player_means=player_means, player_variances=player_variances,HouseModel=ARIMAHouse)
    print("House: RLHouse")
    sim_results = run_simulations_for_house(maxNumberTurns=200, maxNumberOfSimulations=5, player_means=player_means, player_variances=player_variances,HouseModel=RLHouse)
    print("House: Exp3House")
    sim_results = run_simulations_for_house(maxNumberTurns=200, maxNumberOfSimulations=5, player_means=player_means, player_variances=player_variances,HouseModel=Exp3House)
    print("House: FPLHouse")
    sim_results = run_simulations_for_house(maxNumberTurns=200, maxNumberOfSimulations=5, player_means=player_means, player_variances=player_variances,HouseModel=FPLHouse)

    print("House: BayesianHouse")
    sim_results = run_simulations_for_house(maxNumberTurns=200, maxNumberOfSimulations=5, player_means=player_means, player_variances=player_variances,HouseModel=BayesianHouse)

    print("House: BOBWHouse")
    sim_results = run_simulations_for_house(maxNumberTurns=200, maxNumberOfSimulations=5, player_means=player_means, player_variances=player_variances,HouseModel=BOBWHouse)

    print("House: ExpectiminimaxHouse")
    sim_results = run_simulations_for_house(maxNumberTurns=200, maxNumberOfSimulations=5, player_means=player_means, player_variances=player_variances,HouseModel=ExpectiminimaxHouse)
