import numpy as np
import random

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
