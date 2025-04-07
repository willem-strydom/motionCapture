import numpy as np
from scipy.stats import beta

class BehavioralModel:
    def __init__(self):
        self.n_machines = 4
        self.max_adjustment = 0.1
        self.house_edge = 2

    def adjust_winrates(self, play_history, win_history, current_winrates):
        return np.zeros(self.n_machines)
    
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