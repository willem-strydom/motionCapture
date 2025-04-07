import numpy as np

class MocapModel:
    def __init__(self):
        self.n_machines = 4
        self.max_adjustment = 0.1
        self.house_edge = 2

    def adjust_winrates(self,foyer,play_history,win_history,current_winrates):
        return np.zeros(self.n_machines)
    
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
    