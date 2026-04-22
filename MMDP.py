import numpy as np

class MMDP:
    def __init__(self, states, actions, transitions, rewards, gamma):
        """
        Initializes the Multicriteria Markov Decision Process.
        
        :param states: List or range of integers representing states S.
        :param actions: List or range of integers representing actions A.
        :param transitions: A 3D numpy array T[s, a, s'] representing P(s' | s, a).
        :param rewards: A 3D numpy array R[s, a, l] representing the reward vector 
                        for taking action a in state s for l criteria.
        :param gamma: Discount factor between 0 and 1.
        """
        self.S = np.array(states)
        self.A = np.array(actions)
        self.T = np.array(transitions)
        self.R = np.array(rewards)
        self.gamma = gamma
        
        # Metadata for dimensions
        self.num_states = len(self.S)
        self.num_actions = len(self.A)
        self.num_criteria = self.R.shape[2]  # l dimension for criteria 

    def get_transition_prob(self, s, a, s_prime):
        """Returns T(s, a, s')."""
        return self.T[s, a, s_prime]

    def get_reward_vector(self, s, a):
        """Returns the l-dimensional reward vector for state s and action a."""
        return self.R[s, a]
    
    def normalize_rewards(self):
        """
        Calculates the normalized rewards in [0, 1] and stores them in self.R_normalized,
        preserving the original self.R values for final reporting.
        """
        # 1. Find min and max for each criterion across all states and actions
        r_min = self.R.min(axis=(0, 1))
        r_max = self.R.max(axis=(0, 1))
        
        # 2. Avoid division by zero if all rewards for a criterion are the same
        denom = r_max - r_min
        denom[denom == 0] = 1.0 
        
        # 3. Apply the formula and save to a NEW attribute, leaving self.R untouched
        self.R_normalized = (self.R - r_min) / denom
        
        # Optional: You can also save the min and max values if you ever need 
        # to "un-normalize" something later.
        self.r_min = r_min
        self.r_max = r_max

# Example Usage for the Tree Plantation Case Study:
# States: e.g., 0=Young, 1=Mature, 2=Old
# Actions: 0=Wait, 1=Harvest
# Criteria: 0=Profit, 1=Carbon, 2=Biodiversity