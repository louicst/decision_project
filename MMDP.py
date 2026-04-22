import numpy as np

# --- QUESTION 1 : Classe de base MMDP ---
class MMDP:
    def __init__(self, states, actions, transitions, rewards, gamma):
        """
        Initializes the Multicriteria Markov Decision Process.
        """
        self.S = np.array(states)
        self.A = np.array(actions)
        self.T = np.array(transitions)
        
        # On sauvegarde les récompenses brutes pour l'affichage plus tard
        self.R_raw = np.copy(rewards) 
        # R sera la matrice normalisée (ou brute avant l'appel de normalize_rewards)
        self.R = np.copy(rewards)
        
        self.gamma = gamma
        
        # Metadata for dimensions
        self.num_states = len(self.S)
        self.num_actions = len(self.A)
        self.num_criteria = self.R.shape[2] 

    def get_transition_prob(self, s, a, s_prime):
        return self.T[s, a, s_prime]

    def get_reward_vector(self, s, a):
        return self.R[s, a]
    
    def normalize_rewards(self):
        """
        Rescales self.R for each criterion so that they lie in [0, 1].
        """
        r_min = self.R_raw.min(axis=(0, 1))
        r_max = self.R_raw.max(axis=(0, 1))
        
        denom = r_max - r_min
        denom[denom == 0] = 1.0 
        
        self.R = (self.R_raw - r_min) / denom


    # POUR LES QUESTIONS 5 et 6
    def evaluate_policy(self, policy, use_normalized=False):
        """
        Evaluates a given policy by solving the linear system:
        (I - gamma * P_pi) * V_pi = R_pi
        
        :param policy: A list or 1D array of length num_states (actions for each state).
        :param use_normalized: Boolean. True to use normalized rewards [0,1], 
                               False to use raw rewards (Euros, Carbon volume).
        :return: V_pi, a 2D numpy array of expected discounted sum of rewards.
        """
        P_pi = np.zeros((self.num_states, self.num_states))
        R_pi = np.zeros((self.num_states, self.num_criteria))
        
        # Sélection de la matrice de récompense
        reward_matrix = self.R if use_normalized else self.R_raw
        
        for s in range(self.num_states):
            a = policy[s]
            P_pi[s, :] = self.T[s, a, :]       
            R_pi[s, :] = reward_matrix[s, a, :] 
            
        I = np.eye(self.num_states)
        A = I - self.gamma * P_pi
        
        # Résolution du système : V = (I - gamma*P)^-1 * R
        V_pi = np.linalg.solve(A, R_pi)
        
        return V_pi