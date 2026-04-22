import numpy as np

# Question 1
class MMDP:
    def __init__(self, states, actions, transitions, rewards, gamma):
        """
        on initilaise le Multicriteria Markov Decision Process
        """
        self.S = np.array(states)
        self.A = np.array(actions)
        self.T = np.array(transitions)
        
        # On sauvegarde les récompenses brutes pour l'affichage plus tard
        self.R_raw = np.copy(rewards) 
        # R = matrice normalisée (ou brute avant l'appel de normalize_rewards)
        self.R = np.copy(rewards)
        self.gamma =gamma
        
        # Metadata for dimensions
        self.num_states = len(self.S)
        self.num_actions = len(self.A)
        self.num_criteria = self.R.shape[2] 

    def get_transition_prob(self, s, a, s_prime):
        return self.T[s,a , s_prime]


  
    def get_reward_vector(self, s, a):
        return self.R[s, a]
    
    def normalize_rewards(self):
        """
        on normalise self.R for each critère pour que ça appartienne à [0,1]
        """
        r_min = self.R_raw.min(axis =(0, 1))
        r_max = self.R_raw.max(axis =(0, 1))
        
        denom = r_max - r_min
        denom[denom == 0]= 1.0 
        
        self.R = (self.R_raw - r_min)/ denom


    # question 5 et 6
    def evaluate_policy(self, policy, use_normalized=False):
        P_pi = np.zeros((self.num_states, self.num_states))
        R_pi = np.zeros((self.num_states, self.num_criteria))
        
        # on sélectionne la matrice de récompense
        reward_matrix = self.R if use_normalized else self.R_raw
        
        for s in range(self.num_states) :
            a =policy[s]
            P_pi[s, :]= self.T[s, a, :]       
            R_pi[s, :]= reward_matrix[s, a, :] 
            
        I =np.eye(self.num_states)
        A= I - self.gamma *P_pi
        
        # et on résout le systeme
        V_pi = np.linalg.solve(A, R_pi)
 
        return V_pi