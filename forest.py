import numpy as np

# --- QUESTION 2 : Instance de la Plantation d'Arbres ---
def generate_forest_instance():
    """
    Génère les paramètres (S, A, T, R, gamma) pour le MMDP de la forêt.
    """
    # --- 1. Paramètres de base ---
    m = 10  # Nombre d'états (classes d'âge 1 à 10)
    num_actions = 2  # 0: Wait (l), 1: Harvest (h)
    num_criteria = 3 # 0: Profit, 1: Carbone, 2: Biodiversité
    
    # Définition des constantes pour la lisibilité
    WAIT = 0
    HARVEST = 1

    # Données du tableau 1 (Attention: l'indice 0 correspond à la classe d'âge 1)
    v = np.array([14, 25, 43, 73, 117, 175, 240, 299, 345, 377])
    T_price = np.array([7, 7, 7, 8, 9, 12, 16, 19, 20, 21])
    D = np.array([0.14, 0.82, 1.0, 0.82, 0.57, 0.32, 0.14, 0.04, 0.0, 0.0])
    
    # Constantes du problème
    p_f = 0.02
    C_p = 1000
    mu = 0.9
    lambda_c = 0.3
    gamma = 0.9
    
    # --- 2. Initialisation des matrices ---
    T = np.zeros((m, num_actions, m))
    R = np.zeros((m, num_actions, num_criteria))
    
    # --- 3. Remplissage par état ---
    for s in range(m):
        
        # --- ACTION : WAIT (l) ---
        # L'arbre vieillit (s+1) mais ne peut pas dépasser l'âge max (m-1 en Python)
        s_next = min(m - 1, s + 1)
        
        # Transitions (Probabilité de grandir vs Probabilité d'incendie)
        T[s, WAIT, s_next] = 1 - p_f  
        T[s, WAIT, 0] = p_f           # Retour à l'âge 1 (indice 0) si feu
        
        # Récompenses [Profit, Carbone, Bio]
        R[s, WAIT, 0] = 0.0 
        R[s, WAIT, 1] = (1 - p_f) * lambda_c * v[s_next] + p_f * lambda_c * v[0]
        R[s, WAIT, 2] = (1 - p_f) * D[s_next] + p_f * D[0]

        # --- ACTION : HARVEST (h) ---
        # Transitions
        T[s, HARVEST, 0] = 1.0  # On coupe et replante, retour à l'état 1 à 100%
        
        # Récompenses [Profit, Carbone, Bio]
        R[s, HARVEST, 0] = (1 - 0.9 * p_f) * mu * v[s] * T_price[s] - C_p
        R[s, HARVEST, 1] = lambda_c * v[0]
        R[s, HARVEST, 2] = D[0]
        
    return list(range(m)), [WAIT, HARVEST], T, R, gamma

# --- QUESTION 4 : evaluation policy ---

def evaluate_policy(self, policy, use_normalized=False):
        """
        Evaluates a given policy by solving the linear system:
        (I - gamma * P_pi) * V_pi = R_pi
        
        :param policy: A list or 1D array of length num_states, where policy[s] 
                       is the action to take in state s.
        :param use_normalized: Boolean. If True, uses self.R_normalized for calculations.
                               If False, uses the raw self.R values.
        :return: V_pi, a 2D numpy array of shape (num_states, num_criteria) containing 
                 the expected discounted sum of rewards for each state and criterion.
        """
        # 1. Initialize the P_pi and R_pi matrices for the given policy
        P_pi = np.zeros((self.num_states, self.num_states))
        R_pi = np.zeros((self.num_states, self.num_criteria))
        
        # Select which reward matrix to use
        reward_matrix = self.R_normalized if use_normalized else self.R
        
        # 2. Populate P_pi and R_pi according to the actions dictated by the policy
        for s in range(self.num_states):
            a = policy[s]
            P_pi[s, :] = self.T[s, a, :]       # Extracts the transition probabilities
            R_pi[s, :] = reward_matrix[s, a, :] # Extracts the reward vector
            
        # 3. Construct the A matrix: (I - gamma * P_pi)
        I = np.eye(self.num_states)
        A = I - self.gamma * P_pi
        
        # 4. Solve the linear system A * V_pi = R_pi
        # np.linalg.solve is computationally more efficient and stable 
        # than manually calculating the inverse of A.
        V_pi = np.linalg.solve(A, R_pi)
        
        return V_pi