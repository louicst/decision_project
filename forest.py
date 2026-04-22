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
        
        # --- ACTION 0 : WAIT (l) ---
        # L'arbre vieillit (s+1) mais ne peut pas dépasser l'âge max (m-1 en Python)
        s_next = min(m - 1, s + 1)
        
        # Transitions (Probabilité de grandir vs Probabilité d'incendie)
        T[s, 0, s_next] = 1 - p_f  
        T[s, 0, 0] = p_f           # Retour à l'âge 1 (indice 0) si feu
        
        # Récompenses [Profit, Carbone, Bio]
        R[s, 0, 0] = 0.0 
        R[s, 0, 1] = (1 - p_f) * lambda_c * v[s_next] + p_f * lambda_c * v[0]
        R[s, 0, 2] = (1 - p_f) * D[s_next] + p_f * D[0]

        # --- ACTION 1 : HARVEST (h) ---
        # Transitions
        T[s, 1, 0] = 1.0  # On coupe et replante, retour à l'état 1 à 100%
        
        # Récompenses [Profit, Carbone, Bio]
        R[s, 1, 0] = (1 - 0.9 * p_f) * mu * v[s] * T_price[s] - C_p
        R[s, 1, 1] = lambda_c * v[0]
        R[s, 1, 2] = D[0]
        
    return list(range(m)), [0, 1], T, R, gamma