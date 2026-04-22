import numpy as np

# --- QUESTION 4 : Résolution par Somme Pondérée ---

def solve_weighted_sum(mmdp, weights, epsilon=1e-6):
    """
    Résout le problème multicritère en utilisant la scalarisation (somme pondérée)
    avec l'algorithme Value Iteration.
    
    :param mmdp: L'objet MMDP (qui contient T et R normalisé).
    :param weights: Liste de poids (ex: [0.5, 0.25, 0.25]) qui somment à 1.
    :param epsilon: Seuil de convergence.
    :return: (policy, V)
             policy : La meilleure action pour chaque état (liste).
             V : Le vecteur des valeurs scalaires pour chaque état.
    """
    weights = np.array(weights)
    
    # 1. SCALARISATION : On écrase la matrice de récompense multicritère (S, A, L)
    # en une matrice de récompense scalaire (S, A) grâce au produit scalaire avec les poids.
    R_w = np.dot(mmdp.R, weights)
    
    # Initialisation de Value Iteration
    V = np.zeros(mmdp.num_states)
    policy = np.zeros(mmdp.num_states, dtype=int)
    
    # 2. ALGORITHME VALUE ITERATION
    while True:
        V_prev = np.copy(V)
        delta = 0
        
        for s in range(mmdp.num_states):
            Q_s = np.zeros(mmdp.num_actions)
            
            # Calcul de la valeur Q pour chaque action
            for a in range(mmdp.num_actions):
                # Espérance des gains futurs : sum( T(s,a,s') * V(s') )
                expected_future = np.sum(mmdp.T[s, a, :] * V_prev)
                # Equation de Bellman scalaire
                Q_s[a] = R_w[s, a] + mmdp.gamma * expected_future
                
            # L'état prend la valeur de la meilleure action
            V[s] = np.max(Q_s)
            # On mémorise quelle action était la meilleure
            policy[s] = np.argmax(Q_s)
            
            # Calcul de la plus grande différence pour la convergence
            delta = max(delta, abs(V[s] - V_prev[s]))
            
        # Condition d'arrêt : si ça ne bouge presque plus, on s'arrête
        if delta < epsilon:
            break
            
    return policy.tolist(), V