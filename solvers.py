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

import numpy as np

def question_5_threshold_policies(mdp):
    """
    Évalue les politiques à seuil pour tau dans {0..10} et identifie 
    lesquelles sont optimales au sens de Pareto.
    """
    print("\n--- QUESTION 5: Politiques à seuil & Frontière de Pareto ---")
    
    policies_data = []
    
    # 1. Générer et évaluer les politiques à seuil
    for tau in range(11): 
        # Action HARVEST (1) si l'âge (s+1) > tau, sinon WAIT (0)
        policy = [1 if (s + 1) > tau else 0 for s in range(mdp.num_states)]
        
        # On utilise les récompenses brutes (euros, etc.) pour l'affichage
        V_pi = mdp.evaluate_policy(policy, use_normalized=False)
        
        # On regarde la valeur depuis l'état initial (jeunes pousses, s=0)
        v_initial = V_pi[0] 
        
        policies_data.append({
            'tau': tau,
            'policy': policy,
            'profit': v_initial[0],
            'carbon': v_initial[1],
            'bio': v_initial[2]
        })

    # 2. Identifier les politiques efficaces au sens de Pareto
    pareto_efficient_taus = []
    
    for i, p1 in enumerate(policies_data):
        is_efficient = True
        v1 = np.array([p1['profit'], p1['carbon'], p1['bio']])
        
        for j, p2 in enumerate(policies_data):
            if i == j: continue
                
            v2 = np.array([p2['profit'], p2['carbon'], p2['bio']])
            
            # Dominance de Pareto : p2 est >= p1 partout ET strictement > quelque part
            if np.all(v2 >= v1) and np.any(v2 > v1):
                is_efficient = False
                break 
                
        p1['pareto'] = is_efficient
        if is_efficient:
            pareto_efficient_taus.append(p1['tau'])

    # 3. Affichage du tableau
    print(f"{'Tau':<5} | {'Profit (€)':<12} | {'Carbone':<10} | {'Biodiversité':<12} | {'Pareto Efficient'}")
    print("-" * 70)
    for p in policies_data:
        pareto_str = "OUI" if p['pareto'] else "NON"
        print(f"{p['tau']:<5} | {p['profit']:<12.2f} | {p['carbon']:<10.2f} | {p['bio']:<12.2f} | {pareto_str}")
        
    return policies_data


def get_lorenz_vector(v):
    """Calcule le vecteur de Lorenz (somme cumulée des valeurs triées)."""
    return np.cumsum(np.sort(v))

def question_6_lorenz_efficient_policies(mdp, policies_data_from_q5):
    """
    Identifie quelles politiques à seuil sont efficaces au sens de Lorenz.
    CRITIQUE : Doit utiliser les valeurs normalisées.
    """
    print("\n--- QUESTION 6: Équité et Optimalité de Lorenz ---")
    
    lorenz_data = []
    
    # 1. Ré-évaluer pour obtenir les valeurs NORMALISÉES
    for p in policies_data_from_q5:
        tau = p['tau']
        policy = p['policy']
        
        V_pi_norm = mdp.evaluate_policy(policy, use_normalized=True)
        v_initial_norm = V_pi_norm[0] 
        
        l_vec = get_lorenz_vector(v_initial_norm)
        
        lorenz_data.append({
            'tau': tau,
            'v_norm': v_initial_norm,
            'l_vec': l_vec,
            'lorenz_efficient': True 
        })
        
    # 2. Vérifier la dominance de Lorenz
    for i, p1 in enumerate(lorenz_data):
        L1 = p1['l_vec']
        for j, p2 in enumerate(lorenz_data):
            if i == j: continue
            L2 = p2['l_vec']
            
            if np.all(L2 >= L1) and np.any(L2 > L1):
                p1['lorenz_efficient'] = False
                break 
                
    # 3. Affichage
    print(f"{'Tau':<5} | {'Norm (Prof, Carb, Bio)':<30} | {'Vecteur de Lorenz':<30} | {'Lorenz Efficient'}")
    print("-" * 90)
    for p in lorenz_data:
        v_str = f"({p['v_norm'][0]:.2f}, {p['v_norm'][1]:.2f}, {p['v_norm'][2]:.2f})"
        l_str = f"({p['l_vec'][0]:.2f}, {p['l_vec'][1]:.2f}, {p['l_vec'][2]:.2f})"
        eff_str = "OUI" if p['lorenz_efficient'] else "NON"
        print(f"{p['tau']:<5} | {v_str:<30} | {l_str:<30} | {eff_str}")
        
    return lorenz_data