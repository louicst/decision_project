import numpy as np

# la question 4 :
def solve_weighted_sum(mmdp, weights, epsilon=1e-6):
    """
    ça résout le problème multicritère en utilisant la somme pondérée avec l'algo Value Iteration
    
    """
    weights =np.array(weights)
    
    #on écrase la matrice de récompense multicritère
    # pour faire la matrice de récompense scalaire (S, A)
    R_w =np.dot(mmdp.R , weights)
    
    # init Value Iteration
    V =np.zeros(mmdp.num_states )
    policy =np.zeros(mmdp.num_states, dtype =int)
    
    #algo Value Iteration
    while True:
        V_prev = np.copy(V)
        delta = 0
        
        for s in range(mmdp.num_states) :
            Q_s =np.zeros(mmdp.num_actions )
            
            # we calcule valeur Q for each action
            for a in range(mmdp.num_actions):
                # Espérance des gains futurs
                expected_future = np.sum(mmdp.T[s , a,:] * V_prev)

                # la big équation e Bellman
                Q_s[a] = R_w[s , a] + mmdp.gamma * expected_future
                
            # l'état prend le best
            V[s] = np.max(Q_s)
            #et on garde la best action en tête
            policy[s] = np.argmax(Q_s)
            
            #la plus grande différence pour la convergence
            delta = max(delta, abs(V[s] - V_prev[s]))
            
        if delta < epsilon:
            break
            
    return policy.tolist(), V

def question_5_threshold_policies(mdp):  
    """
    Donne les politiques à seuil pour tau dans {0..10} 
    and we indentify lesquelles sont optimales au sens de Pareto
    """
    print("\n Question 5 : ") 
    
    policies_data = []
    
    # générer et évaluer les politiques à seuil
    for tau in range(11): 
        # Action HARVEST (1) si l'âge (s+1) > tau, sinon WAIT (0)
        policy = [1 if (s + 1) > tau else 0 for s in range(mdp.num_states)]
        
        # we use les récompenses brutes (euros, ...)
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

    # les politiques efficaces au sens de Pareto
    pareto_efficient_taus = []
    
    for i, p1 in enumerate(policies_data):
        is_efficient = True
        v1 = np.array([p1['profit'], p1['carbon'], p1['bio']])
        
        for j, p2 in enumerate(policies_data):
            if i == j: continue
                
            v2 = np.array([p2['profit'], p2['carbon'], p2['bio']])
            
            # Dominance de Pareto :
            if np.all(v2 >= v1) and np.any(v2 > v1):
                is_efficient = False
                break 
                
        p1['pareto'] = is_efficient
        if is_efficient:
            pareto_efficient_taus.append(p1['tau'])

    print(f"{'Tau':<5} | {'Profit (€)':<12} | {'Carbone':<10} | {'Biodiversité':<12} | {'Pareto Efficient'}")
    print("-" * 70)
    for p in policies_data:
        pareto_str = "OUI" if p['pareto'] else "NON"
        print(f"{p['tau']:<5} | {p['profit']:<12.2f} | {p['carbon']:<10.2f} | {p['bio']:<12.2f} | {pareto_str}")
        
    return policies_data


def get_lorenz_vector(v):
    """Calcule le vecteur de Lorenz"""
    return np.cumsum(np.sort(v))

def question_6_lorenz_efficient_policies(mdp, policies_data_from_q5):
    """
    Identify quelles politiques à seuil sont efficaces au sens de Lorenz.
    """
    print("\n Question 6 :")
    
    lorenz_data = []
    
    # pour obtenir les valeurs normalized
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
        
    # check la dominance de Lorenz
    for i, p1 in enumerate(lorenz_data):
        L1 = p1['l_vec']
        for j, p2 in enumerate(lorenz_data):
            if i == j: continue
            L2 = p2['l_vec']
            
            if np.all(L2 >= L1) and np.any(L2 > L1):
                p1['lorenz_efficient'] = False
                break 
                
    print(f"{'Tau':<5} | {'Norm (Prof, Carb, Bio)':<30} | {'Vecteur de Lorenz':<30} | {'Lorenz Efficient'}")
    print("-" * 90)
    for p in lorenz_data:
        v_str = f"({p['v_norm'][0]:.2f}, {p['v_norm'][1]:.2f}, {p['v_norm'][2]:.2f})"
        l_str = f"({p['l_vec'][0]:.2f}, {p['l_vec'][1]:.2f}, {p['l_vec'][2]:.2f})"
        eff_str = "OUI" if p['lorenz_efficient'] else "NON"
        print(f"{p['tau']:<5} | {v_str:<30} | {l_str:<30} | {eff_str}")
        
    return lorenz_data

def generate_weight_vectors(step=0.1):
    """
    Question 7 : génère a set of weight vectors (w1, w2, w3) 
    avec la somme = 1.0
    """
    weights = []
    num_steps = int(np.round(1.0 / step))
    
    for w1 in range(num_steps + 1):
        for w2 in range(num_steps + 1 - w1):
            w3 = num_steps - w1 - w2
            weights.append([w1 * step, w2 * step, w3 * step])
            
    return weights

def questions_8_to_10_diverse_lorenz_policies(mdp, step=0.1):
    """
    Questions 8 et 10 : Uses the weight generator to find diverse policies,
    then filters them to keep only the Lorenz-efficient ones.
    """
    print(f"\n--- QUESTIONS 8 & 10: Diverse Weights & Lorenz Filtering ---")
    weights_list = generate_weight_vectors(step)
    
    unique_policies = []
    seen_policies = set() # To track duplicates
    
    # --- QUESTION 8: Generate Policies ---
    for w in weights_list:
        # Run Value Iteration for this specific weight vector
        policy, _ = solve_weighted_sum(mdp, w)
        
        # Convert list to tuple so we can store it in a set
        pol_tuple = tuple(policy)
        
        # If this is a new policy we haven't seen before, process it
        if pol_tuple not in seen_policies:
            seen_policies.add(pol_tuple)
            
            # Get normalized values (for Lorenz math) and raw values (for display)
            V_norm = mdp.evaluate_policy(policy, use_normalized=True)[0]
            V_raw = mdp.evaluate_policy(policy, use_normalized=False)[0]
            
            l_vec = get_lorenz_vector(V_norm)
            
            unique_policies.append({
                'policy': policy,
                'weight_example': w, # Just saving one weight vector that generated it
                'v_norm': V_norm,
                'v_raw': V_raw,
                'l_vec': l_vec,
                'lorenz_efficient': True # Assume true until checked
            })
            
    print(f"Generated {len(weights_list)} weight combinations.")
    print(f"Found {len(unique_policies)} unique policies.")
    
    # --- QUESTION 10: Lorenz Filtering ---
    for i, p1 in enumerate(unique_policies):
        L1 = p1['l_vec']
        for j, p2 in enumerate(unique_policies):
            if i == j: continue
            L2 = p2['l_vec']
            
            # Lorenz Dominance check
            if np.all(L2 >= L1) and np.any(L2 > L1):
                p1['lorenz_efficient'] = False
                break 
                
    # --- Display Results ---
    print(f"\n{'Policy (Actions for States 1 to 10)':<35} | {'Profit (€)':<10} | {'Carbon':<8} | {'Bio':<8} | {'Lorenz Efficient'}")
    print("-" * 90)
    for p in unique_policies:
        pol_str = str(p['policy'])
        eff_str = "YES" if p['lorenz_efficient'] else "NO"
        print(f"{pol_str:<35} | {p['v_raw'][0]:<10.2f} | {p['v_raw'][1]:<8.2f} | {p['v_raw'][2]:<8.2f} | {eff_str}")
        
    return unique_policies


# --- QUESTION 12 : Stratégie Max-Min (Égalitariste) ---

def calculer_politique_max_min(modele, poids, marge_erreur=1e-6):
    """
    Trouve la meilleure politique en utilisant la méthode Max-Min.
    Au lieu d'additionner, on regarde le pire critère et on essaie de l'améliorer.
    """
    poids = np.array(poids)
    
    # On calcule les récompenses du pire scénario pour chaque état et action
    recompenses_pire_cas = np.min(modele.R * poids, axis=2)
    
    valeurs = np.zeros(modele.num_states)
    politique = np.zeros(modele.num_states, dtype=int)
    
    while True:
        anciennes_valeurs = np.copy(valeurs)
        ecart_max = 0
        
        for etat in range(modele.num_states):
            scores_actions = np.zeros(modele.num_actions)
            
            for action in range(modele.num_actions):
                gain_futur = np.sum(modele.T[etat, action, :] * anciennes_valeurs)
                scores_actions[action] = recompenses_pire_cas[etat, action] + (modele.gamma * gain_futur)
                
            valeurs[etat] = np.max(scores_actions)
            politique[etat] = np.argmax(scores_actions)
            
            ecart_max = max(ecart_max, abs(valeurs[etat] - anciennes_valeurs[etat]))
            
        if ecart_max < marge_erreur:
            break
            
    return politique.tolist(), valeurs

def tester_question_12(modele):
    """
    Teste notre nouvelle méthode Max-Min avec plusieurs profils.
    """
    print("\n--- QUESTION 12: Stratégie Alternative (Max-Min) ---")
    
    # 3 tests avec des priorités différentes
    tests_poids = [
        [0.33, 0.33, 0.34], # 1. Équilibre parfait
        [0.50, 0.25, 0.25], # 2. Priorité Profit
        [0.10, 0.45, 0.45]  # 3. Priorité Écologie
    ]
    
    resultats = []
    
    for profil_poids in tests_poids:
        # On calcule la politique avec notre nouvel algo
        ma_politique, _ = calculer_politique_max_min(modele, profil_poids)
        
        # On récupère les vraies valeurs en Euros et Tonnes
        vrais_scores = modele.evaluate_policy(ma_politique, use_normalized=False)[0]
        
        resultats.append({
            'poids': profil_poids,
            'politique': ma_politique,
            'profit': vrais_scores[0],
            'carbone': vrais_scores[1],
            'bio': vrais_scores[2]
        })
        
    # Affichage du tableau final
    print(f"{'Poids (Prof, Carb, Bio)':<25} | {'Politique trouvée':<35} | {'Profit':<8} | {'Carbone':<8} | {'Bio':<8}")
    print("-" * 95)
    
    for res in resultats:
        texte_poids = str(res['poids'])
        texte_politique = str(res['politique'])
        print(f"{texte_poids:<25} | {texte_politique:<35} | {res['profit']:<8.2f} | {res['carbone']:<8.2f} | {res['bio']:<8.2f}")