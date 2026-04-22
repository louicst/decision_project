import numpy as np
from MMDP import MMDP 
from forest import generate_forest_instance, evaluate_policy


def question_5_threshold_policies(mdp):
    """
    Evaluates threshold policies for tau in {0..10} and identifies 
    which ones are Pareto efficient.
    """
    print("--- QUESTION 5: Threshold Policies & Pareto Efficiency ---")
    
    # Store the results: list of dictionaries
    policies_data = []
    
    # 1. Generate and evaluate threshold policies
    for tau in range(11): # tau from 0 to 10
        # Create policy: action is HARVEST (1) if state_age > tau, else WAIT (0)
        # Note: state index 's' goes from 0 to 9, so age is 's + 1'
        policy = [1 if (s + 1) > tau else 0 for s in range(mdp.num_states)]
        
        # Evaluate using the unnormalized rewards to see the real values (Euros, Carbon)
        V_pi = mdp.evaluate_policy(policy, use_normalized_rewards=False)
        
        # The project states the initial state is s = 1 (index 0)
        v_initial = V_pi[0] 
        
        policies_data.append({
            'tau': tau,
            'policy': policy,
            'profit': v_initial[0],
            'carbon': v_initial[1],
            'bio': v_initial[2]
        })

    # 2. Identify Pareto Efficient policies
    pareto_efficient_taus = []
    
    for i, p1 in enumerate(policies_data):
        is_efficient = True
        v1 = np.array([p1['profit'], p1['carbon'], p1['bio']])
        
        for j, p2 in enumerate(policies_data):
            if i == j:
                continue
                
            v2 = np.array([p2['profit'], p2['carbon'], p2['bio']])
            
            # Check if p2 dominates p1:
            # p2 is >= p1 in all criteria AND strictly > in at least one
            if np.all(v2 >= v1) and np.any(v2 > v1):
                is_efficient = False
                break # No need to check other policies, p1 is dominated
                
        if is_efficient:
            pareto_efficient_taus.append(p1['tau'])
            p1['pareto'] = True
        else:
            p1['pareto'] = False

    # 3. Display the Table
    print(f"{'Tau':<5} | {'Profit (€)':<12} | {'Carbon':<10} | {'Biodiversity':<12} | {'Pareto Efficient'}")
    print("-" * 65)
    for p in policies_data:
        pareto_str = "YES" if p['pareto'] else "NO"
        print(f"{p['tau']:<5} | {p['profit']:<12.2f} | {p['carbon']:<10.2f} | {p['bio']:<12.2f} | {pareto_str}")
        
    return policies_data


import numpy as np

def get_lorenz_vector(v):
    """
    Computes the Lorenz vector for a given values vector v.
    """
    # 1. Sort the components in increasing order
    v_sorted = np.sort(v)
    
    # 2. Compute the cumulative sum (v1, v1+v2, v1+v2+v3)
    return np.cumsum(v_sorted)

def question_6_lorenz_efficient_policies(mdp, policies_data_from_q5):
    """
    Identifies which threshold policies are Lorenz efficient.
    CRITICAL: Must use normalized values for sorting to make sense.
    """
    print("\n--- QUESTION 6: Lorenz Efficiency ---")
    
    lorenz_data = []
    
    # 1. Re-evaluate policies to get their NORMALIZED values
    for p in policies_data_from_q5:
        tau = p['tau']
        policy = p['policy']
        
        # Setting use_normalized=True (or use_normalized_rewards=True based on your Q4 code)
        V_pi_norm = mdp.evaluate_policy(policy, use_normalized=True)
        
        # Extract the normalized values for the initial state (s=1, index 0)
        v_initial_norm = V_pi_norm[0] 
        
        # Calculate the Lorenz vector
        l_vec = get_lorenz_vector(v_initial_norm)
        
        lorenz_data.append({
            'tau': tau,
            'v_norm': v_initial_norm,
            'l_vec': l_vec,
            'lorenz_efficient': True # Assume True until proven otherwise
        })
        
    # 2. Check Lorenz dominance (Pareto dominance on the Lorenz vectors)
    for i, p1 in enumerate(lorenz_data):
        L1 = p1['l_vec']
        
        for j, p2 in enumerate(lorenz_data):
            if i == j:
                continue
                
            L2 = p2['l_vec']
            
            # Check if p2 Lorenz-dominates p1:
            # L2 >= L1 in all dimensions AND L2 > L1 in at least one
            if np.all(L2 >= L1) and np.any(L2 > L1):
                p1['lorenz_efficient'] = False
                break # No need to check other policies, p1 is dominated
                
    # 3. Print the results
    print(f"{'Tau':<5} | {'Normalized (P, C, B)':<30} | {'Lorenz Vector':<30} | {'Lorenz Efficient'}")
    print("-" * 88)
    for p in lorenz_data:
        v_str = f"({p['v_norm'][0]:.2f}, {p['v_norm'][1]:.2f}, {p['v_norm'][2]:.2f})"
        l_str = f"({p['l_vec'][0]:.2f}, {p['l_vec'][1]:.2f}, {p['l_vec'][2]:.2f})"
        eff_str = "YES" if p['lorenz_efficient'] else "NO"
        print(f"{p['tau']:<5} | {v_str:<30} | {l_str:<30} | {eff_str}")
        
    return lorenz_data

# --- How to run it in your main script ---
# S, A, T_matrix, R_matrix, gamma = generate_forest_instance()
# my_forest_mdp = MMDP(S, A, T_matrix, R_matrix, gamma)
# my_forest_mdp.normalize_rewards()
# q5_results = question_5_threshold_policies(my_forest_mdp)