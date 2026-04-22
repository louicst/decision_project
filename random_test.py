import numpy as np
import matplotlib.pyplot as plt
from MMDP import MMDP
from solvers import solve_weighted_sum, calculer_politique_max_min, generate_weight_vectors

def run_random_experiments(num_mdps=20, num_states=15, num_actions=4, num_criteria=3):
    print(f"\n Question 13: Tests sur {num_mdps} MDPs aléatoires")
    print("Génération et calcul en cours : peut prendre quelques secondes")

    stats_ws_count, stats_ws_balance = [] , []
    stats_mm_count, stats_mm_balance = [] , []

    weights = generate_weight_vectors(step =0.1) 

    for i in range(num_mdps):
        # génération d'un MDP aléatoire
        T = np.random.rand(num_states, num_actions, num_states)
        for s in range(num_states):
            for a in range(num_actions):
                T[s, a, :] /= np.sum(T[s, a, :]) #on normalise les probas

        R = np.random.rand(num_states , num_actions, num_criteria)
        mdp= MMDP(list(range(num_states)) , list(range(num_actions)), T, R, gamma=0.9)
        mdp.normalize_rewards()

        #test Somme pondéré
        unique_pols_ws = {}
        for w in weights:
            pol, _ = solve_weighted_sum(mdp, w)
            pol_tuple = tuple(pol)
            if pol_tuple not in unique_pols_ws :
                v_norm = mdp.evaluate_policy(pol, use_normalized=True)[0]
                balance = np.std(v_norm) 
                unique_pols_ws[pol_tuple]= balance
                
        stats_ws_count.append(len(unique_pols_ws))
        stats_ws_balance.append(np.mean(list(unique_pols_ws.values())))

        #Test MaxMin
        unique_pols_mm = {}
        for w in weights :
            pol, _ = calculer_politique_max_min(mdp , w)
            pol_tuple = tuple(pol)
            if pol_tuple not in unique_pols_mm:
                v_norm =mdp.evaluate_policy(pol , use_normalized=True)[0]
                balance = np.std(v_norm)
                unique_pols_mm[pol_tuple]= balance
                
        stats_mm_count.append(len(unique_pols_mm))
        stats_mm_balance.append(np.mean(list(unique_pols_mm.values())))

    print("\n=== RÉSULTATS STATISTIQUES ===")
    print(f"Somme Pondérée : {np.mean(stats_ws_count):.1f} solutions | Déséquilibre moyen (Écart-type) : {np.mean(stats_ws_balance):.3f}")
    print(f"Max-Min        : {np.mean(stats_mm_count):.1f} solutions | Déséquilibre moyen (Écart-type) : {np.mean(stats_mm_balance):.3f}")

    fig, ax1 = plt.subplots(figsize=(10, 6))

    labels = ['Somme Pondérée', 'Max-Min']
    counts = [np.mean(stats_ws_count), np.mean(stats_mm_count)]
    balances = [np.mean(stats_ws_balance), np.mean(stats_mm_balance)]

    color = 'tab:blue'
    ax1.set_ylabel('Nombre moyen de politiques uniques trouvées', color=color, fontweight='bold')
    ax1.bar([0, 1], counts, color=color, width=0.4, alpha=0.7)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_xticks([0, 1])
    ax1.set_xticklabels(labels, fontsize=12, fontweight='bold')

    ax2 = ax1.twinx()  
    color = 'tab:red'
    ax2.set_ylabel('Déséquilibre (Écart-type des critères - Plus bas = Mieux)', color=color, fontweight='bold')
    ax2.plot([0, 1], balances, color=color, marker='o', markersize=12, linewidth=3)
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title(f"Comparaison Somme Pondérée vs Max-Min\n(Moyenne sur {num_mdps} MDPs aléatoires)", fontsize=14, pad=15)
    plt.grid(True, linestyle=':', alpha=0.5)
    plt.savefig("random_stats.png", dpi=300, bbox_inches='tight')
    print("[OK] L'image 'random_stats.png' a été sauvegardée !")
    plt.show()

if __name__ == "__main__":
    run_random_experiments()