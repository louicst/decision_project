from MMDP import MMDP
from forest import generate_forest_instance
from solvers import (
    solve_weighted_sum, 
    question_5_threshold_policies, 
    question_6_lorenz_efficient_policies,
    questions_8_to_10_diverse_lorenz_policies,
    tester_question_12
)
from plots import plot_pareto_frontier_2d

def main():
    print("====================================================")
    print("   PROJET MMDP : GESTION DE LA FORÊT DE PINS")
    print("====================================================\n")

    # --- ÉTAPE 1 & 2 : Modèle et Normalisation (Questions 1 et 2) ---
    states, actions, T, R, gamma = generate_forest_instance()
    forest_model = MMDP(states, actions, T, R, gamma)
    forest_model.normalize_rewards()
    print("[OK] Forêt générée et modèle MMDP normalisé.\n")

    # --- ÉTAPE 3 : Évaluation de Politique (Question 4) ---
    print("--- TEST QUESTION 4 : Évaluation de Politique ---")
    
    # Cas de test : "Récolte Systématique" (Action 1 partout)
    # Dans ce cas, on peut vérifier le calcul à la main pour l'état 1 :
    # V(1) = R(1, h) + gamma * V(1)  =>  V(1) = R(1, h) / (1 - gamma)
    policy_always_harvest = [1] * 10
    
    # Évaluation avec les récompenses brutes (use_normalized=False)
    v_eval = forest_model.evaluate_policy(policy_always_harvest, use_normalized=False)
    
    # 1. Calcul théorique "à la main" pour le Profit à l'état 1 (indice 0)
    # R_r(1, h) = (1 - 0.9*pf) * mu * v1 * T1 - Cp
    # R_r(1, h) = (1 - 0.9*0.02) * 0.9 * 14 * 7 - 1000 = -913.388
    # V_théo = -913.388 / (1 - 0.9) = -9133.88
    
    print(f"Politique 'Récolte Toujours' (Actions) : {policy_always_harvest}")
    print(f"Valeur attendue (Profit État 1) : ~ -9133.88")
    print(f"Valeur calculée (Profit État 1) : {v_eval[0, 0]:.2f}")
    print(v_eval)
    
    if abs(v_eval[0, 0] - (-9133.88)) < 1e-2:
        print("[SUCCÈS] La vérification manuelle confirme la Question 4.\n")
    else:
        print("[ERREUR] Écart détecté dans le calcul de la Question 4.\n")


    # --- ÉTAPE 4 & 5 : Politiques à seuil (Questions 5 et 6) ---
    q5_data = question_5_threshold_policies(forest_model)
    q6_data = question_6_lorenz_efficient_policies(forest_model, q5_data)

    # --- ÉTAPE 6 : Diversité et filtrage global (Questions 7, 8 et 10) ---
    # Scanne 66 combinaisons de poids et filtre les meilleures
    q8_10_data = questions_8_to_10_diverse_lorenz_policies(forest_model, step=0.1)

    print ("question 12 :")
    tester_question_12(forest_model)
    
    # --- ÉTAPE 7 : Visualisation (Question 11) ---
    # Génère et sauvegarde le graphique 2D
    plot_pareto_frontier_2d(q5_data, q8_10_data)

    

    print("\n====================================================")
    print("             FIN TOTALE DES TESTS")
    print("====================================================")

if __name__ == "__main__":
    main()