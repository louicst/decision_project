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

    # --- ÉTAPE 3 : Somme Pondérée (Question 4) ---
    print("--- TEST QUESTION 4 : Somme Pondérée ---")
    p_p, _ = solve_weighted_sum(forest_model, [1.0, 0.0, 0.0])
    print(f"Profit Max : {p_p}")
    p_b, _ = solve_weighted_sum(forest_model, [0.33, 0.33, 0.34])
    print(f"Équilibrée : {p_b}")
    p_e, _ = solve_weighted_sum(forest_model, [0.0, 0.5, 0.5])
    print(f"Écologiste : {p_e}\n")

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