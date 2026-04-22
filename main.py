from MMDP import MMDP
from forest import generate_forest_instance
from solvers import solve_weighted_sum, question_5_threshold_policies, question_6_lorenz_efficient_policies

def main():
    print("====================================================")
    print("   PROJET MMDP : GESTION DE LA FORÊT DE PINS")
    print("====================================================\n")

    # --- ÉTAPE 1 : Génération de l'instance (Question 2) ---
    states, actions, T, R, gamma = generate_forest_instance()
    print(f"[OK] Instance générée : {len(states)} états, {len(actions)} actions.")

    # --- ÉTAPE 2 : Création du modèle MMDP (Question 1) ---
    # On crée l'objet avec les récompenses brutes
    forest_model = MMDP(states, actions, T, R, gamma)
    
    # On normalise les récompenses pour les calculs (Q1)
    forest_model.normalize_rewards()
    print("[OK] Modèle MMDP créé et récompenses normalisées.\n")


    # --- ÉTAPE 3 : Test de la Somme Pondérée (Question 4) ---
    print("--- TEST QUESTION 4 : Somme Pondérée ---")
    
    # Test 1 : Profil "Profit Max"
    weights_profit = [1.0, 0.0, 0.0]
    policy_p, _ = solve_weighted_sum(forest_model, weights_profit)
    print(f"Politique 'Profit Max'    : {policy_p}")

    # Test 2 : Profil "Équilibre Parfait"
    weights_bal = [0.33, 0.33, 0.34]
    policy_b, _ = solve_weighted_sum(forest_model, weights_bal)
    print(f"Politique 'Équilibrée'    : {policy_b}")

    # Test 3 : Profil "Écolo" (Biodiversité + Carbone)
    weights_eco = [0.0, 0.5, 0.5]
    policy_e, _ = solve_weighted_sum(forest_model, weights_eco)
    print(f"Politique 'Écologiste'    : {policy_e}\n")


    # --- ÉTAPE 4 : Analyse des Politiques à seuil (Question 5) ---
    # Cette fonction va afficher le tableau Pareto
    q5_data = question_5_threshold_policies(forest_model)


    # --- ÉTAPE 5 : Analyse de l'équité de Lorenz (Question 6) ---
    # Cette fonction va afficher le tableau Lorenz
    q6_data = question_6_lorenz_efficient_policies(forest_model, q5_data)

    print("\n====================================================")
    print("             FIN DES TESTS Q1 À Q6")
    print("====================================================")

if __name__ == "__main__":
    main()