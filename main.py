import numpy as np
from MMDP import MMDP 
from forest import generate_forest_instance

def main():
    print("=== DÉBUT DES TESTS - Q1 & Q2 ===")

    # ---------------------------------------------------------
    # 1. Génération des matrices de la Forêt (Question 2)
    # ---------------------------------------------------------
    states, actions, T, R, gamma = generate_forest_instance()
    print("[OK] Données de la plantation générées (Taille S={}, A={}).".format(len(states), len(actions)))

    # ---------------------------------------------------------
    # 2. Création de l'objet MMDP (Question 1)
    # ---------------------------------------------------------
    # On fait une copie brute de R juste pour l'affichage dans ce test
    R_raw = np.copy(R) 
    
    forest_model = MMDP(states, actions, T, R, gamma)
    print("[OK] Objet MMDP instancié.")

    # ---------------------------------------------------------
    # 3. Normalisation (Question 1)
    # ---------------------------------------------------------
    # Comme ta fonction ne se lance pas toute seule, on l'appelle ici :
    forest_model.normalize_rewards()
    print("[OK] Récompenses normalisées entre [0, 1].")

    # ---------------------------------------------------------
    # 4. Vérification des Résultats (Le Test)
    # ---------------------------------------------------------
    print("\n=== TEST DE COHÉRENCE SUR L'ÂGE 5 ===")
    s_test = 4 # L'âge 5 correspond à l'indice 4 en Python
    a_harvest = 1 # Action Récolter
    a_wait = 0    # Action Attendre

    print("\n> ACTION : RÉCOLTER (Harvest)")
    print(f"Valeurs brutes       : Profit = {R_raw[s_test, a_harvest, 0]:.1f}€ | Carbone = {R_raw[s_test, a_harvest, 1]:.2f} | Bio = {R_raw[s_test, a_harvest, 2]:.2f}")
    
    norm_h = forest_model.get_reward_vector(s_test, a_harvest)
    print(f"Valeurs normalisées  : Profit = {norm_h[0]:.4f} | Carbone = {norm_h[1]:.4f} | Bio = {norm_h[2]:.4f}")

    print("\n> ACTION : ATTENDRE (Wait)")
    norm_w = forest_model.get_reward_vector(s_test, a_wait)
    print(f"Valeurs normalisées  : Profit = {norm_w[0]:.4f} | Carbone = {norm_w[1]:.4f} | Bio = {norm_w[2]:.4f}")

if __name__ == "__main__":
    main()