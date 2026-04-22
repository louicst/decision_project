from MMDP import MMDP
from forest import generate_forest_instance
from solvers import solve_weighted_sum

# (Générer la forêt et créer l'objet MMDP comme d'habitude)
states, actions, T, R, gamma = generate_forest_instance()
forest_model = MMDP(states, actions, T, R, gamma)
forest_model.normalize_rewards()

# Testons un forestier capitaliste (100% Profit, 0% Carbone, 0% Bio)
poids_capitalistes = [1.0, 0.0, 0.0]
policy_cap, V_cap = solve_weighted_sum(forest_model, poids_capitalistes)
print("Politique Capitaliste :", policy_cap)

# Testons un forestier écolo (0% Profit, 50% Carbone, 50% Bio)
poids_ecolos = [0.0, 0.5, 0.5]
policy_eco, V_eco = solve_weighted_sum(forest_model, poids_ecolos)
print("Politique Écologiste :", policy_eco)