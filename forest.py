import numpy as np

# QUESTION 2
def generate_forest_instance():
    """
    on génère les paramètres (S, A, T, R,gamma) pour le MMDP de la forêt

    """
    m =10  
    num_actions = 2  
    num_criteria =3 
    
    WAIT = 0
    HARVEST =1

    v = np.array([14 , 25, 43, 73, 117, 175, 240, 299, 345, 377])
    T_price = np.array([7, 7, 7 , 8, 9,12, 16, 19, 20, 21])
    D = np.array([0.14, 0.82, 1.0, 0.82, 0.57 , 0.32,0.14, 0.04,0.0, 0.0])
    
    p_f =0.02
    C_p = 1000
    mu =0.9
    lambda_c = 0.3
    gamma= 0.9
    
    T = np.zeros((m , num_actions, m))
    R = np.zeros((m , num_actions, num_criteria))
    
    for s in range(m):
        # l'action WAIT (l)
        s_next =min(m -1 , s + 1)
        T[s, WAIT, s_next] = 1 - p_f  
        T[s, WAIT, 0] =p_f           
        
        R[s, WAIT, 0] =0.0 
        R[s, WAIT, 1] = (1 - p_f) * lambda_c * v[s_next] + p_f * lambda_c * v[0]
        R[s, WAIT, 2] = (1 - p_f) * D[s_next] + p_f * D[0]

        #et l'action HARVEST (h)
        T[s, HARVEST, 0] = 1.0  
        R[s, HARVEST, 0] = (1 - 0.9 *p_f) * mu * v[s] * T_price[s] -C_p
        R[s, HARVEST, 1] = lambda_c * v[0]
        R[s, HARVEST, 2] = D[0]
        
    return list(range(m)) , [WAIT , HARVEST], T,R, gamma