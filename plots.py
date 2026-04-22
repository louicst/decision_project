import matplotlib.pyplot as plt

def plot_pareto_frontier_2d(q5_data, q8_data):
    """
    affiche les points et l'enveloppe convexe
     
    """
    print("\n QUESTION 11 : Génération du graphique Pareto")
    
    # on prépare les listes de points
    carb_all = [p['carbon'] for p in q5_data]
    prof_all = [p['profit'] for p in q5_data]
    
    #On trie les points de Q8 pour tracer une ligne
    q8_sorted = sorted(q8_data, key=lambda x: x['v_raw'][1])
    carb_q8 = [p['v_raw'][1] for p in q8_sorted]
    prof_q8 =[p['v_raw'][0] for p in q8_sorted]

    #on trace
    plt.figure(figsize=(10, 6))
    
    #tous les points (Q5)
    plt.scatter(carb_all , prof_all , color='blue', label='Toutes les politiques')
    
    # ligne rouge de l'enveloppe convexe (Q8)
    plt.plot(carb_q8, prof_q8, color='red' , linestyle='--' , marker='x', label='Somme Pondérée (Q8)')

    # we add le texte à côté de chaque puntos
    for p in q5_data:
        plt.text(p['carbon'] , p['profit'] + 100 ,f"T{p['tau']}" , ha ='center')

    plt.title("Frontière de Pareto : Carbone vs Profit")
    plt.xlabel("Carbone")
    plt.ylabel("Profit")
    plt.axhline(0, color='black', linewidth=1) #la ligne du zéro
    plt.grid(True)
    plt.legend()
    
    plt.savefig("pareto_plot.png")
    print("Graphique sauvegardé sous 'pareto_plot.png'")
    plt.show()