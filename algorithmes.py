import pandas as pd
from os import listdir
from os.path import isfile, join
import numpy as np
import random
 
from scipy.spatial.distance import pdist , cdist


import copy
 
import pickle
import os 
import sys
import math
import itertools

##############################################################################################
parent_dir = sys.path[0]
 

data_depot = pd.read_csv(os.path.join(parent_dir, 'data/data_depot.csv'))
data_client_index = pd.read_csv(os.path.join(parent_dir, 'data/data_clients.csv'))

nombre_client = len(data_client_index)

with open('data/distance_matrix.pickle', 'rb') as handle:
    distance_matrix = pickle.load(handle)

with open(os.path.join(parent_dir,'data/times.pickle'), 'rb') as handle:
    times = pickle.load(handle)

with open('matrice_temps_Cij.pickle', 'rb') as handle:
    matrice_temps_Cij = pickle.load(handle)



CUSTOMER_DELIVERY_SERVICE_TIME = data_client_index['d'].values
TOTAL_WEIGHT_KG = data_client_index['TOTAL_WEIGHT_KG'].values
CUSTOMER_DELIVERY_SERVICE_TIME_FROM_DEPOT = data_depot['TIME_DISTANCE_MIN'].values
CUSTOMER_DELIVERY_SERVICE_DISTANCE_FROM_DEPOT = data_depot['DISTANCE_KM'].values

Q = 5000 # capacité maximale des voitures qu'on adopté 
V_moy=60 # vitesse moyenne des voitures qu'on adopté 
time_window = 950 # le temps limite du livraison 
 
#####################################################################

def cout(global_route,w = 1):
    # Initialisation du coût à 0
    cout = 0
    
    # Pour chaque route dans le global_route
    for route in global_route:
        # Calcul du coût pour aller du dépôt jusqu'au premier client
        #cout_depot_depart = CUSTOMER_DELIVERY_SERVICE_DISTANCE_FROM_DEPOT[route[0]]
        # Calcul du coût pour aller du dernier client jusqu'au dépôt
        #cout_depot_retour = CUSTOMER_DELIVERY_SERVICE_DISTANCE_FROM_DEPOT[route[-1]]
        # Calcul de la somme des distances entre chaque client de la route
        cout_route = sum([distance_matrix[route[i],route[i+1]] for i in range(len(route)-1)])
        
        # Ajout du coût de la route à la variable cout
        cout = cout + cout_route 

    # Retourne la somme de tous les coûts plus le nombre de voitures utilisées
    return(cout+w * len(global_route))

######################################################################
 
def get_route(list_client,time_window,Q):
    # Initialisation de la liste des arrêts de chaque voiture avec le dépôt
    arret = [0]
    i = 0
    # Tant qu'il reste des clients à visiter
    while i<len(list_client):
        # Poids du colis du client courant
        weight = TOTAL_WEIGHT_KG[list_client[i]]
        # Temps pour aller du dépôt au client courant
        time = 480 + matrice_temps_Cij[0, list_client[i]+1]

        # Tant que la fenêtre de temps n'est pas dépassée et que la capacité n'est pas dépassée
        while time < time_window and weight < Q: 
            i += 1
            # Si on a visité tous les clients, on sort de la boucle
            if i > len(list_client)-1:
                break
            # On ajoute le temps pour aller du client précédent au client courant
            time += matrice_temps_Cij[list_client[i-1]+1, list_client[i]+1]
            # On ajoute le poids du colis du client courant
            weight += TOTAL_WEIGHT_KG[list_client[i]]
             
        # On ajoute l'indice du dernier client visité à la liste des arrêts
        arret.append(i)
      
    # On crée la liste des routes en utilisant les indices des arrêts
    Global_route = [list_client[arret[k]:arret[k+1]] for k in range(len(arret)-1)]
    
    # On retourne la liste des routes
    return Global_route


#########################################################################################

def Voisinnage(solution_route):
    solution = list(itertools.chain.from_iterable(solution_route))
    voisin = solution.copy()
    i = random.randint(0, len(solution) - 1)  # Sélection d'un élément au hasard
    j = random.randint(0, len(solution) - 1)  # Sélection d'un autre élément au hasard
    voisin[i], voisin[j] = voisin[j], voisin[i]  # Échange des deux éléments

    return get_route(voisin,time_window, Q)


def recuit_simule(initial_state,  temperature_initiale=1.0, temperature_finale=1e-8, alpha=0.99):
    """
    Implémente l'algorithme de recuit simulé pour résoudre le problème de VRP.
    """
    
    current_state = initial_state
    current_energy = cout(current_state)
    best_state = current_state
    best_energy = current_energy
    temperature = temperature_initiale
    # Calculer le paramètre beta pour la loi de Boltzmann
    beta = abs(1/ (5 * math.log(0.66)))

    history_sol=[]
    history = []
    while temperature > temperature_finale:
        # Générer une nouvelle solution voisine
        new_state =  Voisinnage(current_state.copy())
        new_energy = cout(new_state)
        delta_energy = new_energy - current_energy
        # Accepter ou non la nouvelle solution
        if delta_energy < 0 or math.exp(-1 / (beta*temperature)) >random.random():
            current_state = new_state
            current_energy = new_energy

        # Mettre à jour la meilleure solution trouvée jusqu'à présent
         
        best_state = current_state
        best_energy = current_energy

        temperature *= alpha
        history.append(best_energy)
        history_sol.append(best_state)

    # Retourner la meilleure solution trouvée
    indice = history.index(min(history))
    return history_sol[indice] , history[indice],history
     

############################################################################################

 
# Définition de la fonction de sélection par tournoi
def selection(Couts, taille_tournoi):
    participants = random.sample(Couts.keys(), taille_tournoi) # Sélection de "taille_tournoi" éléments au hasard dans le dictionnaire "Couts"
    return min(participants, key=lambda x:Couts[x] ) # Retourne la clé ayant la plus petite valeur de coût dans "Couts"

# Définition de la fonction de croisement à un point
def croisement(parent1_route, parent2_route):
    # Point de croisement aléatoire
    parent1 = list(itertools.chain.from_iterable(parent1_route))
    parent2 = list(itertools.chain.from_iterable(parent2_route))
     
    croisement_point = random.randint(1, len(parent1) - 1)

    # Croisement

    enfant1 = copy.deepcopy(parent1[:croisement_point]) +copy.deepcopy(parent2[croisement_point:])
    enfant2 = copy.deepcopy(parent2[:croisement_point]) + copy.deepcopy(parent1[croisement_point:])
    
    # Correction de la répétition des éléments
    enfant_r1 = []
    for element in enfant1 :
        if enfant1.count(element) > 1 and element not in enfant_r1: # Si un élément est répété dans enfant1
            enfant_r1.append(element) # On l'ajoute dans enfant_r1
    enfant_r2 = []
    for element in enfant2 :
        if enfant2.count(element) > 1 and element not in enfant_r2: # Si un élément est répété dans enfant2
            enfant_r2.append(element) # On l'ajoute dans enfant_r2
    
    # Échange de la répétition d'éléments entre les deux enfants
    if len(enfant_r2)!=0:
        for i in range(len(enfant_r2)):
            enfant2[enfant2.index(enfant_r2[i])],enfant1[enfant1.index(enfant_r1[i])] = enfant_r1[i] , enfant_r2[i]
    
    return get_route(enfant1,time_window, Q), get_route(enfant2,time_window, Q)


# Définition de la fonction de mutation
def mutation(solution_route, taux_mutation):
    solution = list(itertools.chain.from_iterable(solution_route))
    if random.random() < taux_mutation : 

        n = len(solution)
        i = np.random.randint(n)
        j = np.random.randint(n)
        solution[j],solution[i] = solution[i],solution[j] # Échange de la position des éléments i et j dans la solution
    

    return(get_route(solution,time_window, Q))

def genetique(population, taux_mutation, max_iterations):
    # Initialisation des variables
    history = [] # Historique des meilleurs coûts pour chaque itération
    taille_population = len(population) # Taille de la population
    best_cout = float("inf") # Initialisation du meilleur coût à l'infini
    best_element = None # Initialisation du meilleur élément

    # Boucle principale de l'algorithme génétique
    for i in range(max_iterations):
        # Calcul des coûts pour chaque élément de la population
        Couts = {k: cout(population[k]) for k in range(taille_population)}
        
        # Sélection des parents pour la reproduction
        parents = [population[selection(Couts, 2)] for j in range(taille_population)]
        
        # Ajout du meilleur coût de cette itération à l'historique
        history.append(min(list(Couts.values())))
        
        # Reproduction (croisement)
        enfants = []
        for j in range(0, taille_population, 2):
            enfants1, enfants2 = croisement(parents[j], parents[j+1])
            enfants.append(enfants1)
            enfants.append(enfants2)

        # Mutation des enfants
        population = [mutation(enfants[j], taux_mutation) for j in range(taille_population)]
        
        # Mise à jour du meilleur coût et de l'élément associé
        for k in range(taille_population):
            cout_k = cout(population[k])
            if cout_k < best_cout:
                best_cout = cout_k
                best_element = population[k]

    indice = history.index(min(history))
    return best_element,history[indice] , history , population

########################################################################################

# Définition de la fonction de voisinage
def voisinage(solution_route):

    solution = list(itertools.chain.from_iterable(solution_route))
    voisin = solution.copy()
    i = random.randint(0, len(solution) - 1)  # Sélection d'un élément au hasard
    j = random.randint(0, len(solution) - 1)  # Sélection d'un autre élément au hasard
    voisin[i], voisin[j] = voisin[j], voisin[i]  # Échange des deux éléments

    return get_route(voisin,time_window, Q)

# Définition de l'algorithme Tabou
def tabou(liste_initiale, taille_tabou, max_iterations,n_voisin):
    

    meilleure_solution = liste_initiale # la meilleure solution trouvée jusqu'à présent
    meilleure_valeur = cout(meilleure_solution) # la valeur de la meilleure solution
    history_sol=[meilleure_solution]
    liste_tabou = [] # la liste tabou pour stocker les solutions interdites
    history = [meilleure_valeur] # tableau qui stocke la meilleure valeur de chaque itération

    
    for i in range(max_iterations):
        
        # recherche du meilleur voisin parmi les n_voisin voisins générés
        for i in range(n_voisin):
            voisin = voisinage(meilleure_solution)
            
            # Si le voisin est dans la liste tabou, on en génère un autre
            while voisin in liste_tabou:
                voisin = voisinage(meilleure_solution)
                

            if i==0:
                # si c'est le premier voisin, on initialise la meilleure valeur avec celle-ci
                valeur_meilleur_voisin = cout(voisin)
                meilleur_voisin = voisin
                 
            else:
                # sinon, on compare avec la valeur du voisin précédent
                s = cout(voisin)
                if s<valeur_meilleur_voisin:
                    meilleur_voisin = voisin
                    valeur_meilleur_voisin = s
            
        # on passe au meilleur voisin trouvé de la solution courante
        meilleure_solution = meilleur_voisin
        meilleure_valeur = valeur_meilleur_voisin
            
        # ajout de la solution courante à la liste tabou
        liste_tabou.append(meilleure_solution)
        
        # Si la taille de la liste tabou dépasse la taille maximale autorisée, on supprime le premier élément
        if len(liste_tabou) > taille_tabou:
            liste_tabou.pop(0)
            
        # ajout de la meilleure valeur à l'historique
        history.append(meilleure_valeur)
        history_sol.append(meilleure_solution)

    # retourne la meilleure solution trouvée, sa valeur et l'historique des meilleures valeurs
    indice = history.index(min(history))
    return history_sol[indice] , history[indice] , history

 
