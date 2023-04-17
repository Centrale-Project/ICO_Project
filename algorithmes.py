import pandas as pd
from os import listdir
from os.path import isfile, join
import numpy as np
import random
import scipy 
from scipy.spatial.distance import pdist , cdist

import matplotlib.pyplot as plt
 
from math import sin, cos, sqrt, atan2, radians
 
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

with open('distance_matrix.pickle', 'rb') as handle:
    distance_matrix = pickle.load(handle)

with open(os.path.join(parent_dir,'data/times.pickle'), 'rb') as handle:
    times = pickle.load(handle)

with open('matrice_temps_Cij.pickle', 'rb') as handle:
    matrice_temps_Cij = pickle.load(handle)



CUSTOMER_DELIVERY_SERVICE_TIME = data_client_index['d'].values
TOTAL_WEIGHT_KG = data_client_index['TOTAL_WEIGHT_KG'].values
CUSTOMER_DELIVERY_SERVICE_TIME_FROM_DEPOT = data_depot['TIME_DISTANCE_MIN'].values
CUSTOMER_DELIVERY_SERVICE_DISTANCE_FROM_DEPOT = data_depot['DISTANCE_KM'].values

Q = 5000
V_moy=60
time_window = 950

######################################################################################

def get_route_by_car(client_disponible, Q, V_moy, time_window):
    """
    Cette fonction prend en entrée les clients disponibles, la capacité de chaque véhicule, la vitesse moyenne des véhicules,
    et la fenêtre de temps pour la livraison des clients.
    Elle retourne la liste ordonnée des clients visités par le véhicule, en respectant les contraintes de capacité
    et de fenêtre de temps.
    """
    solution = []
    t = 480 # heure de départ du véhicule depuis le dépôt (ici, 8h du matin)
    total_weight = 0 # poids total des colis livrés jusqu'à présent
    i=0

    client_courant = 0
    client_potentiel = client_disponible

    # chosir le client le plus proche de depot
    client_suivant = random.choice(client_potentiel)

    # ajouter le temps de service du client et le temps de trajet jusqu'au client
    t = t +  CUSTOMER_DELIVERY_SERVICE_TIME_FROM_DEPOT[client_suivant] + CUSTOMER_DELIVERY_SERVICE_TIME[client_suivant]
    total_weight = total_weight+TOTAL_WEIGHT_KG[client_suivant]
    
    # vérifier si le véhicule est encore en mesure de livrer le colis et si la livraison respecte la fenêtre de temps du client
    if total_weight<Q and t<time_window:
        solution.append(client_suivant)
    
    # retirer le client traité de la liste des clients disponibles
    client_disponible.remove(client_suivant)

    client_courant = client_suivant
    
    # liste des clients potentiellement livrables en respectant les contraintes
    client_potentiel = [client for client in client_disponible if  t+distance_matrix[client_courant,client]/V_moy < time_window and total_weight+TOTAL_WEIGHT_KG[client]<Q]
    i=1
    
    while len(client_potentiel)>0:
        
        # choisir un client parmi la liste des clients potentiellement livrables
        client_suivant = random.choice(client_potentiel)
         
        # ajouter le client à la solution et le retirer de la liste des clients disponibles
        solution.append(client_suivant)
        client_disponible.remove(client_suivant)

        # ajouter le temps de service du client et le temps de trajet jusqu'au client
        t = t + distance_matrix[client_courant,client_suivant]/V_moy + CUSTOMER_DELIVERY_SERVICE_TIME[client_suivant]
        total_weight = total_weight+TOTAL_WEIGHT_KG[client_suivant]
        client_courant = client_suivant
 
        # mise à jour de la liste des clients potentiellement livrables en respectant les contraintes
        client_potentiel = [client for client in client_disponible if  t+distance_matrix[client_courant,client]/V_moy < time_window and total_weight+TOTAL_WEIGHT_KG[client]<Q]
        i+=1
    
    # retourner la solution ordonnée des clients visités par le véhicule
    return(solution)

########################################################################

def get_route_version0(Q,V_moy,time_window):
    # Liste de tous les clients disponibles
    client_disponible = [i for i in range(len(data_client_index))]
    
    # Liste qui va contenir toutes les routes des voitures
    Global_route = []
    
    # Variable pour numéroter les voitures
    k=0
    
    # Tant qu'il y a encore des clients disponibles
    while len(client_disponible)>0:
        
        # On récupère la route optimale pour une voiture
        route_by_car = get_route_by_car(client_disponible,Q,V_moy,time_window)
        
        # On ajoute la route à la liste des routes des voitures
        Global_route.append(route_by_car)
        
        # On retire les clients de la route de la liste des clients disponibles
        client_disponible = [client for client in  client_disponible if client not in route_by_car]
        
        # On incrémente le compteur de voitures
        k+=1
    
    # On affiche le nombre de voitures utilisées
    print('nombre des voitures utilisées ',len(Global_route))
    
    # On retourne la liste des routes des voitures
    return(Global_route)

#####################################################################

def cout(global_route):
    # Initialisation du coût à 0
    cout = 0
    
    # Pour chaque route dans le global_route
    for route in global_route:
        # Calcul du coût pour aller du dépôt jusqu'au premier client
        cout_depot_depart = CUSTOMER_DELIVERY_SERVICE_DISTANCE_FROM_DEPOT[route[0]]
        # Calcul du coût pour aller du dernier client jusqu'au dépôt
        cout_depot_retour = CUSTOMER_DELIVERY_SERVICE_DISTANCE_FROM_DEPOT[route[-1]]
        # Calcul de la somme des distances entre chaque client de la route
        cout_route = sum([distance_matrix[route[i],route[i+1]] for i in range(len(route)-1)])
        
        # Ajout du coût de la route à la variable cout
        cout = cout + cout_route + cout_depot_retour + cout_depot_depart

    # Retourne la somme de tous les coûts plus le nombre de voitures utilisées
    return(cout+len(global_route))

######################################################################

def get_route_version1(list_client, time_window, Q):
    Global_route = []
    client_numbre = len(list_client)
    client_visited = 0

    while client_visited < client_numbre:
        # clients disponibles pour la tournée
        client_disponible = list_client[client_visited:]
        route_by_car = []
        # heure de départ depuis le dépôt
        t = 480 + CUSTOMER_DELIVERY_SERVICE_TIME_FROM_DEPOT[client_disponible[0]] + CUSTOMER_DELIVERY_SERVICE_TIME[client_disponible[0]]
        # poids total des colis dans le véhicule
        weight = TOTAL_WEIGHT_KG[client_disponible[0]]

        i = 0
        # Tant que la tournée n'est pas terminée et que le poids ne dépasse pas la limite autorisée
        while t < time_window and weight < Q:
            # Ajouter le client suivant à la tournée
            route_by_car.append(client_disponible[i])
            i += 1
            if i >= len(client_disponible):
                break
            # Mise à jour de l'heure d'arrivée chez le client suivant et du poids total du véhicule
            t = t + distance_matrix[client_disponible[i-1],client_disponible[i]] / V_moy + CUSTOMER_DELIVERY_SERVICE_TIME[client_disponible[i]]
            weight += TOTAL_WEIGHT_KG[client_disponible[i]]

        client_visited += i
        # Ajouter la tournée complète à la liste des tournées globales
        Global_route.append(route_by_car)

    return(Global_route)

###################################################################################

def get_route_version2(list_client, time_window, Q):
    # Initialize list with the first stop as depot
    stops = [0]
    
    i = 0
    while i < len(list_client):
        # Initialize weight and time
        weight = TOTAL_WEIGHT_KG[list_client[i]]
        time = 480 + distance_matrix[0][list_client[i]] + CUSTOMER_DELIVERY_SERVICE_TIME[list_client[i]]
        
        # Loop through clients until capacity or time window is reached
        while time < time_window and weight < Q:
            i += 1
            if i > len(list_client)-1:
                break
            # Update time and weight
            time += distance_matrix[list_client[i-1]][list_client[i]] / V_moy + CUSTOMER_DELIVERY_SERVICE_TIME[list_client[i]]
            weight += TOTAL_WEIGHT_KG[list_client[i]]
             
        # Append the index of the last client visited in this loop to stops list
        stops.append(i)
      
    # Create routes using the stops list
    Global_route = [list_client[stops[k]:stops[k+1]] for k in range(len(stops)-1)]
    
    return Global_route


#########################################################################################

def Voisinnage(solution_route):
    solution = list(itertools.chain.from_iterable(solution_route))
    voisin = solution.copy()
    i = random.randint(0, len(solution) - 1)  # Sélection d'un élément au hasard
    j = random.randint(0, len(solution) - 1)  # Sélection d'un autre élément au hasard
    voisin[i], voisin[j] = voisin[j], voisin[i]  # Échange des deux éléments

    return get_route_version2(voisin,time_window, Q)




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

import copy
# Définition de la fonction de sélection par tournoi
def selection(Couts, taille_tournoi):
    participants = random.sample(Couts.keys(), taille_tournoi) # Sélection de "taille_tournoi" éléments au hasard dans le dictionnaire "Couts"
    return min(participants, key=lambda x:Couts[x] ) # Retourne la clé ayant la plus petite valeur de coût dans "Couts"

# Définition de la fonction de croisement à un point
import random

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
    
    return get_route_version2(enfant1,time_window, Q), get_route_version2(enfant2,time_window, Q)


# Définition de la fonction de mutation
def mutation(solution_route, taux_mutation):
    solution = list(itertools.chain.from_iterable(solution_route))
    if random.random() < taux_mutation : 

        n = len(solution)
        i = np.random.randint(n)
        j = np.random.randint(n)
        solution[j],solution[i] = solution[i],solution[j] # Échange de la position des éléments i et j dans la solution
    

    return(get_route_version2(solution,time_window, Q))

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

    return best_element, best_cout, history , population


########################################################################################

import random

# Définition de la fonction de voisinage
def voisinage(solution_route):

    solution = list(itertools.chain.from_iterable(solution_route))
    voisin = solution.copy()
    i = random.randint(0, len(solution) - 1)  # Sélection d'un élément au hasard
    j = random.randint(0, len(solution) - 1)  # Sélection d'un autre élément au hasard
    voisin[i], voisin[j] = voisin[j], voisin[i]  # Échange des deux éléments

    return get_route_version2(voisin,time_window, Q)

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



#####################################################################################################


