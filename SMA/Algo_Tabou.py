from fonctions import *
import fonctions
import math
import importlib
import mesa
from math import inf
from mesa import Agent
from mesa import Model
from mesa.time import RandomActivation
from mesa.time import BaseScheduler

importlib.reload(fonctions)
import random


# Définition de la fonction de voisinage
def voisinage(solution):
    voisin = solution.copy()
    i = random.randint(0, len(solution) - 1)  # Sélection d'un élément au hasard
    j = random.randint(0, len(solution) - 1)  # Sélection d'un autre élément au hasard
    voisin[i], voisin[j] = voisin[j], voisin[i]  # Échange des deux éléments
    return voisin

# Définition de l'algorithme Tabou
def tabou(liste_initiale, taille_tabou, max_iterations):
    
    history = []
    meilleure_solution = liste_initiale
    meilleure_valeur = cout_fonction(code_to_X2(meilleure_solution))
    liste_tabou = []
    for i in range(max_iterations):
        
        voisin = voisinage(meilleure_solution)
        while voisin in liste_tabou:  # Si le voisin est dans la liste tabou, on en génère un autre
            voisin = voisinage(meilleure_solution)
        valeur_voisin = cout_fonction(code_to_X2(voisin))
        if valeur_voisin < meilleure_valeur:  # Si le voisin est meilleur que la solution courante
            meilleure_solution = voisin
            meilleure_valeur = valeur_voisin
            
            
        liste_tabou.append(voisin)  # On ajoute le voisin dans la liste tabou
        if len(liste_tabou) > taille_tabou:  # Si la taille de la liste tabou dépasse la taille maximale autorisée
            liste_tabou.pop(0)  # On supprime le premier élément de la liste tabou
            
        history.append(meilleure_valeur)
        print(meilleure_valeur )
        
       
        
    return meilleure_solution, meilleure_valeur , history

