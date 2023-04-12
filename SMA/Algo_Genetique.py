import math
import importlib
import mesa
import fonctions
importlib.reload(fonctions)
from fonctions import *
from math import inf
from mesa import Agent
from mesa import Model
from mesa.time import RandomActivation
from mesa.time import BaseScheduler
import random


# Définition de la fonction de sélection par tournoi
def selection(Couts, taille_tournoi):
    participants = random.sample(Couts.keys(), taille_tournoi)
    return min(participants, key=lambda x:Couts[x] )

# Définition de la fonction de croisement à un point
def croisement(parent1, parent2):
    point_croisement = random.randint(1, len(parent1) - 1)
    enfant1 = parent1[:point_croisement] + parent2[point_croisement:]
    enfant2 = parent2[:point_croisement] + parent1[point_croisement:]
    return enfant1,enfant2

# Définition de la fonction de mutation
def mutation(solution, taux_mutation):

    if random.random() < taux_mutation : 

        i = random.randint(0,len(solution)-1)
        solution[i] = max(solution) - solution[i]
        return solution
    """
    for i in range(len(solution)):
        if random.random() < taux_mutation:
            solution[i] = max(solution) - solution[i]  # Inversion de la valeur de l'élément
    """
    return solution

# Définition de l'algorithme génétique
def genetique(taille_population, taille_solution, taux_mutation, max_iterations):
    
    history = []
    population = [[random.randint(0, n) for j in range(taille_solution)] for i in range(taille_population)]
    
    for i in range(max_iterations):
        Couts = {i : cout_fonction(code_to_X2(population[i])) for i in  range(len(population))}
        parents = [population[selection(Couts, 2)] for j in range(taille_population)]
        
        if len(history) == 0:
            history.append(min(list(Couts.values())))
            
        else : 
            #history.append(min(min(list(Couts.values())),history[-1]))
            history.append(min(list(Couts.values())))

        
        enfants = []
        
        for j in range(0, taille_population, 2):
            enfants.append(croisement(parents[j], parents[j+1])[0])
            enfants.append(croisement(parents[j], parents[j+1])[1])
            
        population = [mutation(enfants[j], taux_mutation) for j in range(taille_population)]
        

        
        print(f'itération {i} ')

        
    meilleure_solution = min(population, key=lambda x: cout_fonction(code_to_X2(x)))
    meilleure_valeur = cout_fonction(code_to_X2(meilleure_solution))
    
    return meilleure_solution, meilleure_valeur , history
