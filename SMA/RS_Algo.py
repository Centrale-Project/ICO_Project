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
data_depot = pd.read_csv('data_depot.csv')
data_client_index = pd.read_csv('data_clients.csv')



with open('distance_matrix.pickle', 'rb') as handle:
    distance_matrix = pickle.load(handle)

with open('times.pickle', 'rb') as handle:
    times = pickle.load(handle)


n = 50

data_depot = data_depot.iloc[:n]
data_client_index = data_client_index.iloc[:n]
distance_matrix = distance_matrix[:n , :n]
times = times[:n,:]
def Voisinnage(T):
    n = len(T)
    i = np.random.randint(n)
    j = np.random.randint(n)
    T[j] = i
    T[i] = j
    return(T)
def recuit_simule(initial_state,  temperature_initiale=1.0, temperature_finale=1e-8, alpha=0.99):

    """Implémente l'algorithme de recuit simulé."""

    history = []
    current_state = initial_state
    current_energy = cout_fonction(code_to_X2(current_state))
    best_state = current_state
    best_energy = current_energy
    temperature = temperature_initiale

    while temperature > temperature_finale:
        
        #V = Voisinnage(current_state)
        new_state =  Voisinnage(current_state)
        new_energy = cout_fonction(code_to_X2(new_state))
        delta_energy = new_energy - current_energy

        if delta_energy < 0 or math.exp(-delta_energy / temperature) > random.random():
            current_state = new_state
            current_energy = new_energy

        if current_energy < best_energy:
            best_state = current_state
            best_energy = current_energy

        temperature *= alpha

        history.append(best_energy)

        print('best_energy : '  , best_energy)
        print('temperature : '  , temperature)
        print('new state : '  , current_state)


    return best_state, best_energy , history
code_I= [i for i in range(n)]
