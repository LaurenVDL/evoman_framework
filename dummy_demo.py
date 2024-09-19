################################
# EvoMan FrameWork - V1.0 2016 #
# Author: Karine Miras         #
# karine.smiras@gmail.com      #
################################

# imports framework
import sys, os

os.chdir('C:\\Users\\charl\\OneDrive\\Documents\\GitHub\\evoman_framework')


from evoman.environment import Environment
from demo_controller import player_controller, enemy_controller
import random
import numpy as np





experiment_name = 'dummy_demo'
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)



n_hidden_neurons=10
npop = 100
prob_c = 0.8 #probability for doing the crossover
dom_u = 1
dom_l = -1


# initializes simulation in individual evolution mode, for single static enemy.
env = Environment(experiment_name=experiment_name,
                  enemies=[8],
                  playermode="ai",
                  player_controller=player_controller(n_hidden_neurons),
                  enemymode="static",
                  level=2,
                  speed="fastest",
                  visuals=False)




n_vars = (env.get_num_sensors()+1)*n_hidden_neurons + (n_hidden_neurons+1)*5


pop = np.random.uniform(dom_l, dom_u, (npop, n_vars))
#fit_pop = evaluate(pop)


def crossover(pop):
    new_population = []
    
    for i in range(0, len(pop), 2):
        # we can still add a different way of deciding of which parent
        parent1 = pop[i]
        parent2 = pop[i+1]
        
        child1, child2 = [], []
        
        if random.random() < prob_c:
            # For each gene in the parents ( we can still change this to a n.point)
            for gene1, gene2 in zip(parent1, parent2):
                # Flip a coin 
                if random.randint(0, 1) == 0:  
                    child1.append(gene1)
                    child2.append(gene2)
                else:  # Tails
                    child1.append(gene2)
                    child2.append(gene1)
        else:
            # If no crossover happens, children are copies of parents
            child1 = parent1[:]
            child2 = parent2[:]
        
        # Add the children to the new population
        new_population.append(child1)
        new_population.append(child2)
    
    return new_population
























