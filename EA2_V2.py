# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 11:22:16 2024

@author: charl
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 16:38:55 2024

@author: charl
"""
import sys, os

os.chdir('C:\\Users\\charl\\OneDrive\\Documents\\GitHub\\evoman_framework')


from evoman.environment import Environment
from demo_controller import player_controller, enemy_controller
import random
import numpy as np
import time
from sklearn.cluster import DBSCAN
from statistics import mean
from sklearn.preprocessing import StandardScaler





experiment_name = 'dummy_demo'
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)



n_hidden_neurons=10
npop = 100
prob_c = 0.8 #probability for doing the crossover
prob_m = 0.4 #probability if a mutation will accur in a individual
dom_u = 1
dom_l = -1
n_points = 125
max_swaps = 20
min_samples= 5
amount_of_generations = 30
lambda_ = 200  # Number of children

# initializes simulation in individual evolution mode, for single static enemy.
env = Environment(experiment_name=experiment_name,
                  enemies=[2],
                  playermode="ai",
                  player_controller=player_controller(n_hidden_neurons),
                  enemymode="static",
                  level=3,
                  speed="fastest",
                  visuals=False)


n_vars = (env.get_num_sensors()+1)*n_hidden_neurons + (n_hidden_neurons+1)*5

start_pop = np.random.uniform(dom_l, dom_u, (npop, n_vars))

# runs simulation
def simulation(env,x):
    f,p,e,t = env.play(pcont=x)
    return f


# evaluation
def evaluate(x):
    return np.array(list(map(lambda y: simulation(env,y), x)))


def tournament_selection(pop, fitness):
    i1, i2 = np.random.randint(0, len(pop), 2)
    return pop[i1] if fitness[i1] > fitness[i2] else pop[i2]


# #Need to double check
def mutation(pop):
    
    
    mutated_pop = np.copy(pop)
    
    for individual in mutated_pop:
        if np.random.rand() < prob_m:
            # Get the number of weights (shape[0] * shape[1])

            num_swaps = np.random.randint(1, max_swaps)  # Randomly choosing 1 to 10 swaps, we can change also to fixed amount of swaps or more swaps
            
            for _ in range(num_swaps):
                # Randomly choose two positions in the weights array to swap
                column1 = np.random.randint(0, n_vars)
                column2 = np.random.randint(0, n_vars)
                
                #print(column1)
                #STILL NEED TO FIX THIS LINE, how to swap columns
                individual[[column1, column2]] = individual[[column2, column1]]

    return mutated_pop



def crossover_n_point(pop):
    #np.random.shuffle(pop)

    num_individuals, num_genes = pop.shape
    new_population = []
    
    
    fitness = evaluate(pop)

    for _ in range(lambda_ // 2):

        parent1 = tournament_selection(pop, fitness)
        parent2 = tournament_selection(pop, fitness)  
        if random.random() < prob_c:
       
            # Generate sorted unique crossover points
            crossover_points = sorted(random.sample(range(1, num_genes), n_points))
            
            # Add start and end points to create slices
            crossover_points = [0] + crossover_points + [num_genes]
            
            # Create segments from crossover points
            segments1 = [parent1[crossover_points[j]:crossover_points[j+1]] for j in range(len(crossover_points) - 1)]
            segments2 = [parent2[crossover_points[j]:crossover_points[j+1]] for j in range(len(crossover_points) - 1)]
            
            # Alternate segments based on 50% chance
            prob = random.random()
            child1_segments = [segments1[j] if prob < 0.5 else segments2[j] for j in range(len(segments1))]
            child2_segments = [segments2[j] if prob < 0.5 else segments1[j] for j in range(len(segments2))]
            
            # Concatenate segments to form children
            child1 = np.concatenate(child1_segments)
            child2 = np.concatenate(child2_segments)
            
            new_population.append(child1)
            new_population.append(child2)

        else:
            new_population.append(parent1)
            new_population.append(parent2)
    
    new_population = mutation(np.array(new_population))
    return new_population



#--------------------------------------------------------------------------------------
#cluster part



def adjust_eps(generation, max_generations):
    # Gradually reduce eps as we approach the final generation
    initial_eps = 1
    final_eps = 0.1
    
    return  1 #initial_eps - (initial_eps - final_eps) * (generation / max_generations)




def group_individuals_by_clusters(population, labels):
    clusters = {}
    
    for ind, label in zip(population, labels):
        if label not in clusters:
            clusters[label] = []  # Create a new cluster if it doesn't exist
        clusters[label].append(ind)  # Add the individual to the corresponding cluster
    return clusters




def select_fitessed(label, individuals, desired_size):
    selected = []
    total_individuals = lambda_ + desired_size
    


    # Calculate the number of individuals to select from this cluster
    cluster_size = len(individuals)
    ratio = cluster_size / total_individuals
    num_to_select = max(0, int(ratio * desired_size))  # Ensure at least one is selected if the cluster is non-empty

    # Evaluate fitness for individuals
    fitness_scores = evaluate(individuals)  # This returns a numpy array of fitness scores
    #print(mean(fitness_scores))

    # Combine individuals with their fitness scores
    individuals_with_fitness = list(zip(individuals, fitness_scores))

    # Sort individuals by fitness and select the fittest
    sorted_individuals = sorted(individuals_with_fitness, key=lambda pair: pair[1], reverse=True)
    selected.extend([ind for ind, _ in sorted_individuals[:num_to_select]])


    return selected # Ensure the final size is exactly desired_size







def run_generations_EA2(pop, amount_generations):
    for i in range(0, amount_generations):
        # Step 1: Normalize the population
        #scaled_population = scale_population(pop)
        
        print(np.max(evaluate(pop)))
        print(mean(evaluate(pop)))
        # Step 2: Cluster the population
        if i % 5 == 0:
            print(i)
            eps = adjust_eps(i, amount_generations)
            #print(f"Adjusted epsilon for generation {i}: {eps}")
            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            
            labels = dbscan.fit_predict(pop)  # Get cluster labels
    
            # Step 3: Group individuals by clusters
            clusters = group_individuals_by_clusters(pop, labels)

        # Step 4: Initialize a list for the new population
        new_population = []

        # Step 5: Perform crossover within each cluster
        for label, individuals in clusters.items():
            if len(individuals) >=2:
                cluster_offspring = crossover_n_point(np.array(individuals))
                clusters[label] = cluster_offspring
                new_population.extend(cluster_offspring)  # Add all individuals from each cluster to the list

# Convert the list of individuals into a NumPy array with dtype float64
        all_individuals_array = np.array(new_population, dtype=np.float64)

        total_clusters = len(clusters)
        print(f"Total number of clusters: {total_clusters}")
        # Step 6: Check if new_population is empty before proceeding
        if not new_population:
            print(f"No offspring generated in generation {i}.")
            

        # Convert new_population to a numpy array
        new_generation = all_individuals_array

        # Step 7: Select the best individuals from the combined population

        # Update population for the next generation
        pop = new_generation

    return pop



        
new_pop = run_generations_EA2(start_pop, amount_of_generations)
new_pop_ev = evaluate(new_pop)
  
print(mean(new_pop_ev))
print(np.max(new_pop_ev))
        


# print(mean(begin_pop))
print(mean(new_pop_ev))
    
# print(np.max(begin_pop))
print(np.max(new_pop_ev))


