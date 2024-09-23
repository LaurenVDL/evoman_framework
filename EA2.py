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
max_swaps = 10
min_samples= 5
amount_of_generations = 30


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

start_pop = np.random.uniform(dom_l, dom_u, (npop, n_vars))

# runs simulation
def simulation(env,x):
    f,p,e,t = env.play(pcont=x)
    return f


# evaluation
def evaluate(x):
    return np.array(list(map(lambda y: simulation(env,y), x)))


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
    np.random.shuffle(pop)

    num_individuals, num_genes = pop.shape
    new_population = []
    
    if len(pop) % 2 != 0:
       pop = pop[:-1]
        
    for i in range(0, num_individuals, 2):
        parent1 = pop[i]
        parent2 = pop[i+1]
            
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

            
            
        new_population.append(parent1)
        new_population.append(parent2)
    
    new_population = mutation(np.array(new_population))
    return new_population



#--------------------------------------------------------------------------------------
#cluster part


# def scale_population(population):
#     scaler = StandardScaler()
#     return scaler.fit_transform(population)


def adjust_eps(generation, max_generations):
    # Gradually reduce eps as we approach the final generation
    initial_eps = 1.0
    final_eps = 0.1
    
    return 1#initial_eps - (initial_eps - final_eps) * (generation / max_generations)




def group_individuals_by_clusters(population, labels):
    clusters = {}
    
    for ind, label in zip(population, labels):
        if label not in clusters:
            clusters[label] = []  # Create a new cluster if it doesn't exist
        clusters[label].append(ind)  # Add the individual to the corresponding cluster
    
    return clusters




def select_fitessed(clusters, desired_size):
    selected = []
    total_individuals = sum(len(individuals) for individuals in clusters.values())
    
    # Iterate over clusters to select individuals based on the cluster size
    for label, individuals in clusters.items():
        if len(individuals) == 0:
            continue  # Skip empty clusters

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

    # If we still need more individuals, fill from noise (if applicable)
    if len(selected) < desired_size:
        noise_individuals = clusters.get(-1, [])  # Get noise individuals if any
        needed = desired_size - len(selected)
        
        # Evaluate fitness for noise individuals
        noise_fitness_scores = evaluate(noise_individuals)
        noise_individuals_with_fitness = list(zip(noise_individuals, noise_fitness_scores))
        
        # Sort noise individuals by fitness
        sorted_noise = sorted(noise_individuals_with_fitness, key=lambda pair: pair[1], reverse=True)
        selected.extend([ind for ind, _ in sorted_noise[:needed]])
    # Print the total number of clusters
    total_clusters = len(clusters)
    print(f"Total number of clusters: {total_clusters}")

    return np.array(selected, dtype=np.float64)  # Ensure the final size is exactly desired_size



def selection_desired_population_size(population, max_generations, number_generation):
    # Normalize the population
    #scaled_population = scale_population(population)

    # Adjust epsilon dynamically based on progress
    eps = adjust_eps(number_generation, max_generations)
    #print(f"Adjusted epsilon for generation {number_generation}: {eps}")
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)

    # Fit DBSCAN on the scaled population
    dbscan.fit(population)
    labels = dbscan.labels_

    # Group individuals by clusters
    clusters = group_individuals_by_clusters(population, labels)
    selected_population = select_fitessed(clusters, desired_size=npop)

    return np.array(selected_population, dtype=np.float64)





def run_generations_EA2(pop, amount_generations):
    for i in range(1, amount_generations):
        # Step 1: Normalize the population
        #scaled_population = scale_population(pop)
        
        print(np.max(evaluate(pop)))

        # Step 2: Cluster the population
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
            
    
            # Perform crossover within the current cluster
            if len(individuals) % 2 != 0:
               individuals = individuals[:-1]  # Remove the last individual   need to change if needed
            if len(individuals) >=2:
                cluster_offspring = crossover_n_point(np.array(individuals))
                new_population.extend(cluster_offspring)
       

        # Step 6: Check if new_population is empty before proceeding
        if not new_population:
            print(f"No offspring generated in generation {i}.")
            

        # Convert new_population to a numpy array
        new_population = np.array(new_population)

        # Step 7: Select the best individuals from the combined population
        new_generation = selection_desired_population_size(new_population, amount_generations, i)

        # Update population for the next generation
        pop = new_generation

    return pop


#-----------------------------------------------------------------------------------
#EA1 


def selection(population, num_best=npop):
    # Compute fitness scores for each individual
    

    fitness_scores = evaluate(population)  # This returns a numpy array of fitness scores
    print(np.max(fitness_scores))
        
    # Pair individuals with their fitness scores
    individuals_with_fitness = list(zip(population, fitness_scores))
    
    # Sort by fitness scores in descending order
    sorted_individuals = sorted(individuals_with_fitness, key=lambda pair: pair[1], reverse=True)
    
    # Select the top individuals based on fitness scores
    top_individuals = [ind for ind, _ in sorted_individuals[:num_best]]
    
    return np.array(top_individuals, dtype=np.float64)
    

def run_generations_EA1(pop, amount_generations):

    for i in range(0,amount_generations):
        new_pop_n_point = crossover_n_point(pop)
        new_generation = selection(new_pop_n_point)
        pop = new_generation
    return pop

#-----------------------------------------------------------------------------------------

        
new_pop = run_generations_EA2(start_pop, amount_of_generations)
new_pop = evaluate(new_pop)
  
print(mean(new_pop))
print(np.max(new_pop))
        

begin_pop = run_generations_EA1(start_pop,amount_of_generations)


begin_pop = evaluate(begin_pop)

print(mean(begin_pop))
print(mean(new_pop))
    
print(np.max(begin_pop))
print(np.max(new_pop))



