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

# os.chdir('C:\\Users\\charl\\OneDrive\\Documents\\GitHub\\evoman_framework')


from evoman.environment import Environment
from demo_controller import player_controller, enemy_controller
import random
import numpy as np
import time
from sklearn.cluster import DBSCAN
from statistics import mean
from sklearn.preprocessing import StandardScaler

# Data Gatherer class
class DataGatherer:
    def __init__(self, experiment_name):
        self.name = experiment_name
        self.mean_fitness = np.array([])
        self.std_fitness = np.array([])
        self.best_fitness = np.array([])
        self.generations = np.array([])
        self.best_gen = 0
        self.stats = None

        # Create directories for storing data
        os.makedirs(f"{self.name}/best", exist_ok=True)

    def gather(self, pop, pop_fit, gen):
        current_mean = np.mean(pop_fit)
        current_std = np.std(pop_fit)
        current_best = np.max(pop_fit)
        
        self.mean_fitness = np.append(self.mean_fitness, current_mean)
        self.std_fitness = np.append(self.std_fitness, current_std)
        self.best_fitness = np.append(self.best_fitness, current_best)
        self.generations = np.append(self.generations, gen)

        # Update the generation with the best solution if the new best fitness is higher
        if current_best >= np.max(self.best_fitness):
            self.best_gen = gen

        # Stack the gathered data
        self.stats = np.column_stack([self.generations, self.mean_fitness, self.std_fitness, self.best_fitness])

        # Save stats without header
        np.savetxt(f"{self.name}/stats.out", self.stats, delimiter=',', fmt='%.6f')

        # Save best solution
        np.savetxt(f"{self.name}/best/{gen}.out", pop[np.argmax(pop_fit)], delimiter=',', fmt='%1.2e')

        # Save the simulation state for future evaluation
        solutions = [pop, pop_fit]
        env.update_solutions(solutions)
        env.save_state()

    def add_header_to_stats(self):
        header = "Generation,Mean_Fitness,Std_Fitness,Best_Fitness\n"
        
        # Read existing content
        with open(f"{self.name}/stats.out", 'r') as f:
            content = f.read()
        
        # Write header and content
        with open(f"{self.name}/stats.out", 'w') as f:
            f.write(header + content)



headless = True
if headless:
    os.environ["SDL_VIDEODRIVER"] = "dummy"


experiment_name = 'EA_2_V2'
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)



n_hidden_neurons=10
npop = 100
prob_c = 0.7 #probability for doing the crossover
mutation_rate = 0.25 #probability if a mutation will accur in a individual
dom_u = 1
dom_l = -1
n_points = 10
#max_swaps = 20
min_samples= 5  #number of individuals in a cluster
amount_generations = 30
eps = 0.5

# initializes simulation in individual evolution mode, for single static enemy.
env = Environment(experiment_name=experiment_name,
                  enemies=[5],
                  playermode="ai",
                  player_controller=player_controller(n_hidden_neurons),
                  enemymode="static",
                  level=2,
                  speed="fastest",
                  visuals=False)

env.state_to_log()  # Checks environment state

# Track execution time
start_time = time.time()


n_vars = (env.get_num_sensors()+1)*n_hidden_neurons + (n_hidden_neurons+1)*5

start_pop = np.random.uniform(dom_l, dom_u, (npop, n_vars))

# Initialize DataGatherer
data_gatherer = DataGatherer(experiment_name)

# runs simulation
def simulation(env,x):
    print("individual shape: ", x.shape)
    f,p,e,t = env.play(pcont=x)
    print(f"Player Energy: {p}, Enemy Energy: {e}")
    print("Gain: ", p-e)
    return f


# evaluation
def evaluate(x):
    return np.array(list(map(lambda y: simulation(env,y), x)))


def tournament_selection(pop, fitness):
    i1, i2 = np.random.randint(0, len(pop), 2)
    return pop[i1] if fitness[i1] > fitness[i2] else pop[i2]


def apply_limits(individual):
    return np.clip(individual, dom_l, dom_u)


# #Need to double check
# def mutation(pop):
    
    
    
#     for individual in pop:
#         if np.random.rand() < prob_m:
#             # Get the number of weights (shape[0] * shape[1])

#             num_swaps = np.random.randint(1, max_swaps)  # Randomly choosing 1 to max_swaps swaps, we can change also to fixed amount of swaps or more swaps
            
#             for _ in range(num_swaps):

#                 column1 = np.random.randint(0, n_vars)
#                 individual[column1] = np.random.randint(-1, 1)

#     return pop

# Swap mutation
def mutation(individual, mutation_rate):
    if np.random.uniform(0, 1) <= mutation_rate:
        idx1, idx2 = np.random.randint(0, len(individual), 2)
        individual[idx1], individual[idx2] = individual[idx2], individual[idx1]
    return individual

def adaptive_mutation(individual, mutation_rate, generation, max_generations):
    # Increase mutation rate as generations progress
    adaptive_rate = mutation_rate * (1 + generation / max_generations)
    return mutation(individual, adaptive_rate)


# n-point crossover function
def crossover_n_point(pop):
    num_individuals, num_genes = pop.shape
    new_population = []
    
    for i in range(0, num_individuals, 2):
        parent1 = pop[i]
        parent2 = pop[i + 1]
        
        if random.random() < prob_c:
            # Generate sorted unique crossover points
            crossover_points = sorted(random.sample(range(1, num_genes), n_points))
            crossover_points = [0] + crossover_points + [num_genes]  # Add boundaries
            
            # Create segments from crossover points
            segments1 = [parent1[crossover_points[j]:crossover_points[j + 1]] for j in range(len(crossover_points) - 1)]
            segments2 = [parent2[crossover_points[j]:crossover_points[j + 1]] for j in range(len(crossover_points) - 1)]
            
            # Alternate segments based on random choice
            child1_segments = [segments1[j] if random.random() < 0.5 else segments2[j] for j in range(len(segments1))]
            child2_segments = [segments2[j] if random.random() < 0.5 else segments1[j] for j in range(len(segments2))]
            
            # Concatenate segments to form children
            child1 = np.concatenate(child1_segments)
            child2 = np.concatenate(child2_segments)
        else:
            # If no crossover occurs, children are copies of parents
            child1 = np.copy(parent1)
            child2 = np.copy(parent2)
        
        new_population.append(child1)
        new_population.append(child2)
    
    return np.array(new_population)

#--------------------------------------------------------------------------------------
#cluster part



# def adjust_eps(generation, max_generations):
#     # Gradually reduce eps as we approach the final generation
#     initial_eps = 1
#     final_eps = 0.1
    
#     return  1#initial_eps - (initial_eps - final_eps) * (generation / max_generations)



def group_individuals_by_clusters(population, labels):
    clusters = {}
    
    for ind, label in zip(population, labels):
        if label not in clusters:
            clusters[label] = []  # Create a new cluster if it doesn't exist
        clusters[label].append(ind)  # Add the individual to the corresponding cluster
    return clusters



def run_generations_EA2(pop, amount_generations):
    for i in range(0, amount_generations):
        # Step 1: Normalize the population
        #scaled_population = scale_population(pop)
        
        # print('Generation: ', i)
        # print('Best fitness: ', np.max(evaluate(pop)))
        # print('Average fitness: ', mean(evaluate(pop)))
        # Step 2: Cluster the population
        if i % 5 == 0:
            # eps = adjust_eps(i, amount_generations)
            # #print(f"Adjusted epsilon for generation {i}: {eps}")
            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            
            labels = dbscan.fit_predict(pop)  # Get cluster labels
    
            # Step 3: Group individuals by clusters
            clusters = group_individuals_by_clusters(pop, labels)
    
        # Step 4: Initialize a list for the new population
        new_population = []
        
        # Step 5: Perform crossover within each cluster
        for label, individuals in clusters.items():
            #print('Number of individuals in cluster: ', len(individuals))
            if len(individuals) >=2:
                individuals = np.array(individuals)
                fitness = evaluate(individuals)
                new_off_spring = []
                
                for _ in range(len(individuals)):       
                    # Select parents
                    parent1 = tournament_selection(individuals, fitness)
                    parent2 = tournament_selection(individuals, fitness)
                    
                    # Perform n-point crossover
                    offspring1, offspring2 = crossover_n_point(np.array([parent1, parent2]))
                    
                    # Apply mutation (swap mutation)
                    offspring1 = adaptive_mutation(offspring1, mutation_rate, i, amount_generations)
                    offspring2 = adaptive_mutation(offspring2, mutation_rate, i, amount_generations)
                    
                    # Apply limits to offspring
                    offspring1 = apply_limits(offspring1)
                    offspring2 = apply_limits(offspring2)
                    
                    new_off_spring.append(offspring1)
                    new_off_spring.append(offspring2)

                
                cluster_offspring = np.array(new_off_spring, dtype=np.float64)   
                
                offspring_fitness = evaluate(cluster_offspring)
                
                
                best_indices = np.argsort(offspring_fitness)[-len(individuals):]  # Get indices of the top fittest individuals proportional to the cluster size
                population = cluster_offspring[best_indices]  # Select the best individuals based on those indices
                fitness = offspring_fitness[best_indices]  # Get the fitness scores for the selected individuals
    
                
                #print('Population: ', len(population))
                # Overwrite the existing individuals in the cluster with the new population
                clusters[label] = population  # Replace existing individuals with the new population
                
                new_population.extend(population)  # Add all individuals from each cluster to the list
    
        total_clusters = len(clusters)
        #print(f"Total number of clusters: {total_clusters}")
        # Step 6: Check if new_population is empty before proceeding
        if not new_population:
            print(f"No offspring generated in generation {i}.")
            
    
        # Convert new_population to a numpy array
        pop = np.array(new_population, dtype=np.float64)

        print('Generation: ', i, 'Best fitness: ', np.max(evaluate(pop)))

        data_gatherer.gather(pop, evaluate(pop), i)


    return pop



        
new_pop = run_generations_EA2(start_pop, amount_generations)
new_pop_ev = evaluate(new_pop)

data_gatherer.add_header_to_stats()

# Track the end time
end_time = time.time()

print(f"Best solution found at generation: {data_gatherer.best_gen}")

# Calculate and print total execution time
execution_time = end_time - start_time
print(f"Execution Time: {execution_time:.2f} seconds")

print("Evolution completed!")

env.state_to_log()  # Checks environment state


