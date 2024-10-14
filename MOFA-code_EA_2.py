# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 13:44:36 2024

@author: charl
"""

import os
os.chdir('C:\\Users\\charl\\OneDrive\\Documents\\GitHub\\evoman_framework')


import sys
from evoman.environment import Environment
from demo_controller import player_controller
import time
import numpy as np
import matplotlib.pyplot as plt


import random



# # DataGatherer class to gather data for plotting later
# class DataGatherer:
#     def _init_(self, name):
#         self.name = name
#         self.mean_fitness = np.array([])  # Now this will store multi-objective values
#         self.std_fitness = np.array([])
#         self.best_fitness = np.array([])
#         self.generations = np.array([])
#         self.best_gen = -1  # Generation where best solution is found
#         self.stats = []
        
#         # Create main directory and 'best' subdirectory
#         os.makedirs(os.path.join(name, "best"), exist_ok=True)

#     def gather(self, pop, pop_fit, gen):
#         #With axis=0: You track how the population performs against each enemy (objective) individually
#         current_mean = np.mean(pop_fit, axis=0)  # Mean fitness per enemy (per objective)
#         current_std = np.std(pop_fit, axis=0)    # Standard deviation per enemy (per objective)
#         current_best = np.max(pop_fit, axis=0)   # Best fitness per enemy (per objective)

#         #Without axis=0: You aggregate the performance across all enemies and track only a single value per generation
#         #current_mean = np.mean(np.mean(pop_fit, axis=1))  # Average fitness across all enemies and all individuals
#         #current_std = np.std(np.mean(pop_fit, axis=1))    # Standard deviation of the mean fitness across all enemies
#         #current_best = np.max(np.mean(pop_fit, axis=1))   # Best individual performance across enemies

        
#         self.mean_fitness = np.concatenate([self.mean_fitness, [current_mean]])
#         self.std_fitness = np.concatenate([self.std_fitness, [current_std]])
#         self.best_fitness = np.concatenate([self.best_fitness, [current_best]])
#         self.generations = np.concatenate([self.generations, [gen]])

#         # Update the generation with the best solution if new best is found
#         if current_best.max() >= np.max(self.best_fitness):
#             self.best_gen = gen

#         # Stack the gathered data
#         self.stats = np.stack([self.generations, self.mean_fitness, self.std_fitness, self.best_fitness])

#         # Save stats without header
#         np.savetxt(f"{self.name}/stats.out", self.stats.T, delimiter=',', fmt='%.6f')

#         # Save best solution
#         np.savetxt(f"{self.name}/best/{gen}.out", pop[np.argmax(pop_fit[:,0])], delimiter=',', fmt='%1.2e')

#         # Save the simulation state for future evaluation
#         solutions = [pop, pop_fit]
#         env.update_solutions(solutions)
#         env.save_state()

#     def add_header_to_stats(self):
#         header = "Generation,Mean_Fitness,Std_Fitness,Best_Fitness\n"
        
#         # Read existing content
#         with open(f"{self.name}/stats.out", 'r') as f:
#             content = f.read()
        
#         # Write header and content
#         with open(f"{self.name}/stats.out", 'w') as f:
#             f.write(header + content)


# Set headless mode for faster simulation
headless = True
if headless:
    os.environ["SDL_VIDEODRIVER"] = "dummy"

# # Check if the correct number of arguments is provided
# if len(sys.argv) != 2:
#     print("Usage: python Task2EA_2.py <experiment_name>")
#     sys.exit(1)


# # Get the arguments
# experiment_name = sys.argv[1]

# os.makedirs(experiment_name, exist_ok=True)

n_hidden_neurons = 10
experiment_name = 'dummy_demo'
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)


# Initializes simulation in individual evolution mode, for multiple static enemies.
env = Environment(experiment_name=experiment_name,
                    enemies=[1],  # Multiple enemies
                    playermode="ai",
                    player_controller=player_controller(n_hidden_neurons),
                    enemymode="static",
                    level=2,
                    speed="fastest",
                    visuals=False)  # 8 enemies



# for env in envs:
#     env.state_to_log()  # Checks environment state

# Track execution time
start_time = time.time()

# Genetic algorithm parameters
n_vars = (env.get_num_sensors()+1)*n_hidden_neurons + (n_hidden_neurons+1)*5

dom_u = 1
dom_l = -1
mu = 100  # Number of parents
lambda_ = 200  # Number of children
gens = 30
mutation_rate = 0.25
n_points = 5  # Number of crossover points

prob_c = 0.7  # Probability of crossover occurring
amount_enemies=2

population = np.random.uniform(dom_l, dom_u, (mu, n_vars))


# Data Gatherer instance
#data_gatherer = DataGatherer(experiment_name)

# Simulation function
def simulation(env, x):
    f, p, e, t = env.play(pcont=x)
    return f


# evaluation
def evaluate(x,env):
    return np.array(list(map(lambda y: simulation(env,y), x)))


# Evaluation function for multiple objectives (enemies)
def evaluate_multiobjective(population):
    fitnesses = []
    for i in range(1,9):
        env = Environment(experiment_name=experiment_name,
                            enemies=[i],  # Multiple enemies
                            playermode="ai",
                            player_controller=player_controller(n_hidden_neurons),
                            enemymode="static",
                            level=2,
                            speed="fastest",
                            visuals=False)
    
        f = evaluate(population, env)
        fitnesses.append(f)  # Collect fitness for each enemy (multiple objectives)
    return np.array(fitnesses).T


#bla = evaluate_multiobjective(population)
# Dominance function for non-dominated sorting


# Function to compute a Pareto front based on the highest y values
#THIS IS 2 DIMENSIONAL:
def compute_front_based_on_highest_y(fitness_array):
    # Sort by y-values (second column) in descending order
    sorted_fitness = fitness_array[np.argsort(-fitness_array[:, 1])]
    
    front = []
    last_added = sorted_fitness[0]  # Start with the highest y-value
    front.append(last_added)

    # Compare x-values with the last added point in the front
    for i in range(1, len(sorted_fitness)):
        current_point = sorted_fitness[i]
        if current_point[0] >= last_added[0]:  # Check if current x is greater than last added x
            front.append(current_point)
            last_added = current_point  # Update the last added point

    return np.array(front)

#MORE THEN 2 Dimensions 


def compute_front_based_on_one_value_higher(fitness_array):
    # Sort by the first objective (can be modified to sort by other objectives if needed)
    sorted_fitness = fitness_array[np.argsort(-fitness_array[:, 0])]  # Sorting by the first dimension (desc)

    front = []
    last_added = sorted_fitness[0]  # Start with the highest value in the first dimension
    front.append(last_added)

    # Compare all dimensions with the last added point in the front
    for current_point in sorted_fitness[1:]:
        # Check if current point is not dominated by last added
        if np.all(current_point >= last_added) and np.any(current_point > last_added):  
            front.append(current_point)
            last_added = current_point  # Update the last added point

    return np.array(front)





# Function to iteratively compute all Pareto fronts
def compute_all_pareto_fronts(fitness_array):
    fronts = []
    remaining_fitness = fitness_array

    # Keep computing fronts until no solutions are left
    while len(remaining_fitness) > 0:
        current_front = compute_front_based_on_one_value_higher(remaining_fitness)
        fronts.append(current_front)

        # Remove the current front from the remaining solutions
        remaining_fitness = np.array([sol for sol in remaining_fitness if sol.tolist() not in current_front.tolist()])
    
    return fronts


# fitness = evaluate_multiobjective(population)
# fronts = compute_all_pareto_fronts(fitness)

# front1 = fronts[0]
# front2 = fronts[1]


# # Create a 3D scatter plot
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')

# # Plot the first front in red
# ax.scatter(front1[:, 0], front1[:, 1], front1[:, 2], c='r', label='Front 1', marker='o')

# # Plot the second front in blue
# ax.scatter(front2[:, 0], front2[:, 1], front2[:, 2], c='b', label='Front 2', marker='^')

# # Add labels
# ax.set_xlabel('X-axis')
# ax.set_ylabel('Y-axis')
# ax.set_zlabel('Z-axis')
# ax.set_title('3D Pareto Fronts')
# ax.legend()

# # Show the plot
# plt.show()








# Crowding distance calculation for diversity preservation
def calculate_crowding_distance(front, population_fitness):
    distances = np.zeros(len(front))
    for i in range(population_fitness.shape[1]):  # For each objective
        sorted_indices = np.argsort(population_fitness[front, i])
        max_fitness = population_fitness[front, i].max()
        min_fitness = population_fitness[front, i].min()
        
        distances[sorted_indices[0]] = distances[sorted_indices[-1]] = float('inf')
        for j in range(1, len(front) - 1):
            distances[sorted_indices[j]] += (population_fitness[front, i][sorted_indices[j + 1]] -
                                             population_fitness[front, i][sorted_indices[j - 1]]) / (max_fitness - min_fitness)
    return distances


def find_corresponding_indices(fitness, parents):
    corresponding_indices = []
    
    for parent in parents:
        # Find the index of the row in fitness that matches the parent
        index = np.where((fitness == parent).all(axis=1))[0]
        if index.size > 0:
            corresponding_indices.append(index[0])  # Get the first matching index

    return corresponding_indices


# valid_fronts = []
# total_items = 0
# fitness = evaluate_multiobjective(population)

# fronts = compute_all_pareto_fronts(fitness)

# # Iterate through the fronts and collect them until the total exceeds 20
# for front in fronts:
    
#     if total_items + len(front) > 20:
#         break
#     valid_fronts.append(front)
#     total_items += len(front)

def select_parents_nsga2(population, fitness, fronts):
    # Assign weights to each front based on their rank
    valid_fronts = []
    total_items = 0

    # Iterate through the fronts and collect them until the total exceeds 20
    for front in fronts:

        valid_fronts.append(front)
        total_items += len(front)
        if total_items + len(front) >= 100:
            
            break
    # Assign weights to each valid front based on their rank
    front_weights = np.linspace(1, 0.1, len(valid_fronts))  # Higher weight for better fronts
    
    array = []
    # Select a front based on weights
    for i in range(2):
        selected_front_index = np.random.choice(len(valid_fronts), p=front_weights / front_weights.sum())
        selected_front = valid_fronts[selected_front_index]
        
        
        index_front = (np.random.choice(len(selected_front), size=1, replace=False))
        array.append(selected_front[index_front])
    array = np.array(array)
    index_parents = find_corresponding_indices(fitness, array)

    return  population[index_parents[0]], population[index_parents[1]]

# Apply limits to genes
def apply_limits(individual):
    return np.clip(individual, dom_l, dom_u)

# Adaptive mutation
def adaptive_mutation(individual, mutation_rate, generation, max_generations):
    adaptive_rate = mutation_rate * (1 + generation /  max_generations)
    idx1, idx2 = np.random.randint(0, len(individual), 2)
    individual[idx1], individual[idx2] = individual[idx2], individual[idx1]
    return apply_limits(individual)

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


# Evolution loop with NSGA-II for crossover and selection
def evolve_population_nsga2(population, generation, uitslag):
    offspring_population = []
    fitness = evaluate_multiobjective(population)
    
        
    # Calculate the average for each row
    averages = np.mean(fitness, axis=1)
    
    # Find the index of the highest average
    highest_average_index = np.argmax(averages)
    
    uitslag.append(fitness[highest_average_index])
    # Get the highest average value
    highest_average = averages[highest_average_index]
    
    # Print the highest average
    print("Highest Average:", highest_average)
    
    # Print the whole row corresponding to the highest average
    print("Row with Highest Average:", fitness[highest_average_index])
    
    
    
    
    fronts = compute_all_pareto_fronts(fitness)
    parents = []
    for _ in range(lambda_ // 2):
        # Select parents using NSGA-II
        parent1 , parent2= select_parents_nsga2(population, fitness, fronts)
        parents.append(parent1)
        offspring1, offspring2 = crossover_n_point(np.array([parent1, parent2]))
        
        # Apply mutation (swap mutation)
        offspring1 = adaptive_mutation(offspring1, mutation_rate, generation, gens)
        offspring2 = adaptive_mutation(offspring2, mutation_rate, generation, gens)
        
        # Apply limits to offspring
        offspring1 = apply_limits(offspring1)
        offspring2 = apply_limits(offspring2)
        
        offspring_population.append(offspring1)
        offspring_population.append(offspring2)
    # Select next generation using NSGA-II
    return np.array(offspring_population), uitslag


uitslag = []

# Evolution loop
for generation in range(gens):
    population, uitslag = evolve_population_nsga2(population, generation,uitslag)
    
    # # Create scatter plot
    # plt.figure(figsize=(8, 6))
    # plt.scatter(fitness_levels[:, 0], fitness_levels[:, 1], color='blue', label='Fitness Levels')
    # plt.xlabel('f1(x)')  # x-axis label
    # plt.ylabel('f2(x)')  # y-axis label
    # plt.title('Scatter Plot of Fitness Levels')  # Plot title
    # plt.axhline(0, color='black', linewidth=0.5, linestyle='--')  # Add x-axis line
    # plt.axvline(0, color='black', linewidth=0.5, linestyle='--')  # Add y-axis line
    # plt.grid()  # Add grid
    # plt.xlim(-10, 100)  # Example limits, adjust as needed
    # plt.ylim(-10, 100)  # Example limits, adjust as needed
    
    # plt.legend()  # Show legend
    # plt.show()  # Display the plot

    # Generate offspring population

        
#     data_gatherer.gather(population, fitness, generation)

# # After all generations are complete
# data_gatherer.add_header_to_stats()

# # Track the end time
# end_time = time.time()

# # Log the generation where the best solution was found
# print(f"Best solution found at generation: {data_gatherer.best_gen}")

# # Calculate and print total execution time
# execution_time = end_time - start_time
# print(f"Execution Time: {execution_time:.2f} seconds")

# print("Evolution completed!")

# env.state_to_log()  # Checks  state