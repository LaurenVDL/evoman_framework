import sys
from evoman.environment import Environment
from demo_controller import player_controller
import time
import numpy as np
import os
import random
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

# DataGatherer class to gather data for plotting later
class DataGatherer:
    def __init__(self, name):
        self.name = name
        self.mean_fitness = np.array([])
        self.std_fitness = np.array([])
        self.best_fitness = np.array([])
        self.generations = np.array([])
        self.best_gen = -1  # Generation where best solution is found
        self.stats = []
        
        # Create main directory and 'best' subdirectory
        os.makedirs(os.path.join(name, "best"), exist_ok=True)

    def gather(self, pop, pop_fit, gen):
        current_mean = np.mean(pop_fit)
        current_std = np.std(pop_fit)
        current_best = np.max(pop_fit)
        
        self.mean_fitness = np.concatenate([self.mean_fitness, [current_mean]])
        self.std_fitness = np.concatenate([self.std_fitness, [current_std]])
        self.best_fitness = np.concatenate([self.best_fitness, [current_best]])
        self.generations = np.concatenate([self.generations, [gen]])

        # Update the generation with the best solution if the new best fitness is higher
        if current_best >= np.max(self.best_fitness):
            self.best_gen = gen

        # Stack the gathered data
        self.stats = np.stack([self.generations, self.mean_fitness, self.std_fitness, self.best_fitness])

        # Save stats without header
        np.savetxt(f"{self.name}/stats.out", self.stats.T, delimiter=',', fmt='%.6f')

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

# Set headless mode for faster simulation
headless = True
if headless:
    os.environ["SDL_VIDEODRIVER"] = "dummy"

experiment_name = 'EA_2'
os.makedirs(experiment_name, exist_ok=True)

n_hidden_neurons = 10

# Initializes simulation in individual evolution mode, for single static enemy.
env = Environment(experiment_name=experiment_name,
                  enemies=[8],
                  playermode="ai",
                  player_controller=player_controller(n_hidden_neurons),
                  enemymode="static",
                  level=2,
                  speed="fastest",
                  visuals=False)

env.state_to_log()  # Checks environment state

# Track execution time
start_time = time.time()

# Genetic algorithm parameters
n_vars = (env.get_num_sensors() + 1) * n_hidden_neurons + (n_hidden_neurons + 1) * 5
dom_u = 1
dom_l = -1
npop = 100  # Population size
mu = 100  # Number of parents
lambda_ = 200  # Number of children
gens = 30  # Number of generations
mutation_rate = 0.2
n_points = 5  # Number of crossover points
prob_c = 0.7  # Probability of crossover occurring 


# Data Gatherer instance
data_gatherer = DataGatherer(experiment_name)

# Simulation function
def simulation(env, x):
    f, p, e, t = env.play(pcont=x)
    return f

# Evaluation function
def evaluate(x):
    return np.array(list(map(lambda y: simulation(env, y), x)))

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

# Swap mutation
def swap_mutation(individual):
    if np.random.uniform(0, 1) <= mutation_rate:
        idx1, idx2 = np.random.randint(0, len(individual), 2)
        individual[idx1], individual[idx2] = individual[idx2], individual[idx1]
    return individual

# Tournament selection
def tournament_selection(pop, fitness):
    i1, i2 = np.random.randint(0, len(pop), 2)
    return pop[i1] if fitness[i1] > fitness[i2] else pop[i2]

# Apply limits to genes
def apply_limits(individual):
    return np.clip(individual, dom_l, dom_u)

# DBSCAN clustering for survival selection
def dbscan_survival_selection(population, fitness, eps=0.5, min_samples=5):
    # Combine population and fitness for clustering
    data = np.column_stack((population, fitness.reshape(-1, 1)))
    
    # Normalize the data
    scaler = StandardScaler()
    data_normalized = scaler.fit_transform(data)
    
    # Perform DBSCAN clustering
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    cluster_labels = dbscan.fit_predict(data_normalized)
    
    # Calculate the mean fitness for each cluster
    unique_labels = set(cluster_labels)
    cluster_mean_fitness = {}
    for label in unique_labels:
        if label != -1:  # Ignore noise points
            cluster_mean_fitness[label] = np.mean(fitness[cluster_labels == label])
    
    # Sort clusters by mean fitness
    sorted_clusters = sorted(cluster_mean_fitness.items(), key=lambda x: x[1], reverse=True)
    
    # Select individuals from top clusters
    selected_population = []
    selected_fitness = []
    for label, _ in sorted_clusters:
        cluster_individuals = population[cluster_labels == label]
        cluster_fitness = fitness[cluster_labels == label]
        
        # Sort individuals within cluster by fitness
        sorted_indices = np.argsort(cluster_fitness)[::-1]
        
        selected_population.extend(cluster_individuals[sorted_indices])
        selected_fitness.extend(cluster_fitness[sorted_indices])
        
        if len(selected_population) >= mu:
            break
    
    # Trim to exactly mu individuals if we've selected more
    selected_population = selected_population[:mu]
    selected_fitness = selected_fitness[:mu]
    
    return np.array(selected_population), np.array(selected_fitness)

# Main evolution function
def evolve_population(population, fitness):
    offspring_population = []
    for _ in range(lambda_ // 2):
        # Select parents
        parent1 = tournament_selection(population, fitness)
        parent2 = tournament_selection(population, fitness)
        
        # Perform n-point crossover
        offspring1, offspring2 = crossover_n_point(np.array([parent1, parent2]))
        
        # Apply mutation (swap mutation)
        offspring1 = swap_mutation(offspring1)
        offspring2 = swap_mutation(offspring2)
        
        # Apply limits to offspring
        offspring1 = apply_limits(offspring1)
        offspring2 = apply_limits(offspring2)
        
        offspring_population.extend([offspring1, offspring2])
    
    return np.array(offspring_population)

# Initialize population
population = np.random.uniform(dom_l, dom_u, (npop, n_vars))
fitness = evaluate(population)

# Evolution loop
for generation in range(gens):
    print(f'Generation {generation}, Best fitness: {np.max(fitness)}')

    # Generate offspring population
    offspring_population = evolve_population(population, fitness)
    
    # Evaluate offspring
    offspring_fitness = evaluate(offspring_population)
    
    # Combine parent and offspring populations
    combined_population = np.vstack((population, offspring_population))
    combined_fitness = np.concatenate((fitness, offspring_fitness))
    
    # Use DBSCAN for survival selection
    population, fitness = dbscan_survival_selection(combined_population, combined_fitness)

    # Gather data (assuming you have a data_gatherer object)
    data_gatherer.gather(population, fitness, generation)

# After all generations are complete
data_gatherer.add_header_to_stats()

# Track the end time
end_time = time.time()

print(f"Best solution found at generation: {data_gatherer.best_gen}")

# Calculate and print total execution time
execution_time = end_time - start_time
print(f"Execution Time: {execution_time:.2f} seconds")

print("Evolution completed!")

env.state_to_log()  # Checks environment state
