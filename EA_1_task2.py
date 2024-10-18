import sys
from evoman.environment import Environment
from demo_controller import player_controller
import time
import numpy as np
import os
import random

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

# # Check if the correct number of arguments is provided
if len(sys.argv) != 3:
    print("Usage: python EA_1.py <experiment_name> <enemy_number>")
    sys.exit(1)

# # Get the arguments
experiment_name = sys.argv[1]
enemies = sys.argv[2].split(',')  # Split the string into a list

# Convert the list of strings to integers
enemies = list(map(int, enemies))  # Convert to integers

# enemies = [2,3,4]
# experiment_name = 'EA_1_task2'
os.makedirs(experiment_name, exist_ok=True)

n_hidden_neurons = 10

# Initializes simulation in individual evolution mode, for single static enemy.
env = Environment(experiment_name=experiment_name,
                  enemies=enemies,
                  multiplemode="yes",
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
#npop = 100
mu = 100  # Number of parents
lambda_ = 200  # Number of children
gens = 30
mutation_rate = 0.2    # 0.25 0.5 0.3 0.2 0.2 0.2 0.2 0.25 0.2
n_points = 15          # 5 10 15 15 20 10 5 5 15       # Number of crossover points
prob_c = 0.8           # 0.7 0.8 0.6 0.8 0.9 0.7 0.7 0.7 0.8        # Probability of crossover occurring
                      # 74 73.5 56.09 78.06 72.4 60.9 69.7 87.0 88.2
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

# # Swap mutation
def swap_mutation(individual, mutation_rate):
    if np.random.uniform(0, 1) <= mutation_rate:
        idx1, idx2 = np.random.randint(0, len(individual), 2)
        individual[idx1], individual[idx2] = individual[idx2], individual[idx1]
    return individual

# def gaussian_mutation(individual, mutation_rate, scale=0.1):
#     for i in range(len(individual)):
#         if np.random.random() < mutation_rate:
#             individual[i] += np.random.normal(0, scale)
#     return np.clip(individual, dom_l, dom_u)


def adaptive_mutation(individual, mutation_rate, generation, max_generations):
    # Increase mutation rate as generations progress
    adaptive_rate = mutation_rate * (1 + generation / max_generations)
    return swap_mutation(individual, adaptive_rate)

# Tournament selection
def tournament_selection(pop, fitness):
    i1, i2 = np.random.randint(0, len(pop), 2)
    return pop[i1] if fitness[i1] > fitness[i2] else pop[i2]

# Apply limits to genes
def apply_limits(individual):
    return np.clip(individual, dom_l, dom_u)

# Main evolution function
def evolve_population(population, fitness):
    new_population = []
    for _ in range(lambda_ // 2):       # Generating λ offspring from μ parents
        # Select parents
        parent1 = tournament_selection(population, fitness)
        parent2 = tournament_selection(population, fitness)
        
        # Perform n-point crossover
        offspring1, offspring2 = crossover_n_point(np.array([parent1, parent2]))
        
        # Apply mutation (swap mutation)
        offspring1 = adaptive_mutation(offspring1, mutation_rate, generation, gens)
        offspring2 = adaptive_mutation(offspring2, mutation_rate, generation, gens)
        
        # Apply limits to offspring
        offspring1 = apply_limits(offspring1)
        offspring2 = apply_limits(offspring2)
        
        new_population.append(offspring1)
        new_population.append(offspring2)
    
    return np.array(new_population)


# Initialize population
population = np.random.uniform(dom_l, dom_u, (mu, n_vars))
fitness = evaluate(population)


# Evolution loop
for generation in range(gens):

    # Generate offspring population
    offspring_population = evolve_population(population, fitness)
    
    # Evaluate offspring
    offspring_fitness = evaluate(offspring_population)
    
    # Select the best μ individuals from the offspring (Comma strategy) - survival selection
    best_indices = np.argsort(offspring_fitness)[-mu:]
    population = offspring_population[best_indices]
    fitness = offspring_fitness[best_indices]

    print(f'Generation {generation}, Best fitness: {np.max(fitness)}')

    # Gather data
    data_gatherer.gather(population, fitness, generation)

# After all generations are complete
data_gatherer.add_header_to_stats()

# Track the end time
end_time = time.time()

# Log the generation where the best solution was found
print(f"Best solution found at generation: {data_gatherer.best_gen}")

# Calculate and print total execution time
execution_time = end_time - start_time
print(f"Execution Time: {execution_time:.2f} seconds")

print("Evolution completed!")

env.state_to_log()  # Checks environment state
