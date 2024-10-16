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

# # # Check if the correct number of arguments is provided
# if len(sys.argv) != 3:
#     print("Usage: python EA_1.py <experiment_name> <enemy_number>")
#     sys.exit(1)

# # Get the arguments
# experiment_name = sys.argv[1]
# enemy_number = int(sys.argv[2])

enemy_list = [1, 4, 7]
experiment_name = 'EA_2_task2'
os.makedirs(experiment_name, exist_ok=True)

n_hidden_neurons = 10

# Initializes simulation in individual evolution mode, for multiple static enemies.
# env = Environment(experiment_name=experiment_name,
#                   enemies=enemy_list,
#                   multiplemode="yes",
#                   playermode="ai",
#                   player_controller=player_controller(n_hidden_neurons),
#                   enemymode="static",
#                   level=2,
#                   speed="fastest",
#                   visuals=False)
def evaluate_multiple_enemies(population):
    enemy_fitnesses = np.zeros((len(population), len(enemy_list)))  # Store fitness for each enemy

    for i, enemy in enumerate(enemy_list):
        # Create a new environment for each enemy in single mode
        env = Environment(experiment_name=experiment_name,
                          enemies=[enemy],
                          multiplemode="no",  # Single mode for each enemy
                          playermode="ai",
                          player_controller=player_controller(n_hidden_neurons),
                          enemymode="static",
                          level=2,
                          speed="fastest",
                          visuals=False)
        
        env.state_to_log()  # Checks environment state
        
        # Evaluate fitness against the current enemy
        for j, individual in enumerate(population):
            fitness, p_life, e_life, time = env.play(pcont=individual)
            enemy_fitnesses[j, i] = fitness

    return enemy_fitnesses

# env.state_to_log()  # Checks environment state

# Track execution time
start_time = time.time()

# Genetic algorithm parameters
#n_vars = (env.get_num_sensors() + 1) * n_hidden_neurons + (n_hidden_neurons + 1) * 5
n_vars = 265
dom_u = 1
dom_l = -1
mu = 100  # Number of parents
lambda_ = 200  # Number of children
gens = 30
mutation_rate = 0.25
n_points = 5  # Number of crossover points
prob_c = 0.7  # Probability of crossover occurring

# Data Gatherer instance
data_gatherer = DataGatherer(experiment_name)

# Simulation function
def simulation(env, x):
    f, p, e, t = env.play(pcont=x)
    return f, p, e, t

# Aggregate fitness for final comparison (mean or other aggregation)
def aggregate_fitness(enemy_fitnesses):
    # Example: Mean fitness across enemies
    return enemy_fitnesses.mean(axis=1) - enemy_fitnesses.std(axis=1)

# # Evaluation function
# def evaluate(x):
#     results = np.array([simulation(env, y) for y in x])
#     # fitness = results[:, 0]
#     return results
#     # return np.array(list(map(lambda y: simulation(env, y), x)))
# Function to evaluate the population for multiple enemies using single mode

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
def swap_mutation(individual, mutation_rate):
    if np.random.uniform(0, 1) <= mutation_rate:
        idx1, idx2 = np.random.randint(0, len(individual), 2)
        individual[idx1], individual[idx2] = individual[idx2], individual[idx1]
    return individual

def adaptive_mutation(individual, mutation_rate, generation, max_generations):
    # Increase mutation rate as generations progress
    adaptive_rate = mutation_rate * (1 + generation / max_generations)
    return swap_mutation(individual, adaptive_rate)

# Apply limits to genes
def apply_limits(individual):
    return np.clip(individual, dom_l, dom_u)

# # Fast Non-Dominated Sorting function for NSGA-II
# def fast_non_dominated_sort(fitness):
#     population_size = len(fitness)
#     domination_count = np.zeros(population_size)
#     dominated_solutions = [[] for _ in range(population_size)]
#     rank = np.zeros(population_size)

#     fronts = [[]]
#     for p in range(population_size):
#         for q in range(population_size):
#             if np.all(fitness[p] >= fitness[q]) and np.any(fitness[p] > fitness[q]):
#                 dominated_solutions[p].append(q)
#             elif np.all(fitness[q] >= fitness[p]) and np.any(fitness[q] > fitness[p]):
#                 domination_count[p] += 1
        
#         if domination_count[p] == 0:
#             rank[p] = 0
#             fronts[0].append(p)
    
#     i = 0
#     while len(fronts[i]) > 0:
#         next_front = []
#         for p in fronts[i]:
#             for q in dominated_solutions[p]:
#                 domination_count[q] -= 1
#                 if domination_count[q] == 0:
#                     rank[q] = i + 1
#                     next_front.append(q)
#         i += 1
#         fronts.append(next_front)
    
#     return fronts[:-1]

# def crowding_distance(fitness):
#     distances = np.zeros(fitness.shape[0])
#     for m in range(fitness.shape[1]):
#         sorted_indices = np.argsort(fitness[:, m])
#         min_value = fitness[sorted_indices[0], m]
#         max_value = fitness[sorted_indices[-1], m]
#         distances[sorted_indices[0]] = distances[sorted_indices[-1]] = np.inf
#         for i in range(1, len(sorted_indices) - 1):
#             distances[sorted_indices[i]] += (fitness[sorted_indices[i + 1], m] - fitness[sorted_indices[i - 1], m]) / (max_value - min_value + 1e-9)
#     return distances

# # Parent selection function using NSGA-II fast non-dominated sorting and crowding distance
# def select_parents_nsga2(population, fitness):
#     fronts = fast_non_dominated_sort(fitness)
#     new_population = []
#     print(f"Fitness shape: {fitness.shape}")
#     for front in fronts:
#         distances = crowding_distance(fitness[front])
#         sorted_indices = np.argsort(distances)[::-1]
#         new_population.extend(population[front][sorted_indices])
#     return np.array(new_population[:mu])

# # Main evolution function
# def evolve_population(population, fitness):
#     new_population = []
#     for _ in range(lambda_ // 2):       # Generating λ offspring from μ parents
#         # Select parents using NSGA-II
#         parent1, parent2 = select_parents_nsga2(population, fitness)[:2]
        
#         # Perform n-point crossover
#         offspring1, offspring2 = crossover_n_point(np.array([parent1, parent2]))
        
#         # Apply mutation (adaptive mutation)
#         offspring1 = adaptive_mutation(offspring1, mutation_rate, generation, gens)
#         offspring2 = adaptive_mutation(offspring2, mutation_rate, generation, gens)
        
#         # Apply limits to offspring
#         offspring1 = apply_limits(offspring1)
#         offspring2 = apply_limits(offspring2)
        
#         new_population.append(offspring1)
#         new_population.append(offspring2)
    
#     return np.array(new_population)

# Fast Non-Dominated Sorting
def fast_non_dominated_sort(fitness):
    S = [[] for _ in range(len(fitness))]
    n = [0] * len(fitness)
    rank = [0] * len(fitness)
    fronts = [[]]

    for p, p_fitness in enumerate(fitness):
        for q, q_fitness in enumerate(fitness):
            # if all(pf >= qf for pf, qf in zip(p_fitness, q_fitness)) and any(pf > qf for pf, qf in zip(p_fitness, q_fitness)):
            if np.all(p_fitness >= q_fitness) and np.any(p_fitness > q_fitness):
                S[p].append(q)
            # elif all(qf >= pf for pf, qf in zip(p_fitness, q_fitness)) and any(qf > pf for pf, qf in zip(p_fitness, q_fitness)):
            elif np.all(q_fitness >= p_fitness) and np.any(q_fitness > p_fitness):
                n[p] += 1
        
        if n[p] == 0:
            rank[p] = 0
            fronts[0].append(p)

    i = 0
    while fronts[i]:
        next_front = []
        for p in fronts[i]:
            for q in S[p]:
                n[q] -= 1
                if n[q] == 0:
                    rank[q] = i + 1
                    next_front.append(q)
        i += 1
        fronts.append(next_front)

    return fronts[:-1]

# Crowding Distance
def crowding_distance(fitness, front):
    distances = [0] * len(front)
    for obj in range(len(fitness[0])):
        sorted_front = sorted(front, key=lambda x: fitness[x][obj])
        distances[0] = distances[-1] = float('inf')
        for i in range(1, len(front) - 1):
            distances[i] += (fitness[sorted_front[i+1]][obj] - fitness[sorted_front[i-1]][obj]) / (fitness[sorted_front[-1]][obj] - fitness[sorted_front[0]][obj] + 1e-15)
    return distances

# NSGA-II Selection
def nsga2_selection(population, fitness, k):
    fronts = fast_non_dominated_sort(fitness)
    solutions = []
    for front in fronts:
        if len(solutions) + len(front) <= k:
            solutions.extend(front)
        else:
            distances = crowding_distance(fitness, front)
            sorted_front = [x for _, x in sorted(zip(distances, front), reverse=True)]
            solutions.extend(sorted_front[:k - len(solutions)])
            break
    return [population[i] for i in solutions]

# Main evolution function
def evolve_population(population, fitness):
    offspring = []
    while len(offspring) < lambda_:
        parent1, parent2 = random.sample(nsga2_selection(population, fitness, mu), 2)
        if random.random() < prob_c:
            child1, child2 = crossover_n_point(np.array([parent1, parent2]))
        else:
            child1, child2 = parent1.copy(), parent2.copy()
        
        child1 = adaptive_mutation(child1, mutation_rate, generation, gens)
        child2 = adaptive_mutation(child2, mutation_rate, generation, gens)
        
        child1 = apply_limits(child1)
        child2 = apply_limits(child2)
        
        offspring.extend([child1, child2])
    
    return np.array(offspring[:lambda_])

# Initialize population
population = np.random.uniform(dom_l, dom_u, (mu, n_vars))
fitness = evaluate_multiple_enemies(population)


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

    print(f"Fitness shape: {fitness[:, 0].shape}")

    print(f'Generation {generation}, Best fitness: {np.max(fitness[:, 0])}')

    # Gather data
    data_gatherer.gather(population, fitness[:, 0], generation)

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

# env.state_to_log()  # Checks environment state
