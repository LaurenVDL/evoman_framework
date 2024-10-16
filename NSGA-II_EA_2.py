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
        
        os.makedirs(os.path.join(name, "best"), exist_ok=True)

    def gather(self, pop, pop_fit, gen):
        current_mean = np.mean(pop_fit)
        current_std = np.std(pop_fit)
        current_best = np.max(pop_fit)
        
        self.mean_fitness = np.concatenate([self.mean_fitness, [current_mean]])
        self.std_fitness = np.concatenate([self.std_fitness, [current_std]])
        self.best_fitness = np.concatenate([self.best_fitness, [current_best]])
        self.generations = np.concatenate([self.generations, [gen]])

        if current_best >= np.max(self.best_fitness):
            self.best_gen = gen

        self.stats = np.stack([self.generations, self.mean_fitness, self.std_fitness, self.best_fitness])
        np.savetxt(f"{self.name}/stats.out", self.stats.T, delimiter=',', fmt='%.6f')
        np.savetxt(f"{self.name}/best/{gen}.out", pop[np.argmax(pop_fit)], delimiter=',', fmt='%1.2e')

        solutions = [pop, pop_fit]
        # env.update_solutions(solutions)
        # env.save_state()

    def add_header_to_stats(self):
        header = "Generation,Mean_Fitness,Std_Fitness,Best_Fitness\n"
        with open(f"{self.name}/stats.out", 'r') as f:
            content = f.read()
        with open(f"{self.name}/stats.out", 'w') as f:
            f.write(header + content)

headless = True
if headless:
    os.environ["SDL_VIDEODRIVER"] = "dummy"

# if len(sys.argv) != 3:
#     print("Usage: python EA_1.py <experiment_name> <enemy_number>")
#     sys.exit(1)

# experiment_name = sys.argv[1]
# enemy_number = int(sys.argv[2])

enemies = [1,4,7]
experiment_name = 'NSGA-II_EA2_task2'
os.makedirs(experiment_name, exist_ok=True)

n_hidden_neurons = 10
# env = Environment(experiment_name=experiment_name,
#                   enemies=enemies,
#                   multiplemode="yes",
#                   playermode="ai",
#                   player_controller=player_controller(n_hidden_neurons),
#                   enemymode="static",
#                   level=2,
#                   speed="fastest",
#                   visuals=False)

# env.state_to_log()

start_time = time.time()

# n_vars = (env.get_num_sensors() + 1) * n_hidden_neurons + (n_hidden_neurons + 1) * 5
n_vars = 265
dom_u = 1
dom_l = -1
mu = 100
lambda_ = 200
gens = 30
mutation_rate = 0.25
n_points = 5
prob_c = 0.7

data_gatherer = DataGatherer(experiment_name)

def simulation(env, x):
    f, p, e, t = env.play(pcont=x)
    return f

# def evaluate(x):
#     return np.array(list(map(lambda y: simulation(env, y), x)))

# # Evaluation function for multiple objectives (enemies)
def evaluate_multiobjective(population):
# #     # fitnesses = []
# #     # for i in enemies:  # Assuming you want to evaluate against enemies 1 to 8
# #     #     env = Environment(experiment_name=experiment_name,
# #     #                       enemies=[i],  # Single enemy for each evaluation
# #     #                       playermode="ai",
# #     #                       player_controller=player_controller(n_hidden_neurons),
# #     #                       enemymode="static",
# #     #                       level=2,
# #     #                       speed="fastest",
# #     #                       visuals=False)
# #     #     f, p, e, t = env.play(pcont=population)
# #     #     # f = evaluate(population)
# #     #     fitnesses.append(f)  # Collect fitness for each enemy (multiple objectives)
# #     # return np.array(fitnesses).T
    enemy_fitnesses = np.zeros((len(population), len(enemies)))  # Store fitness for each enemy

    for i, enemy in enumerate(enemies):
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

        fitness = aggregate_fitness(enemy_fitnesses)
        solutions = [population, fitness]
        env.update_solutions(solutions)
        env.save_state()
        
    return enemy_fitnesses

# Aggregate fitness manually for single objective comparison
def aggregate_fitness(fitnesses):
    return np.mean(fitnesses, axis=1) - np.std(fitnesses, axis=1)

def crossover_n_point(pop):
    num_individuals, num_genes = pop.shape
    new_population = []
    
    for i in range(0, num_individuals, 2):
        parent1 = pop[i]
        parent2 = pop[i + 1]
        
        if random.random() < prob_c:
            crossover_points = sorted(random.sample(range(1, num_genes), n_points))
            crossover_points = [0] + crossover_points + [num_genes]
            
            segments1 = [parent1[crossover_points[j]:crossover_points[j + 1]] for j in range(len(crossover_points) - 1)]
            segments2 = [parent2[crossover_points[j]:crossover_points[j + 1]] for j in range(len(crossover_points) - 1)]
            
            child1_segments = [segments1[j] if random.random() < 0.5 else segments2[j] for j in range(len(segments1))]
            child2_segments = [segments2[j] if random.random() < 0.5 else segments1[j] for j in range(len(segments2))]
            
            child1 = np.concatenate(child1_segments)
            child2 = np.concatenate(child2_segments)
        else:
            child1 = np.copy(parent1)
            child2 = np.copy(parent2)
        
        new_population.append(child1)
        new_population.append(child2)
    
    return np.array(new_population)

def swap_mutation(individual, mutation_rate):
    if np.random.uniform(0, 1) <= mutation_rate:
        idx1, idx2 = np.random.randint(0, len(individual), 2)
        individual[idx1], individual[idx2] = individual[idx2], individual[idx1]
    return individual

def adaptive_mutation(individual, mutation_rate, generation, max_generations):
    adaptive_rate = mutation_rate * (1 + generation / max_generations)
    # print(f"Individual shape before mutation: {individual.shape}")
    return swap_mutation(individual, adaptive_rate)

def apply_limits(individual):
    return np.clip(individual, dom_l, dom_u)

# Fast Non-Dominated Sorting function for NSGA-II
def fast_non_dominated_sort(fitness):
    population_size = len(fitness)
    domination_count = np.zeros(population_size)
    dominated_solutions = [[] for _ in range(population_size)]
    rank = np.zeros(population_size)

    fronts = [[]]
    for p in range(population_size):
        for q in range(population_size):
            if np.all(fitness[p] >= fitness[q]) and np.any(fitness[p] > fitness[q]):
                dominated_solutions[p].append(q)
            elif np.all(fitness[q] >= fitness[p]) and np.any(fitness[q] > fitness[p]):
                domination_count[p] += 1
        
        if domination_count[p] == 0:
            rank[p] = 0
            fronts[0].append(p)
    
    i = 0
    while len(fronts[i]) > 0:
        next_front = []
        for p in fronts[i]:
            for q in dominated_solutions[p]:
                domination_count[q] -= 1
                if domination_count[q] == 0:
                    rank[q] = i + 1
                    next_front.append(q)
        i += 1
        fronts.append(next_front)
    
    return fronts[:-1]

def crowding_distance(fitness):
    distances = np.zeros(fitness.shape[0])
    for m in range(fitness.shape[1]):
        sorted_indices = np.argsort(fitness[:, m])
        min_value = fitness[sorted_indices[0], m]
        max_value = fitness[sorted_indices[-1], m]
        distances[sorted_indices[0]] = distances[sorted_indices[-1]] = np.inf
        for i in range(1, len(sorted_indices) - 1):
            distances[sorted_indices[i]] += (fitness[sorted_indices[i + 1], m] - fitness[sorted_indices[i - 1], m]) / (max_value - min_value + 1e-9)
    return distances

# Parent selection function using NSGA-II fast non-dominated sorting and crowding distance
def select_parents_nsga2(population, fitness):
    fronts = fast_non_dominated_sort(fitness)
    new_population = []
    for front in fronts:
        distances = crowding_distance(fitness[front])
        sorted_indices = np.argsort(distances)[::-1]
        new_population.extend(population[front][sorted_indices])
    return np.array(new_population[:mu])

# def evolve_population(population, generation):
#     new_population = []
#     fitness = evaluate(population)
#     for _ in range(lambda_ // 2):
#         parent1, parent2 = select_parents_nsga2(population, fitness)[:2]
#         offspring1, offspring2 = crossover_n_point(np.array([parent1, parent2]))
#         offspring1 = adaptive_mutation(offspring1, mutation_rate, generation, gens)
#         offspring2 = adaptive_mutation(offspring2, mutation_rate, generation, gens)
#         offspring1 = apply_limits(offspring1)
#         offspring2 = apply_limits(offspring2)
#         new_population.append(offspring1)
#         new_population.append(offspring2)
#     return np.array(new_population)

# # Initialize population
# population = np.random.uniform(dom_l, dom_u, (mu, n_vars))
# fitness = evaluate(population)

# # Evolution loop
# for generation in range(gens):

#     # Survival selection - NSGA-II

#     # offspring_population = evolve_population(population, fitness, generation)
#     # offspring_fitness = evaluate(offspring_population)
    
#     # combined_population = np.vstack((population, offspring_population))
#     # combined_fitness = np.concatenate((fitness, offspring_fitness))
    
#     # fronts = fast_non_dominated_sort(combined_fitness)
    
#     # new_population = []
#     # for front in fronts:
#     #     if len(new_population) + len(front) > mu:
#     #         distances = crowding_distance(combined_fitness[front])
#     #         sorted_front = np.argsort(distances)[::-1]
#     #         new_population.extend(combined_population[front][sorted_front][:mu - len(new_population)])
#     #         break
#     #     new_population.extend(combined_population[front])
    
#     # population = np.array(new_population)
#     # fitness = evaluate(population)

#     # Survival selection - COMMA 

#     # Generate offspring population using evolution process
#     offspring_population = evolve_population(population, fitness)
    
#     # Evaluate offspring population
#     offspring_fitness = evaluate(offspring_population)
    
#     # Apply comma strategy: survival selection only selects from offspring (λ individuals)
#     best_indices = np.argsort(offspring_fitness)[-mu:]  # Select the top mu individuals from offspring
#     population = offspring_population[best_indices]  # Replace parent population with selected offspring
#     fitness = offspring_fitness[best_indices]  # Update fitness for new population

#     # aggregate_fitness = aggregate_fitness(fitness)

#     # print(f'Generation {generation}, Best fitness: {aggregate_fitness}')
    
#     # print(f'Generation {generation}, Best fitness: {np.max(fitness)}')
#     data_gatherer.gather(population, fitness, generation)

# data_gatherer.add_header_to_stats()

def evolve_population(population, fitness, generation):
    new_population = []
    for _ in range(lambda_ // 2):
        parent1, parent2 = select_parents_nsga2(population, fitness)[:2]
        offspring1, offspring2 = crossover_n_point(np.array([parent1, parent2]))
        offspring1 = adaptive_mutation(offspring1, mutation_rate, generation, gens)
        offspring2 = adaptive_mutation(offspring2, mutation_rate, generation, gens)
        offspring1 = apply_limits(offspring1)
        offspring2 = apply_limits(offspring2)
        new_population.extend([offspring1, offspring2])
    return np.array(new_population)

# Initialize population
population = np.random.uniform(dom_l, dom_u, (mu, n_vars))
fitness = evaluate_multiobjective(population)

# Evolution loop
for generation in range(gens):
    offspring_population = evolve_population(population, fitness, generation)
    offspring_fitness = evaluate_multiobjective(offspring_population)
    
    combined_population = np.vstack((population, offspring_population))
    combined_fitness = np.vstack((fitness, offspring_fitness))
    
    fronts = fast_non_dominated_sort(combined_fitness)
    
    new_population = []
    new_fitness = []
    for front in fronts:
        if len(new_population) + len(front) > mu:
            distances = crowding_distance(combined_fitness[front])
            sorted_indices = np.argsort(distances)[::-1]
            selected = sorted_indices[:mu - len(new_population)]
            new_population.extend(combined_population[front][selected])
            new_fitness.extend(combined_fitness[front][selected])
            break
        new_population.extend(combined_population[front])
        new_fitness.extend(combined_fitness[front])
    
    population = np.array(new_population[:mu])
    fitness = np.array(new_fitness[:mu])

    # Calculate aggregate fitness for reporting
    agg_fitness = aggregate_fitness(fitness)
    
    print(f'Generation {generation}, Best aggregate fitness: {np.max(agg_fitness)}')
    data_gatherer.gather(population, agg_fitness, generation)

data_gatherer.add_header_to_stats()
end_time = time.time()

print(f"Best solution found at generation: {data_gatherer.best_gen}")
execution_time = end_time - start_time
print(f"Execution Time: {execution_time:.2f} seconds")
print("Evolution completed!")

# env.state_to_log()