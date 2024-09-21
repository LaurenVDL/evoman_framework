
import sys, os

os.chdir('C:\\Users\\charl\\OneDrive\\Documents\\GitHub\\evoman_framework')


from evoman.environment import Environment
from demo_controller import player_controller, enemy_controller
import random
import numpy as np
import time


experiment_name = 'dummy_demo'
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)



n_hidden_neurons=10
npop = 100
prob_c = 0.3 #probability for doing the crossover
prob_m = 0.1 #probability if a mutation will accur in a individual
dom_u = 1
dom_l = -1
n_points = 100
max_swaps = 10


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


# runs simulation
def simulation(env,x):
    f,p,e,t = env.play(pcont=x)
    return f


# evaluation
def evaluate(x):
    return np.array(list(map(lambda y: simulation(env,y), x)))


#NOT FINISHED YET
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
                #individual[:, [column1, column2]] = individual[:, [column2, column1]]

        
    return mutated_pop




def crossover_n_point(pop):
    num_individuals, num_genes = pop.shape
    new_population = []
    
    
        
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
    
    return np.array(new_population)


start_time = time.time()
new_pop_n_point = crossover_n_point(pop)
end_time = time.time()




# def selection(pop):
#     #decide who goes through to the next generation
    
#     return new generation 
    