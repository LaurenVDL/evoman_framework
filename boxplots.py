import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from evoman.environment import Environment
from demo_controller import player_controller

def find_best_individual(algorithm, enemy):
    best_fitness = -float('inf')
    best_gen = -1
    best_run = -1
    
    for run in range(1, 11):  # Assuming 10 runs
        stats_file = f'{algorithm}_enemy{enemy}/R{run}/stats.out'
        if os.path.exists(stats_file):
            data = np.loadtxt(stats_file, delimiter=',', skiprows=1)
            max_fitness = np.max(data[:, 3])  # Best fitness is in the 4th column
            if max_fitness > best_fitness:
                best_fitness = max_fitness
                best_gen = np.argmax(data[:, 3])
                best_run = run
    
    return best_run, best_gen, best_fitness

def get_best_weights(algorithm, enemy, run, gen):
    weights_file = f'{algorithm}_enemy{enemy}/R{run}/best/{gen}.out'
    return np.loadtxt(weights_file)

def run_simulation(env, individual, num_trials=5):
    results = []
    for _ in range(num_trials):
        f, p, e, t = env.play(pcont=individual)
        gain = p - e  # Calculate gain
        results.append(gain)
    return results

def create_boxplot(all_results):
    plt.figure(figsize=(15, 8))
    box_data = all_results
    positions = [1, 2, 4, 5, 7, 8]
    labels = ['EA1\nEnemy 1', 'EA2\nEnemy 1', 'EA1\nEnemy 5', 'EA2\nEnemy 5', 'EA1\nEnemy 6', 'EA2\nEnemy 6']
    
    bp = plt.boxplot(box_data, positions=positions, labels=labels, widths=0.6)
    
    # Color the boxes
    colors = ['lightblue', 'lightgreen'] * 3
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    
    plt.title('Gain Comparison for EA1 and EA2 across Enemies')
    plt.ylabel('Gain (Player Energy - Enemy Energy)')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Add vertical lines to separate enemy groups
    plt.axvline(x=3, color='gray', linestyle='--')
    plt.axvline(x=6, color='gray', linestyle='--')
    
    plt.savefig('boxplot_all_enemies.png')
    plt.close()

def perform_t_test(ea1_results, ea2_results):
    t_statistic, p_value = stats.ttest_ind(ea1_results, ea2_results)
    return t_statistic, p_value

def main():
    all_results = []
    
    for enemy in [1, 5, 6]:
        print(f"Processing enemy {enemy}")
        
        # Set up environment
        env = Environment(enemies=[enemy],
                          playermode="ai",
                          player_controller=player_controller(10),
                          enemymode="static",
                          level=2,
                          speed="fastest",
                          visuals=False)
        
        # Find best individual for EA1
        ea1_run, ea1_gen, ea1_fitness = find_best_individual('EA1', enemy)
        ea1_weights = get_best_weights('EA1', enemy, ea1_run, ea1_gen)
        
        # Find best individual for EA2
        ea2_run, ea2_gen, ea2_fitness = find_best_individual('EA2', enemy)
        ea2_weights = get_best_weights('EA2', enemy, ea2_run, ea2_gen)
        
        print(f"EA1 best: Run {ea1_run}, Gen {ea1_gen}, Fitness {ea1_fitness}")
        print(f"EA2 best: Run {ea2_run}, Gen {ea2_gen}, Fitness {ea2_fitness}")
        
        # Run best individuals 5 times
        ea1_results = run_simulation(env, ea1_weights)
        ea2_results = run_simulation(env, ea2_weights)
        
        all_results.extend([ea1_results, ea2_results])
        
        # Perform t-test
        t_statistic, p_value = perform_t_test(ea1_results, ea2_results)
        print(f"T-test results for enemy {enemy}:")
        print(f"t-statistic: {t_statistic}")
        print(f"p-value: {p_value}")
        print()

    # Create boxplot for all enemies
    create_boxplot(all_results)
    print("Boxplot for all enemies created")

if __name__ == "__main__":
    main()


