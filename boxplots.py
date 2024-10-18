import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from evoman.environment import Environment
from demo_controller import player_controller
import random

headless = True
if headless:
    os.environ["SDL_VIDEODRIVER"] = "dummy"

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
    for i in range(num_trials):
        # Set a new random seed for each trial
        #env.state_to_log() # Reset the environment
        # random_seed = random.randint(1, 1000000)
        # env.update_parameter('random_seed', random_seed)

        f, p, e, t = env.play(pcont=individual)
        gain = p - e  # Calculate gain
        results.append(gain)
        print(f"Player Energy: {p}, Enemy Energy: {e}")
        print("Gain: ", p-e)
        env.update_solutions([individual,f])
        env.save_state()
        #env.state_to_log()
        # print(f"  Random Seed: {random_seed}")
        
    return results

def create_boxplot(all_results):
    fig, ax = plt.subplots(figsize=(12, 6))
    
    #labels = ['EA1 E1', 'EA2 E1', 'EA1 E5', 'EA2 E5', 'EA1 E6', 'EA2 E6']
    labels = ['EA1 G1', 'EA2 G2', 'EA1 G2', 'EA2 G2']               # group of enemies
    
    # Calculate means for each set of results
    means = [np.mean(results) for results in all_results]
    
    # Create boxplot
    bp = ax.boxplot(all_results, patch_artist=True, labels=labels)
    
    # Customize boxplot colors
    colors = ['#FF1493', '#00BFFF'] * 3  # Deep Pink for EA1, Deep Sky Blue for EA2
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)  # Add some transparency
        patch.set_edgecolor(color)
    
    # Set median lines to dark black
    for median in bp['medians']:
        median.set_color('black')
        median.set_linewidth(2)
    
    # Customize whiskers and caps
    for whisker in bp['whiskers']:
        whisker.set_color('black')
        whisker.set_linestyle('--')
    for cap in bp['caps']:
        cap.set_color('black')
    
    # Add mean points
    mean_points = ax.scatter(range(1, len(all_results) + 1), means, color='yellow', s=50, zorder=3)
    
    # Set title with increased size and bold font
    ax.set_title('Mean individual gain of best performing individual in EA1 and EA2', 
                 fontsize=16, fontweight='bold')
    
    # Increase size of x and y labels
    ax.set_xlabel('Experiment name', fontsize=13)
    ax.set_ylabel('Individual gain', fontsize=13)
    
    # Adjust y-axis to prevent compression
    ax.set_ylim(min(min(result) for result in all_results) - 10, 
                max(max(result) for result in all_results) + 10)
    
    # Add legend for EA1 and EA2 with increased font size
    ea1_patch = plt.Rectangle((0, 0), 1, 1, fc='#FF1493', alpha=0.7, edgecolor='#FF1493')
    ea2_patch = plt.Rectangle((0, 0), 1, 1, fc='#00BFFF', alpha=0.7, edgecolor='#00BFFF')
    ax.legend([ea1_patch, ea2_patch, mean_points], ['EA1', 'EA2', 'Mean'], 
              loc='upper right', bbox_to_anchor=(1, 1), fontsize=12)
    
    # Increase tick label font size
    ax.tick_params(axis='both', which='major', labelsize=12)
    
    plt.tight_layout()
    plt.savefig('boxplot_comparison.png', dpi=300)
    plt.close()

def main():
    all_results = []
    
    for enemy in [1, 5, 6]:
        print(f"\nProcessing enemy {enemy}")
        
        # Set up environment
        env = Environment(enemies=[enemy],
                          playermode="ai",
                          player_controller=player_controller(10),
                          enemymode="static",
                          level=2,
                          speed="fastest",
                          randomini="yes",
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

        print("\nEA1 Results:")
        print(ea1_results)
        print("\nEA2 Results:")
        print(ea2_results)

        all_results.extend([ea1_results, ea2_results])

        # Perform Mann-Whitney U test
        statistic, p_value = stats.mannwhitneyu(ea1_results, ea2_results, alternative='two-sided')
        print(f"Mann-Whitney U test results for enemy {enemy}:")
        print(f"Statistic: {statistic}")
        print(f"p-value: {p_value}")
        print()

    # Create boxplot for all enemies
    create_boxplot(all_results)
    print("Boxplot for all enemies created")

    for i, enemy in enumerate([1, 5, 6]):
        ea1_index = i * 2
        ea2_index = i * 2 + 1
        print(f"Enemy {enemy}:")
        print(f"EA1 - Mean: {np.mean(all_results[ea1_index]):.6f}, Std: {np.std(all_results[ea1_index]):.6f}")
        print(f"EA2 - Mean: {np.mean(all_results[ea2_index]):.6f}, Std: {np.std(all_results[ea2_index]):.6f}")
        print()

if __name__ == "__main__":
    main()