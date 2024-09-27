### LINEPLOTS
import matplotlib.pyplot as plt

def read_fitness_results(file_path):
    """
    Reads the fitness results from the given file.
    Assumes the format: `gen best mean std` on each line for generations.
    """
    generations = []
    mean_fitness = []
    max_fitness = []

    with open(file_path, 'r') as f:
        for line in f:
            if line.strip() and 'gen' not in line:  # Skip headers and empty lines
                data = line.split()
                gen, best, mean, std = int(data[0]), float(data[1]), float(data[2]), float(data[3])
                generations.append(gen)
                max_fitness.append(best)
                mean_fitness.append(mean)
    
    return np.array(generations), np.array(mean_fitness), np.array(max_fitness)

def aggregate_fitness_results(exp_dir, num_runs=10):
    """
    Aggregates the fitness results from the 10 indepentden runs.
    Calculates the average mean and max fitness over all runs.
    """
    all_mean_fitness = []
    all_max_fitness = []

    for run in range(1, num_runs + 1):
        file_path = os.path.join(exp_dir, f'results_run{run}.txt')  # Assuming results are stored as results_run1.txt, etc.
        gens, mean_fitness, max_fitness = read_fitness_results(file_path)
        
        if run == 1:  # Initialize lists on the first run
            aggregated_gens = gens  # Generation numbers should be the same for all runs
            all_mean_fitness = np.zeros((num_runs, len(mean_fitness)))
            all_max_fitness = np.zeros((num_runs, len(max_fitness)))

        all_mean_fitness[run - 1, :] = mean_fitness
        all_max_fitness[run - 1, :] = max_fitness

    avg_mean_fitness = np.mean(all_mean_fitness, axis=0)
    avg_max_fitness = np.mean(all_max_fitness, axis=0)

    return aggregated_gens, avg_mean_fitness, avg_max_fitness

def plot_fitness_comparison(alg1_dir, alg2_dir, enemy, params, num_runs=10):
    """
    Generates a plot comparing the fitness of two algorithms for a specific enemy.
    The plot will show four lines:
      1. Mean fitness of Algorithm 1
      2. Max fitness of Algorithm 1
      3. Mean fitness of Algorithm 2
      4. Max fitness of Algorithm 2
    """
    # Get fitness results for both algorithms
    gens1, alg1_mean, alg1_max = aggregate_fitness_results(alg1_dir, num_runs)
    gens2, alg2_mean, alg2_max = aggregate_fitness_results(alg2_dir, num_runs)

    # Create the plot
    plt.figure(figsize=(10, 6))
    
    # Algorithm 1 plots
    plt.plot(gens1, alg1_mean, label=f'Algorithm 1 - Mean Fitness', color='blue', linestyle='--')
    plt.plot(gens1, alg1_max, label=f'Algorithm 1 - Max Fitness', color='blue')
    
    # Algorithm 2 plots
    plt.plot(gens2, alg2_mean, label=f'Algorithm 2 - Mean Fitness', color='red', linestyle='--')
    plt.plot(gens2, alg2_max, label=f'Algorithm 2 - Max Fitness', color='red')
    
    # Plot formatting
    plt.title(f'Fitness Comparison for Enemy {enemy} - Params: {params}')
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.legend()
    plt.grid(True)
    
    # Save plot as an image or show
    plt.savefig(f'fitness_comparison_enemy_{enemy}.png')
    plt.show()


# Example usage:
# fill in enemy number, directories that contain the result files for each algorithm, string identifying parameter settings, 
plot_fitness_comparison('alg1/enemy1/params', 'alg2/enemy1/params', enemy=1, params='setting1')
plot_fitness_comparison('alg1/enemy2/params', 'alg2/enemy2/params', enemy=2, params='setting2')
plot_fitness_comparison('alg1/enemy3/params', 'alg2/enemy3/params', enemy=3, params='setting3')




### BOXPLOTS
def read_gain_results(file_path):
    """
    Reads the player energy and enemy energy from the file and computes the individual gain.
    Assumes each line has the format: player_energy enemy_energy
    """
    player_energy = []
    enemy_energy = []
    
    with open(file_path, 'r') as f:
        for line in f:
            if line.strip():
                data = line.split()
                player_energy.append(float(data[0]))
                enemy_energy.append(float(data[1]))
    
    # Calculate individual gain as player_energy - enemy_energy
    individual_gain = np.array(player_energy) - np.array(enemy_energy)
    
    return individual_gain

def compute_mean_gain_for_runs(exp_dir, num_runs=10, num_tests=5):
    """
    Computes the mean individual gain for each algorithm's 10 independent runs.
    For each run, it averages the results over 5 test runs.
    """
    all_mean_gains = []

    for run in range(1, num_runs + 1):
        run_gains = []
        for test in range(1, num_tests + 1):
            file_path = os.path.join(exp_dir, f'run{run}_test{test}.txt')  # Assuming the file is named like 'run1_test1.txt'
            individual_gain = read_gain_results(file_path)
            run_gains.append(np.mean(individual_gain))  # Average gain over this test
        
        mean_gain = np.mean(run_gains)  # Mean of 5 tests for this run
        all_mean_gains.append(mean_gain)

    return all_mean_gains

def create_boxplot_comparison(alg1_dir, alg2_dir, enemies, num_runs=10, num_tests=5):
    """
    Creates a figure containing 6 boxplots (3 enemies, 2 algorithms).
    Each box plot will compare the individual gains for the two algorithms on each enemy.
    """
    # Initialize data storage for each enemy
    all_gains_alg1 = []
    all_gains_alg2 = []

    for enemy in enemies:
        # Compute mean gains for Algorithm 1 and 2 for the current enemy
        alg1_gains = compute_mean_gain_for_runs(os.path.join(alg1_dir, f'enemy{enemy}'), num_runs, num_tests)
        alg2_gains = compute_mean_gain_for_runs(os.path.join(alg2_dir, f'enemy{enemy}'), num_runs, num_tests)
        
        # Store the results
        all_gains_alg1.append(alg1_gains)
        all_gains_alg2.append(alg2_gains)

    # Create box plots
    plt.figure(figsize=(10, 6))
    positions = np.array([1, 2, 4, 5, 7, 8])  # Positions for the box plots (pairs for each enemy)
    
    # Box plots for Algorithm 1 (blue) and Algorithm 2 (red)
    plt.boxplot(all_gains_alg1, positions=positions[::2], widths=0.6, patch_artist=True,
                boxprops=dict(facecolor='lightblue'), medianprops=dict(color='blue'))
    plt.boxplot(all_gains_alg2, positions=positions[1::2], widths=0.6, patch_artist=True,
                boxprops=dict(facecolor='lightcoral'), medianprops=dict(color='red'))
    
    # X-axis labels for enemies
    plt.xticks([1.5, 4.5, 7.5], [f'Enemy {enemy}' for enemy in enemies])
    
    # Add title and labels
    plt.title('Comparison of Individual Gain by Enemy (Algorithm 1 vs Algorithm 2)')
    plt.ylabel('Individual Gain (Player Energy - Enemy Energy)')
    plt.xlabel('Enemy')

    # Add a legend
    plt.legend([plt.Line2D([0], [0], color='blue', lw=2),
                plt.Line2D([0], [0], color='red', lw=2)],
               ['Algorithm 1', 'Algorithm 2'], loc='upper right')

    # Show or save the plot
    plt.tight_layout()
    plt.savefig('individual_gain_boxplot_comparison.png')
    plt.show()

# Example usage for 3 enemies
create_boxplot_comparison('alg1/results', 'alg2/results', enemies=[1, 2, 3], num_runs=10, num_tests=5)