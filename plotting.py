# import numpy as np
# import matplotlib.pyplot as plt

# # Load data from EA_1, EA_2, and EA2_V2
# ea1_data = np.loadtxt('EA_1/stats.out', delimiter=',', skiprows=1)
# ea2_data = np.loadtxt('EA_2/stats.out', delimiter=',', skiprows=1)
# ea2v2_data = np.loadtxt('EA2_V2/stats.out', delimiter=',', skiprows=1)

# # Extract generations and fitness data
# generations = ea1_data[:, 0]  # Assuming all have same number of generations
# ea1_mean, ea1_std, ea1_max = ea1_data[:, 1], ea1_data[:, 2], ea1_data[:, 3]
# ea2_mean, ea2_std, ea2_max = ea2_data[:, 1], ea2_data[:, 2], ea2_data[:, 3]
# ea2v2_mean, ea2v2_std, ea2v2_max = ea2v2_data[:, 1], ea2v2_data[:, 2], ea2v2_data[:, 3]

# # Create the plot
# plt.figure(figsize=(12, 8))

# # Plot mean fitness
# plt.plot(generations, ea1_mean, label='EA_1 Mean', color='blue')
# plt.plot(generations, ea2_mean, label='EA_2 Mean', color='red')
# plt.plot(generations, ea2v2_mean, label='EA2_V2 Mean', color='green')

# # Plot standard deviation as shaded area
# plt.fill_between(generations, ea1_mean - ea1_std, ea1_mean + ea1_std, alpha=0.2, color='blue')
# plt.fill_between(generations, ea2_mean - ea2_std, ea2_mean + ea2_std, alpha=0.2, color='red')
# plt.fill_between(generations, ea2v2_mean - ea2v2_std, ea2v2_mean + ea2v2_std, alpha=0.2, color='green')

# # Plot max fitness
# plt.plot(generations, ea1_max, label='EA_1 Max', color='blue', linestyle='--')
# plt.plot(generations, ea2_max, label='EA_2 Max', color='red', linestyle='--')
# plt.plot(generations, ea2v2_max, label='EA2_V2 Max', color='green', linestyle='--')

# # Customize the plot
# plt.title('Fitness over Generations: EA_1 vs EA_2 vs EA2_V2')
# plt.xlabel('Generation')
# plt.ylabel('Fitness')
# plt.legend()
# plt.grid(True, linestyle='--', alpha=0.7)

# # Save the plot
# plt.savefig('fitness_comparison.png')
# plt.close()

# print("Plot saved as fitness_comparison.png")

import os
import numpy as np
import matplotlib.pyplot as plt

def load_and_process_data(base_folder, num_runs=10, num_generations=30):
    all_means = []
    all_bests = []
    all_stds = []
    for run in range(1, num_runs + 1):
        file_path = os.path.join(base_folder, f'R{run}', 'stats.out')
        if os.path.exists(file_path):
            data = np.loadtxt(file_path, delimiter=',', skiprows=1)
            if len(data) >= num_generations:
                all_means.append(data[:num_generations, 1])
                all_bests.append(data[:num_generations, 3])
                all_stds.append(data[:num_generations, 2])
    return (np.mean(all_means, axis=0), np.mean(all_bests, axis=0), 
            np.mean(all_stds, axis=0))

def plot_comparison(ea1_folder, ea2_folder, enemy, output_filename):
    generations = np.arange(30)
    
    ea1_mean, ea1_best, ea1_std = load_and_process_data(ea1_folder)
    ea2_mean, ea2_best, ea2_std = load_and_process_data(ea2_folder)
    
    plt.figure(figsize=(12, 8))
    
    # EA1 color: Deep Pink
    plt.plot(generations, ea1_mean, label='EA1 Mean', color='#FF1493')
    plt.plot(generations, ea1_best, label='EA1 Best', color='#FF1493', linestyle='--')
    plt.fill_between(generations, ea1_mean - ea1_std, ea1_mean + ea1_std, 
                     alpha=0.2, color='#FF1493')
    
    # EA2 color: Deep Sky Blue
    plt.plot(generations, ea2_mean, label='EA2 Mean', color='#00BFFF')
    plt.plot(generations, ea2_best, label='EA2 Best', color='#00BFFF', linestyle='--')
    plt.fill_between(generations, ea2_mean - ea2_std, ea2_mean + ea2_std, 
                     alpha=0.2, color='#00BFFF')
    
    plt.title(f'EA1 vs EA2 (Enemy {enemy})')
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.savefig(output_filename)
    plt.close()
    
    print(f"Plot saved as {output_filename}")

# Generate plots for enemies 1, 5, and 6
plot_comparison('EA1_enemy1', 'EA2_enemy1', 1, 'fitness_comparison_enemy1.png')
plot_comparison('EA1_enemy5', 'EA2_enemy5', 5, 'fitness_comparison_enemy5.png')
plot_comparison('EA1_enemy6', 'EA2_enemy6', 6, 'fitness_comparison_enemy6.png')

