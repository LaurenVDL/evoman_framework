import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from evoman.environment import Environment
from demo_controller import player_controller

# def find_best_individual(algorithm, enemy):
#     best_fitness = -float('inf')
#     best_run = -1
#     best_gen = -1
    
#     for run in range(1, 11):
#         stats_file = f'{algorithm}_enemy{enemy}/R{run}/stats.out'
#         if os.path.exists(stats_file):
#             data = np.loadtxt(stats_file, delimiter=',', skiprows=1)
#             max_fitness = np.max(data[:, 3])  # Best fitness is in the 4th column
#             if max_fitness > best_fitness:
#                 best_fitness = max_fitness
#                 best_run = run
#                 best_gen = np.argmax(data[:, 3])
    
#     return best_run, best_gen

def get_best_weights(algorithm, enemy, run, gen):
    weights_file = f'{algorithm}_enemy{enemy}/R{run}/best/{gen}.out'
    return np.loadtxt(weights_file)

# def run_simulation(env, individual, num_trials=5):
#     gains = []
#     for _ in range(num_trials):
#         f, p, e, t = env.play(pcont=individual)
#         gain = p - e
#         gains.append(gain)
#     return np.mean(gains)

# def create_boxplot(data, labels):
#     fig, ax = plt.subplots(figsize=(12, 6))
#     bp = ax.boxplot(data, patch_artist=True, labels=labels)
    
#     colors = ['#3498db', '#e74c3c'] * 3
#     for patch, color in zip(bp['boxes'], colors):
#         patch.set_facecolor(color)
    
#     ax.set_title('Mean individual gain of best performing individual in forced and no migration EA')
#     ax.set_xlabel('Experiment name')
#     ax.set_ylabel('Individual gain')
    
#     for i in range(0, len(data), 2):
#         t_stat, p_value = stats.ttest_ind(data[i], data[i+1])
#         ax.text((i+1.5), ax.get_ylim()[1], f'p = {p_value:.3f}', 
#                 horizontalalignment='center', verticalalignment='bottom')
#         ax.annotate('', xy=(i+1, ax.get_ylim()[1]), xytext=(i+2, ax.get_ylim()[1]),
#                     arrowprops=dict(arrowstyle='-', lw=1.5))
    
#     plt.tight_layout()
#     plt.savefig('boxplot_comparison.png')
#     plt.close()

# def main():
#     enemies = [1, 5, 6]
#     algorithms = ['EA1', 'EA2']
#     all_results = []
#     labels = []
    
#     for enemy in enemies:
#         env = Environment(enemies=[enemy],
#                           playermode="ai",
#                           player_controller=player_controller(10),
#                           enemymode="static",
#                           level=2,
#                           speed="fastest",
#                           visuals=False)
        
#         for algorithm in algorithms:
#             best_run, best_gen = find_best_individual(algorithm, enemy)
#             best_weights = get_best_weights(algorithm, enemy, best_run, best_gen)
            
#             gains = []
#             for _ in range(10):  # 10 independent runs
#                 gain = run_simulation(env, best_weights)
#                 gains.append(gain)
            
#             all_results.append(gains)
#             labels.append(f'{algorithm} E{enemy}')
        
#         print(f"Enemy {enemy} processed")
    
#     create_boxplot(all_results, labels)

# Modify the find_best_individual function
def find_best_individual(algorithm, enemy, run):
    stats_file = f'{algorithm}_enemy{enemy}/R{run}/stats.out'
    if os.path.exists(stats_file):
        data = np.loadtxt(stats_file, delimiter=',', skiprows=1)
        best_gen = np.argmax(data[:, 3])  # Best fitness is in the 4th column
        best_fitness = data[best_gen, 3]
        return best_gen, best_fitness
    return None, None

# Reset all_results and labels for the new approach
all_results = []
labels = []
enemies = [1, 5, 6]
algorithms = ['EA1', 'EA2']

headless = True
if headless:
    os.environ["SDL_VIDEODRIVER"] = "dummy"

for enemy in enemies:
    env = Environment(enemies=[enemy],
                        playermode="ai",
                        player_controller=player_controller(10),
                        enemymode="static",
                        level=2,
                        speed="fastest",
                        visuals=False)
    
    for algorithm in algorithms:
        run_gains = []
        
        for run in range(1, 11):  # For each of the 10 runs
            best_gen, _ = find_best_individual(algorithm, enemy, run)
            if best_gen is not None:
                best_weights = get_best_weights(algorithm, enemy, run, best_gen)
                
                # Run the best individual from this run 5 times
                gains = []
                for _ in range(5):
                    f, p, e, t = env.play(pcont=best_weights)
                    gain = p - e
                    gains.append(gain)
                
                # Calculate average gain for this run
                avg_run_gain = np.mean(gains)
                run_gains.append(avg_run_gain)
        
        # Calculate overall average gain across all runs
        overall_avg_gain = np.mean(run_gains)
        
        all_results.append(overall_avg_gain)
        labels.append(f'{algorithm} E{enemy}')
    
    print(f"Enemy {enemy} processed")

# Convert all_results to a list of lists
all_results = [result.tolist() if isinstance(result, np.ndarray) else [result] for result in all_results]

# Create a boxplot
plt.figure(figsize=(12, 6))
bp = plt.boxplot(all_results, labels=labels, patch_artist=True)

colors = ['#3498db', '#e74c3c'] * 3
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)

plt.xlabel('Experiment')
plt.ylabel('Average Gain')
plt.title('Average Gain of Best Individuals per EA and Enemy')
plt.xticks(rotation=45)

plt.tight_layout()
plt.savefig('average_gain_comparison.png')
plt.close()

# Print the results
for label, result in zip(labels, all_results):
    print(f"{label}: Mean = {np.mean(result):.2f}, Std = {np.std(result):.2f}")

# Perform t-tests to compare EA1 and EA2 for each enemy
for i in range(0, len(all_results), 2):
    ea1_gains = all_results[i]
    ea2_gains = all_results[i+1]
    t_stat, p_value = stats.ttest_ind(ea1_gains, ea2_gains)
    enemy = enemies[i//2]
    print(f"\nT-test results for Enemy {enemy}:")
    print(f"t-statistic: {t_stat}")
    print(f"p-value: {p_value}")

# # Create a boxplot
# plt.figure(figsize=(12, 6))
# bp = plt.boxplot(all_results, labels=labels, patch_artist=True)

# colors = ['#3498db', '#e74c3c'] * 3
# for patch, color in zip(bp['boxes'], colors):
#     patch.set_facecolor(color)

# plt.xlabel('Experiment')
# plt.ylabel('Average Gain')
# plt.title('Average Gain of Best Individuals per EA and Enemy')
# plt.xticks(rotation=45)

# plt.tight_layout()
# plt.savefig('average_gain_comparison.png')
# plt.close()

# # Print the results
# for label, result in zip(labels, all_results):
#     print(f"{label}: Mean = {np.mean(result):.2f}, Std = {np.std(result):.2f}")

# # Perform t-tests to compare EA1 and EA2 for each enemy
# for i in range(0, len(all_results), 2):
#     ea1_gains = all_results[i]
#     ea2_gains = all_results[i+1]
#     t_stat, p_value = stats.ttest_ind(ea1_gains, ea2_gains)
#     enemy = enemies[i//2]
#     print(f"\nT-test results for Enemy {enemy}:")
#     print(f"t-statistic: {t_stat}")
#     print(f"p-value: {p_value}")


