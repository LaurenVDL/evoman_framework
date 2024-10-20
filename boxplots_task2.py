import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from evoman.environment import Environment
from demo_controller import player_controller
#import matplotlib.patches as mpatches

def find_best_individuals(algorithm, group):
    best_individuals = []
    for run in range(1, 11):  # 10 runs
        stats_file = f'{algorithm}_task2_egroup{group}/R{run}/stats.out'
        if os.path.exists(stats_file):
            data = np.loadtxt(stats_file, delimiter=',', skiprows=1)
            best_gen = np.argmax(data[:, 3])  # Best fitness is in the 4th column
            weights_file = f'{algorithm}_task2_egroup{group}/R{run}/best/{best_gen}.out'
            best_weights = np.loadtxt(weights_file)
            best_individuals.append(best_weights)
    return best_individuals

def run_simulation(env, individual):
    f, p, e, t = env.play(pcont=individual)
    return p - e, p, e  # Return gain, player energy, and enemy energy

def evaluate_against_all_enemies(env, individuals):
    results = []
    energy_details = []
    for individual in individuals:
        total_gain = 0
        individual_energy_details = []
        for enemy in range(1, 9):  # All 8 enemies
            env.update_parameter('enemies', [enemy])
            gain, p, e = run_simulation(env, individual)
            total_gain += gain
            individual_energy_details.append((enemy, p, e))
        results.append(total_gain)
        energy_details.append(individual_energy_details)
    return results, energy_details

def create_boxplot(all_results):
    fig, ax = plt.subplots(figsize=(12, 6))
    
    labels = ['EA1 G1', 'EA2 G1', 'EA1 G2', 'EA2 G2']
    
    # Create boxplot
    bp = ax.boxplot(all_results, patch_artist=True, labels=labels)
    
    # Customize boxplot colors
    colors = ['#FF1493', '#00BFFF'] * 2  # Deep Pink for EA1, Deep Sky Blue for EA2
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
    means = [np.mean(result) for result in all_results]
    mean_points = ax.scatter(range(1, len(all_results) + 1), means, color='yellow', s=50, zorder=3)
    
    # Set title with increased size and bold font
    ax.set_title('Mean individual gain of best performing individuals in EA1 and EA2', 
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
    # g1_patch = mpatches.Patch(color='gray', label='G1')  # Customize color as needed
    # g2_patch = mpatches.Patch(color='lightgray', label='G2')  # Customize color as needed
    ax.legend([ea1_patch, ea2_patch, mean_points], ['EA1', 'EA2', 'Mean'], 
              loc='upper right', bbox_to_anchor=(1, 1), fontsize=12)
    # Add G1 and G2 as separate text labels
    ax.text(0.875, 0.1, 'G1-[2,3,4]', fontsize=12, color='black', transform=ax.transAxes)
    ax.text(0.875, 0.05, 'G2-[1,5,6,7,8]', fontsize=12, color='black', transform=ax.transAxes)
    
    # Increase tick label font size
    ax.tick_params(axis='both', which='major', labelsize=12)
    
    plt.tight_layout()
    plt.savefig('boxplot_comparison_task2.png', dpi=300)
    plt.close()

def main():
    env = Environment(
        playermode="ai",
        player_controller=player_controller(10),
        enemymode="static",
        level=2,
        speed="fastest",
        visuals=False
    )

    all_results = []
    all_individuals = []
    all_energy_details = []

    # Process in the order: EA1 G1, EA2 G1, EA1 G2, EA2 G2
    for group in [1, 2]:
        for algorithm in ['EA1', 'EA2']:
            print(f"\nProcessing {algorithm} Group {group}")
            best_individuals = find_best_individuals(algorithm, group)
            results, energy_details = evaluate_against_all_enemies(env, best_individuals)
            all_results.append(results)
            all_energy_details.extend(energy_details)
            all_individuals.extend([(algorithm, group, individual) for individual in best_individuals])
            print(f"Results for {algorithm} Group {group}: {results}")

    # Create boxplot
    create_boxplot(all_results)
    print("Boxplot created and saved as 'boxplot_comparison_task2.png'")

    # Find the overall best individual
    best_gain = max(max(results) for results in all_results)
    best_index = next(i for i, results in enumerate(all_results) if max(results) == best_gain)
    best_run = all_results[best_index].index(best_gain)
    best_algorithm, best_group, best_individual = all_individuals[best_index * 10 + best_run]

    print(f"\nBest Individual Overall:")
    print(f"Algorithm: {best_algorithm}")
    print(f"Group: {best_group}")
    print(f"Run: {best_run + 1}")
    print(f"Total Gain: {best_gain}")

    # Save the weights of the best individual
    np.savetxt('best_individual_weights.txt', best_individual)
    print("Best individual weights saved to 'best_individual_weights.txt'")

    # Print energy points for the best individual against all enemies
    print("\nEnergy points for the best individual against all enemies:")
    best_energy_details = all_energy_details[best_index * 10 + best_run]
    for enemy, p, e in best_energy_details:
        print(f"Enemy {enemy}: Player Energy = {p:.2f}, Enemy Energy = {e:.2f}, Gain = {p - e:.2f}")

    # Perform Mann-Whitney U test for each group
    for i in range(0, len(all_results), 2):
        ea1_gains = all_results[i]
        ea2_gains = all_results[i+1]
        # Perform independent t-test
        t_statistic, p_value = stats.ttest_ind(ea1_gains, ea2_gains, equal_var=False)  # Use equal_var=False for Welch's t-test
        print(f"\nT-Test results for Group {i//2 + 1}:")
        print(f"T-Statistic: {t_statistic}")
        print(f"p-value: {p_value}")
        # statistic, p_value = stats.mannwhitneyu(ea1_gains, ea2_gains, alternative='two-sided')
        # print(f"\nMann-Whitney U test results for Group {i//2 + 1}:")
        # print(f"Statistic: {statistic}")
        # print(f"p-value: {p_value}")

    # Print summary statistics
    for i, results in enumerate(all_results):
        algorithm = 'EA1' if i % 2 == 0 else 'EA2'
        group = i // 2 + 1
        print(f"\n{algorithm} Group {group}:")
        print(f"Mean: {np.mean(results):.6f}, Std: {np.std(results):.6f}")

    

if __name__ == "__main__":
    main()
