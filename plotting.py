import numpy as np
import matplotlib.pyplot as plt

# Load data from EA_1, EA_2, and EA2_V2
ea1_data = np.loadtxt('EA_1/stats.out', delimiter=',', skiprows=1)
ea2_data = np.loadtxt('EA_2/stats.out', delimiter=',', skiprows=1)
ea2v2_data = np.loadtxt('EA2_V2/stats.out', delimiter=',', skiprows=1)

# Extract generations and fitness data
generations = ea1_data[:, 0]  # Assuming all have same number of generations
ea1_mean, ea1_std, ea1_max = ea1_data[:, 1], ea1_data[:, 2], ea1_data[:, 3]
ea2_mean, ea2_std, ea2_max = ea2_data[:, 1], ea2_data[:, 2], ea2_data[:, 3]
ea2v2_mean, ea2v2_std, ea2v2_max = ea2v2_data[:, 1], ea2v2_data[:, 2], ea2v2_data[:, 3]

# Create the plot
plt.figure(figsize=(12, 8))

# Plot mean fitness
plt.plot(generations, ea1_mean, label='EA_1 Mean', color='blue')
plt.plot(generations, ea2_mean, label='EA_2 Mean', color='red')
plt.plot(generations, ea2v2_mean, label='EA2_V2 Mean', color='green')

# Plot standard deviation as shaded area
plt.fill_between(generations, ea1_mean - ea1_std, ea1_mean + ea1_std, alpha=0.2, color='blue')
plt.fill_between(generations, ea2_mean - ea2_std, ea2_mean + ea2_std, alpha=0.2, color='red')
plt.fill_between(generations, ea2v2_mean - ea2v2_std, ea2v2_mean + ea2v2_std, alpha=0.2, color='green')

# Plot max fitness
plt.plot(generations, ea1_max, label='EA_1 Max', color='blue', linestyle='--')
plt.plot(generations, ea2_max, label='EA_2 Max', color='red', linestyle='--')
plt.plot(generations, ea2v2_max, label='EA2_V2 Max', color='green', linestyle='--')

# Customize the plot
plt.title('Fitness over Generations: EA_1 vs EA_2 vs EA2_V2')
plt.xlabel('Generation')
plt.ylabel('Fitness')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)

# Save the plot
plt.savefig('fitness_comparison.png')
plt.close()

print("Plot saved as fitness_comparison.png")

