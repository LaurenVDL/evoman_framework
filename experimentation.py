import os
import subprocess

# Set up parameters
enemies = [5]  # You can extend this list for multiple enemies
num_runs = 10

# Main loop to run EA multiple times
for enemy in enemies:
    ea_folder = f"EA1_enemy{enemy}"
    os.makedirs(ea_folder, exist_ok=True)
    
    for run in range(1, num_runs + 1):
        experiment_name = f"{ea_folder}/R{run}"
        # os.makedirs(experiment_name, exist_ok=True)
        
        print(f"Running EA1 against enemy {enemy}, run {run}")
        
        # Run the EA script with the current experiment name
        subprocess.run(["python", "EA_1.py", experiment_name, str(enemy)])

print("All runs completed.")




