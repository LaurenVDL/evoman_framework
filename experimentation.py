import os
import subprocess

# Set up parameters
#enemies = [3]  # You can extend this list for multiple enemies
enemy_groups = [[2,3,4]]#, [1,5,6,7,8]]
num_runs = 10

# Main loop to run EA multiple times
#for enemy in enemies:
for i in range(enemy_groups):
    ea_folder = f"EA1_task2_egroup{i+1}"
    os.makedirs(ea_folder, exist_ok=True)
    
    for run in range(1, num_runs + 1):
        experiment_name = f"{ea_folder}/R{run}"
        # os.makedirs(experiment_name, exist_ok=True)
        
        print(f"Running EA1 against group {i}, run {run}")
        
        # Run the EA script with the current experiment name
        subprocess.run(["python", "EA_1_task2.py", experiment_name, enemy_groups[i]])

print("All runs completed.")




