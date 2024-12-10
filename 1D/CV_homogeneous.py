import numpy as np
import matplotlib.pyplot as plt
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.abspath(os.path.join(current_dir, "../"))
sys.path.insert(0, src_dir)
from src.poissons import HomogeneousPoissonsEquation1D

###### Deterministic Solution ######
params = {
    "length": 1.0,     # Length of the domain
    "k": 0.5,          # Conductivity
    "source": 0.25,     # Forcing term
    "num_points": 101  # Number of grid points
}
problem_determ = HomogeneousPoissonsEquation1D(params)
problem_determ.setup_matrix()
solution_determ = problem_determ.solve()

##### MC Simulation Parameters ####
UQ=0.15

num_samples_mc = 10000  # Number of Monte Carlo simulations
k_mean, k_std = params.get("k"), UQ  # Mean and standard deviation of k
length = params.get("length")
source = params.get("source")
num_points = params.get("num_points")

# Different UQ values to analyze
uq_values = [0.001, 0.01, 0.1, 0.125, 0.15, 0.175, 0.2, 0.225, 0.25]
efficiency_values = []

for UQ in uq_values:
    k_std = UQ

    # Vanilla MC Simulation (reference mean and std)
    solutions_mc = np.zeros((num_samples_mc, num_points))
    for i in range(num_samples_mc):  # Start from 0 to num_samples_mc
        k = np.random.normal(k_mean, k_std)
        k = max(k, 1e-3)
        
        params = {
        "length": length,
        "k": k,
        "source": source,
        "num_points": num_points
        }

        problem = HomogeneousPoissonsEquation1D(params)
        problem.setup_matrix()
        solution = problem.solve()
        solutions_mc[i, :] = solution

    mean_mc = np.mean(solutions_mc, axis=0)
    std_mc = np.std(solutions_mc, axis=0)

    ###### Control Variate Simulation ######

    # Control variate: use the deterministic solution as the control variate
    control_variate = solution_determ

    # Iterative sample size for control variate method
    for samples in range(1, num_samples_mc + 1):  # Start from 1 and go up to num_samples_mc
        solutions_cv = []  # Initialize an empty list for storing solutions for each sample
        
        for i in range(samples):
            k = np.random.normal(k_mean, k_std)
            k = max(k, 1e-3)
            
            params = {
            "length": length,
            "k": k,
            "source": source,
            "num_points": num_points
            }

            problem = HomogeneousPoissonsEquation1D(params)
            problem.setup_matrix()
            solution = problem.solve()

            # Apply the control variate method: adjust the solution using the control variate
            adjusted_solution = solution - (solution_determ - control_variate)
            solutions_cv.append(adjusted_solution)

        solutions_cv = np.array(solutions_cv)
        mean_cv = np.mean(solutions_cv, axis=0)
        std_cv = np.std(solutions_cv, axis=0)

        # Check if the mean and std of the control variate method are close to the vanilla MC method
        mean_diff = np.abs(mean_cv - mean_mc).max()
        std_diff = np.abs(std_cv - std_mc).max()

        # If both mean and std are sufficiently close to the vanilla MC solution, stop
        if mean_diff < 1e-2 and std_diff < 1e-2:
            # Efficiency comparison
            efficiency = (num_samples_mc - samples) / num_samples_mc
            efficiency_values.append(efficiency * 100)  # Store efficiency as percentage
            print(f"UQ = {UQ:.3f}: Control variate method reaches the same mean and std with {samples} samples. Efficiency: {efficiency * 100:.2f}%")
            break

# Plot results
plt.figure(figsize=(8, 6))
plt.plot(uq_values, efficiency_values, marker='o', color='blue', linestyle='-', label="Efficiency (%)")
plt.xlabel("Uncertainty in input (std. deviation of K)")
plt.ylabel("Efficiency gain (%)")
plt.title("1D homogeneous problem")
plt.show()
