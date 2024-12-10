import numpy as np
import matplotlib.pyplot as plt
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.abspath(os.path.join(current_dir, "../"))
sys.path.insert(0, src_dir)
from src.poissons import PoissonsEquation1D_MultipleInterfaces

###### Deterministic Solution ######
params = {
    "length": 1.0,            # Length of the domain
    "interfaces": [0.2, 0.4, 0.6, 0.8],  # Interface locations
    "k_values": [1.0, 0.25, 0.9, 0.1, 0.8],  # Conductivities in each region
    "f_values": [1, 1, 1, 1, 1],  # Forcing terms in each region
    "num_points": 201         # Number of grid points
}

problem_determ = PoissonsEquation1D_MultipleInterfaces(params)
problem_determ.setup_matrix()
solution_determ = problem_determ.solve()

##### MC Simulation Parameters ####
num_samples_mc = 10000  # Number of Monte Carlo simulations
k_means = np.array(params.get("k_values"))
f_means = np.array(params.get("f_values"))
length = params.get("length")
num_points = params.get("num_points")

# Different UQ values to analyze
uq_values = [0.001, 0.01, 0.1, 0.125, 0.15, 0.175, 0.2, 0.225, 0.25]
efficiency_values = []

for UQ in uq_values:
    k_stds = np.ones_like(k_means) * UQ  
    f_stds = np.ones_like(f_means) * UQ  

    # Vanilla MC Simulation (reference mean and std)
    solutions_mc = np.zeros((num_samples_mc, num_points))
    k_samples = np.random.normal(k_means, k_stds, (num_samples_mc, len(k_means)))
    f_samples = np.random.normal(f_means, f_stds, (num_samples_mc, len(f_means)))

    for i in range(num_samples_mc):
        params["k_values"] = k_samples[i]
        params["f_values"] = f_samples[i]

        problem = PoissonsEquation1D_MultipleInterfaces(params)
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
        k_samples = np.random.normal(k_means, k_stds, (samples, len(k_means)))
        f_samples = np.random.normal(f_means, f_stds, (samples, len(f_means)))
        
        # Setup solutions for the current sample batch
        solutions_cv = []
        for i in range(samples):
            params["k_values"] = k_samples[i]
            params["f_values"] = f_samples[i]
            problem = PoissonsEquation1D_MultipleInterfaces(params)
            problem.setup_matrix()
            solution = problem.solve()
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
plt.title("1D multiple-interface problem")
plt.show()
