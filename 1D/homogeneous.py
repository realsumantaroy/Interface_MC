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

print()

problem_determ = HomogeneousPoissonsEquation1D(params)
problem_determ.setup_matrix()
solution_determ = problem_determ.solve()

##### MC Simulation Parameters ####
UQ=0.15

num_samples = 10000  # Number of Monte Carlo simulations
k_mean, k_std = params.get("k"), UQ  # Mean and standard deviation of k
length = params.get("length")
source = params.get("source")
num_points = params.get("num_points")

solutions = np.zeros((num_samples, num_points))

# Monte Carlo loop
for i in range(num_samples):
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
    solutions[i, :] = solution

# Compute statistics
mean_solution = np.mean(solutions, axis=0)
std_solution = np.std(solutions, axis=0)
x = np.linspace(0, length, num_points)

# Plot results
plt.figure(figsize=(8, 6))
plt.plot(x, mean_solution,label="Mean", color="blue")
plt.fill_between(x, mean_solution - std_solution, mean_solution + std_solution, 
                 color="blue", alpha=0.3, label="Mean ± 1 Std. dev.")
plt.scatter(x, solution_determ, label='Deterministic')
plt.xlabel("x")
plt.ylabel("u(x)")
plt.title(f"UQ via MC ($\\sigma = {k_std:.2f}$)")
plt.legend()
plt.show()
