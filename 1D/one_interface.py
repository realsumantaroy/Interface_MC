import numpy as np
import matplotlib.pyplot as plt
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.abspath(os.path.join(current_dir, "../"))
sys.path.insert(0, src_dir)
from src.poissons import PoissonsEquation1D

###### Deterministic Solution ######
params = {
    "length": 1.0,     # Length of the domain
    "interface": 0.4,  #interface location
    "k1": 0.1,          # Conductivity k1
    "k2": 1.0,          # Conductivity k2
    "f1": 0.75,
    "f2": 0.75,
    "num_points": 101  # Number of grid points
}


problem_determ = PoissonsEquation1D(params)
problem_determ.setup_matrix()
solution_determ = problem_determ.solve()

##### MC Simulation Parameters ####
UQ=0.02

num_samples = 10000  # Number of Monte Carlo simulations
k1_mean, k1_std = params.get("k1"), UQ  # Mean and standard deviation of k
k2_mean, k2_std = params.get("k2"), UQ  # Mean and standard deviation of k
length = params.get("length")
f1 = params.get("f1")
f2 = params.get("f2")
num_points = params.get("num_points")

solutions = np.zeros((num_samples, num_points))

# Monte Carlo loop
for i in range(num_samples):
    k1 = np.random.normal(k1_mean, k1_std)
    k2 = np.random.normal(k2_mean, k2_std)
    k1, k2 = max(k1, 1e-3), max(k2, 1e-3)
    params = {
        "length": length,
        "k1": k1,
        "k2": k2,
        "f1": f1,
        "f2": f2,
        "num_points": num_points
    }

    problem = PoissonsEquation1D(params)
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
plt.title(f"UQ via MC ($\\sigma = {UQ:.2f}$)")
plt.legend()
plt.show()
