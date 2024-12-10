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
UQ = 0.05
num_samples = 10000 
k_means = np.array(params.get("k_values"))
f_means = np.array(params.get("f_values"))
k_stds = np.ones_like(k_means) * UQ  
f_stds = np.ones_like(f_means) * UQ 

solutions = np.zeros((num_samples, params["num_points"]))

# Monte Carlo loop
for i in range(num_samples):
    k_samples = np.random.normal(k_means, k_stds)
    f_samples = np.random.normal(f_means, f_stds)

    params["k_values"] = k_samples
    params["f_values"] = f_samples

    problem = PoissonsEquation1D_MultipleInterfaces(params)
    problem.setup_matrix()
    solution = problem.solve()
    solutions[i, :] = solution

mean_solution = np.mean(solutions, axis=0)
std_solution = np.std(solutions, axis=0)

x = np.linspace(0, params["length"], params["num_points"])
plt.figure(figsize=(8, 6))
plt.plot(x, mean_solution, label="Mean", color="blue")
plt.fill_between(x, mean_solution - std_solution, mean_solution + std_solution, 
                 color="blue", alpha=0.3, label="Mean ± 1 Std. dev.")
plt.plot(x, solution_determ, label='Deterministic', color="black", linestyle='--')
plt.xlabel("x")
plt.ylabel("u(x)")
plt.title(f"UQ via MC Simulation ($\\sigma = {UQ:.2f}$)")
plt.legend()
plt.show()