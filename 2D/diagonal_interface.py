import numpy as np
import matplotlib.pyplot as plt
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.abspath(os.path.join(current_dir, "../"))
sys.path.insert(0, src_dir)
from src.poissons import PoissonsEquation2D

###### Deterministic Solution ######
params = {
    "length": 1.0,
    "num_points": 101,
    "k1": 1.0,
    "k2": 0.25,
    "f1": 0.5,
    "f2": 0.5
}

problem_determ = PoissonsEquation2D(params)
problem_determ.setup_matrix()
solution_determ = problem_determ.solve()

##### MC Simulation Parameters ####
UQ = 0.08
num_samples = 1000
k1_mean, k1_std = params["k1"], UQ
k2_mean, k2_std = params["k2"], UQ
f1, f2 = params["f1"], params["f2"]
num_points = params["num_points"]

solutions = np.zeros((num_samples, num_points, num_points))

# Monte Carlo loop
for i in range(num_samples):
    k1 = np.random.normal(k1_mean, k1_std)
    k2 = np.random.normal(k2_mean, k2_std)
    k1, k2 = max(k1, 1e-3), max(k2, 1e-3)
    mc_params = {
        "length": params["length"],
        "num_points": num_points,
        "k1": k1,
        "k2": k2,
        "f1": f1,
        "f2": f2
    }
    problem_mc = PoissonsEquation2D(mc_params)
    problem_mc.setup_matrix()
    solutions[i, :, :] = problem_mc.solve()

# Compute mean and standard deviation
mean_solution = np.mean(solutions, axis=0)
std_solution = np.std(solutions, axis=0)

# Plot mean solution
plt.figure(figsize=(8, 6))
plt.contourf(problem_determ.x, problem_determ.y, mean_solution, levels=50, cmap="viridis")
plt.colorbar(label="Mean solution u(x, y)")
plt.plot(
    [0, params["length"]], [0, params["length"]],
    color='red', linestyle='--', label="Interface (x=y)"
)
plt.xlabel("x")
plt.ylabel("y")
plt.title("Mean via MC")
plt.legend()
plt.show()

# Plot standard deviation
plt.figure(figsize=(8, 6))
plt.contourf(problem_determ.x, problem_determ.y, std_solution, levels=50, cmap="inferno")
plt.colorbar(label="Standard deviation of u(x, y)")
plt.plot(
    [0, params["length"]], [0, params["length"]],
    color='red', linestyle='--', label="Interface (x=y)"
)
plt.xlabel("x")
plt.ylabel("y")
plt.title("Uncertainty (Std. Dev) via MC")
plt.legend()
plt.show()