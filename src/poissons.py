import numpy as np
import matplotlib.pyplot as plt

class PoissonsEquation1D:
    def __init__(self, params):
        """
        Initialize the problem with given parameters.
        
        Args:
        params (dict): Dictionary containing problem parameters.
        """
        # Physical parameters
        self.length = params.get("length", 1.0)
        self.interface = params.get("interface", 0.4)
        self.k1 = params.get("k1", 1.0)
        self.k2 = params.get("k2", 0.2)
        self.f1 = params.get("f1", 1.0)  # Forcing term in region 1
        self.f2 = params.get("f2", 0.5)  # Forcing term in region 2

        # Discretization
        self.num_points = params.get("num_points", 101)
        self.dx = self.length / (self.num_points - 1)
        self.x = np.linspace(0, self.length, self.num_points)
        self.interface_index = np.where(self.x >= self.interface)[0][0]

        # Matrix and RHS vector
        self.A = np.zeros((self.num_points, self.num_points))
        self.b = np.zeros(self.num_points)

    def setup_matrix(self):
        for i in range(1, self.num_points - 1):
            if i < self.interface_index:  # Region with k1
                self.A[i, i - 1] = self.k1 / self.dx**2
                self.A[i, i] = -2 * self.k1 / self.dx**2
                self.A[i, i + 1] = self.k1 / self.dx**2
                self.b[i] = -self.f1  # Forcing term in region 1
            elif i > self.interface_index:  # Region with k2
                self.A[i, i - 1] = self.k2 / self.dx**2
                self.A[i, i] = -2 * self.k2 / self.dx**2
                self.A[i, i + 1] = self.k2 / self.dx**2
                self.b[i] = -self.f2  # Forcing term in region 2

        # Interface conditions
        self.A[self.interface_index, self.interface_index - 1] = self.k1 / self.dx**2
        self.A[self.interface_index, self.interface_index] = -self.k1 / self.dx**2 - self.k2 / self.dx**2
        self.A[self.interface_index, self.interface_index + 1] = self.k2 / self.dx**2
        self.b[self.interface_index] = -(self.f1 + self.f2) / 2  # Average forcing term at the interface

        # Boundary conditions
        self.A[0, 0] = 1.0  # u=0 at x=0
        self.A[-1, -1] = 1.0  # u=0 at x=1
        self.b[0] = 0.0
        self.b[-1] = 0.0

    def solve(self):
        """Solves the linear system and returns the solution."""
        return np.linalg.solve(self.A, self.b)

    def plot_solution(self, u):
        """Plots the solution."""
        plt.plot(self.x, u, label="Numerical solution")
        plt.axvline(x=self.interface, color='r', linestyle='--', label="Interface")
        plt.xlabel("x")
        plt.ylabel("u(x)")
        # plt.title("Solution of Poisson's Equation with Interface")
        plt.legend()
        plt.show()

# Usage
if __name__ == "__main__":
    params = {
        "length": 1.0,
        "interface": 0.4,
        "k1": 1.0,
        "k2": 0.2,
        "f1": 1.0,
        "f2": 1.0,
        "num_points": 101
    }

    problem = PoissonsEquation1D(params)
    problem.setup_matrix()
    solution = problem.solve()
    problem.plot_solution(solution)
