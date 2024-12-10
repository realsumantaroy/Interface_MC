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

class HomogeneousPoissonsEquation1D:
    def __init__(self, params):
        """
        Initialize the homogeneous Poisson's equation problem.

        Args:
        params (dict): Dictionary containing problem parameters.
        """
        # Physical parameters
        self.length = params.get("length", 1.0)
        self.k = params.get("k", 1.0) 
        self.source = params.get("source", 1.0)

        # Discretization
        self.num_points = params.get("num_points", 101)
        self.dx = self.length / (self.num_points - 1)
        self.x = np.linspace(0, self.length, self.num_points)

        # Matrix and RHS vector
        self.A = np.zeros((self.num_points, self.num_points))
        self.b = np.zeros(self.num_points)

    def setup_matrix(self):
        for i in range(1, self.num_points - 1):
            self.A[i, i - 1] = self.k / self.dx**2
            self.A[i, i] = -2 * self.k / self.dx**2
            self.A[i, i + 1] = self.k / self.dx**2
            self.b[i] = -self.source  # Forcing term

        # Boundary conditions
        self.A[0, 0] = 1.0  # u=0 at x=0
        self.A[-1, -1] = 1.0  # u=0 at x=L
        self.b[0] = 0.0
        self.b[-1] = 0.0

    def solve(self):
        """Solves the linear system and returns the solution."""
        return np.linalg.solve(self.A, self.b)

    def plot_solution(self, u):
        """Plots the solution."""
        plt.plot(self.x, u, label="Numerical solution")
        plt.xlabel("x")
        plt.ylabel("u(x)")
        plt.title("Homogeneous Poisson's equation")
        plt.legend()
        plt.show()



class PoissonsEquation2D:
    def __init__(self, params):
        """
        Initialize the 2D Poisson's equation problem with interface along x=y.
        
        Args:
        params (dict): Dictionary containing problem parameters.
        """
        # Physical parameters
        self.length = params.get("length", 1.0)
        self.num_points = params.get("num_points", 101)
        self.k1 = params.get("k1", 1.0)
        self.k2 = params.get("k2", 0.2)
        self.f1 = params.get("f1", 1.0)  # Forcing term in region 1
        self.f2 = params.get("f2", 0.5)  # Forcing term in region 2

        # Discretization
        self.dx = self.length / (self.num_points - 1)
        self.x, self.y = np.meshgrid(
            np.linspace(0, self.length, self.num_points),
            np.linspace(0, self.length, self.num_points)
        )
        self.interface = self.x >= self.y  # Diagonal interface (x >= y)
        self.num_grid_points = self.num_points**2  # Total number of grid points

        # Matrix and RHS vector
        self.A = np.zeros((self.num_grid_points, self.num_grid_points))
        self.b = np.zeros(self.num_grid_points)

    def _index(self, i, j):
        """Converts 2D (i, j) index to 1D index for matrix representation."""
        return i * self.num_points + j

    def setup_matrix(self):
        """Setup matrix A and vector b."""
        for i in range(self.num_points):
            for j in range(self.num_points):
                idx = self._index(i, j)
                
                if i == 0 or j == 0 or i == self.num_points - 1 or j == self.num_points - 1:
                    # Boundary conditions: u = 0 at the edges
                    self.A[idx, idx] = 1.0
                    self.b[idx] = 0.0
                else:
                    # Determine region: region 1 (k1, f1) or region 2 (k2, f2)
                    if self.interface[i, j]:  # Region 1 (above x=y)
                        k = self.k1
                        f = self.f1
                    else:  # Region 2 (below x=y)
                        k = self.k2
                        f = self.f2

                    # Poisson equation finite-difference
                    self.A[idx, self._index(i - 1, j)] = k / self.dx**2  # u(i-1, j)
                    self.A[idx, self._index(i + 1, j)] = k / self.dx**2  # u(i+1, j)
                    self.A[idx, self._index(i, j - 1)] = k / self.dx**2  # u(i, j-1)
                    self.A[idx, self._index(i, j + 1)] = k / self.dx**2  # u(i, j+1)
                    self.A[idx, idx] = -4 * k / self.dx**2  # u(i, j)
                    self.b[idx] = -f  # Forcing term

    def solve(self):
        """Solve the linear system and reshape the solution into 2D."""
        u_flat = np.linalg.solve(self.A, self.b)
        return u_flat.reshape((self.num_points, self.num_points))

    def plot_solution(self, u):
        """Plots the solution."""
        plt.figure(figsize=(8, 6))
        plt.contourf(self.x, self.y, u, levels=50, cmap="viridis")
        plt.colorbar(label="u(x, y)")
        plt.plot(
            [0, self.length], [0, self.length], color='red', linestyle='--', label="Interface (x=y)"
        )
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title("Deterministic")
        plt.legend()
        plt.grid(False)
        plt.show()




class PoissonsEquation1D_MultipleInterfaces:
    def __init__(self, params):
        """
        Initialize the problem with given parameters.
        
        Args:
        params (dict): Dictionary containing problem parameters.
        """
        # Physical parameters
        self.length = params.get("length", 1.0)
        self.interfaces = params.get("interfaces", [0.2, 0.4, 0.6, 0.8])  # Interface locations
        self.k_values = params.get("k_values", [1.0, 0.5, 0.2, 0.7, 1.2])  # Conductivities in each region
        self.f_values = params.get("f_values", [0.5, 0.8, 0.3, 0.6, 0.9])  # Forcing terms in each region

        # Discretization
        self.num_points = params.get("num_points", 101)
        self.dx = self.length / (self.num_points - 1)
        self.x = np.linspace(0, self.length, self.num_points)

        # Interface indices
        self.interface_indices = [np.where(self.x >= interface)[0][0] for interface in self.interfaces]
        
        # Matrix and RHS vector
        self.A = np.zeros((self.num_points, self.num_points))
        self.b = np.zeros(self.num_points)

    def setup_matrix(self):
        for i in range(1, self.num_points - 1):
            region = 0
            for idx in self.interface_indices:
                if i < idx:
                    break
                region += 1
            
            k = self.k_values[region]
            f = self.f_values[region]

            # Fill in matrix A and RHS vector b
            self.A[i, i - 1] = k / self.dx**2
            self.A[i, i] = -2 * k / self.dx**2
            self.A[i, i + 1] = k / self.dx**2
            self.b[i] = -f

        # Interface conditions
        for idx, interface_idx in enumerate(self.interface_indices):
            k1 = self.k_values[idx]
            k2 = self.k_values[idx + 1]
            f1 = self.f_values[idx]
            f2 = self.f_values[idx + 1]

            self.A[interface_idx, interface_idx - 1] = k1 / self.dx**2
            self.A[interface_idx, interface_idx] = -k1 / self.dx**2 - k2 / self.dx**2
            self.A[interface_idx, interface_idx + 1] = k2 / self.dx**2
            self.b[interface_idx] = -(f1 + f2) / 2

        # Boundary conditions
        self.A[0, 0] = 1.0  # u=0 at x=0
        self.A[-1, -1] = 1.0  # u=0 at x=1
        self.b[0] = 0.0
        self.b[-1] = 0.0

    def solve(self):
        return np.linalg.solve(self.A, self.b)

    def plot_solution(self, u):
        plt.plot(self.x, u, label="Numerical solution")
        for interface in self.interfaces:
            plt.axvline(x=interface, color='r', linestyle='--', label="Interface")
        plt.xlabel("x")
        plt.ylabel("u(x)")
        plt.show()




# Usage
if __name__ == "__main__":

    # #### 2D:
    # params = {
    #     "length": 1.0,  # Domain size
    #     "num_points": 101,  # Number of points in each dimension
    #     "k1": 1.0,  # Conductivity in region 1
    #     "k2": 0.25,  # Conductivity in region 2
    #     "f1": 0.5,  # Forcing term in region 1
    #     "f2": 0.5   # Forcing term in region 2
    # }
    # problem = PoissonsEquation2D(params)
    # problem.setup_matrix()
    # solution = problem.solve()
    # problem.plot_solution(solution)


    # #### 1D Homo:
    # params = {
    #     "length": 1.0,     # Length of the domain
    #     "k": 1.0,          # Conductivity
    #     "source": 1.0,     # Forcing term
    #     "num_points": 101  # Number of grid points
    # }

    # problem = HomogeneousPoissonsEquation1D(params)
    # problem.setup_matrix()
    # solution = problem.solve()
    # problem.plot_solution(solution)


    # #### 1D Hetero:
    # params = {
    #     "length": 1.0,
    #     "interface": 0.4,
    #     "k1": 1.0,
    #     "k2": 0.2,
    #     "f1": 1.0,
    #     "f2": 1.0,
    #     "num_points": 101
    # }

    # problem = PoissonsEquation1D(params)
    # problem.setup_matrix()
    # solution = problem.solve()
    # problem.plot_solution(solution)

    ### 1D multiple interface
    params = {
    "length": 1.0,
    "interfaces": [0.2, 0.4, 0.6, 0.8],
    "k_values": [1.0, 0.25, 0.9, 0.1, 0.8],
    "f_values": [1,1,1,1,1],
    "num_points": 201
    }

    problem = PoissonsEquation1D_MultipleInterfaces(params)
    problem.setup_matrix()
    solution = problem.solve()
    problem.plot_solution(solution)