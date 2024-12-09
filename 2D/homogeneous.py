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
        "length": 1.0,  # Domain size
        "num_points": 101,  # Number of points in each dimension
        "k1": 1.0,  # Conductivity in region 1
        "k2": 0.25,  # Conductivity in region 2
        "f1": 0.5,  # Forcing term in region 1
        "f2": 0.5   # Forcing term in region 2
    }

problem = PoissonsEquation2D(params)
problem.setup_matrix()
solution = problem.solve()
problem.plot_solution(solution)
