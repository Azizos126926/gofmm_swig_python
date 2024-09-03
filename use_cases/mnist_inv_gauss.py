import os
import matplotlib.pyplot as plt
import numpy as np
import tools  # gofmm shared lib stuff
from math import sqrt
from scipy.linalg import inv
import time
import sys
sys.path.insert(1, '../python')

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from tensorflow.keras.datasets import mnist

random_state = 42

# GOFMM Kernel inverse class
class Inverse_calculator:
    def __init__(self, executable, problem_size, max_leaf_node_size, num_of_neighbors, max_off_diagonal_ranks, num_rhs, user_tolerance, computation_budget,
                 distance_type, matrix_type, kernel_type, spd_matrix):
        self.executable = executable
        self.problem_size = problem_size
        self.max_leaf_node_size = max_leaf_node_size
        self.num_of_neighbors = num_of_neighbors
        self.max_off_diagonal_ranks = max_off_diagonal_ranks
        self.num_rhs = num_rhs
        self.user_tolerance = user_tolerance
        self.computation_budget = computation_budget
        self.distance_type = distance_type
        self.matrix_type = matrix_type
        self.kernel_type = kernel_type
        self.spd_matrix = np.float32(spd_matrix)  # from input

        # Construct a fix spd matrix and load it into SPDMATRIX_DENSE structure
        self.denseSpd = tools.LoadDenseSpdMatrixFromConsole(self.spd_matrix)
        self.matrix_length = self.problem_size * self.problem_size

    def matinv(self, lambda_inv):
        gofmmCalculator = tools.GofmmTree(self.executable, self.problem_size,
                                          self.max_leaf_node_size,
                                          self.num_of_neighbors, self.max_off_diagonal_ranks, self.num_rhs,
                                          self.user_tolerance, self.computation_budget,
                                          self.distance_type, self.matrix_type,
                                          self.kernel_type, self.denseSpd)
        c = gofmmCalculator.InverseOfDenseSpdMatrix(lambda_inv, self.matrix_length)
        print("GOFMM Inverse passed")

        # Resize inverse to be n*n matrix
        inv_matrix = np.resize(c, (self.problem_size, self.problem_size))
        return inv_matrix


    def compute_rse(self, matExp, matThe):
        return np.linalg.norm(matExp - matThe) / sqrt(np.sum(matThe ** 2)) * 100

# Load MNIST data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Preprocess data
x_train = x_train.reshape(x_train.shape[0], -1) / 255.0
x_test = x_test.reshape(x_test.shape[0], -1) / 255.0

# Set problem size
problem_size = int(os.getenv('PROBLEM_SIZE', 1024))  # default if not set

# Use a subset of the data for quicker computation
x_train, _, y_train, _ = train_test_split(x_train, y_train, train_size=problem_size, stratify=y_train, random_state=random_state)
x_test, _, y_test, _ = train_test_split(x_test, y_test, test_size=1024, stratify=y_test, random_state=random_state)

# Define kernel
kernel_standard = 1.0 * RBF(length_scale=1.0)

# Standard GP with scikit-learn
gp_standard = GaussianProcessRegressor(kernel=kernel_standard, alpha=0.1)
gp_standard.fit(x_train, y_train)
mu_star_sklearn, std_star_sklearn = gp_standard.predict(x_test, return_std=True)

# Parameters for GOFMM
executable = "./test_gofmm"
max_leaf_node_size = int(problem_size / 2)
num_of_neighbors = 0
max_off_diagonal_ranks = int(problem_size / 2)
num_rhs = 10
user_tolerance = 1E-5
computation_budget = 0.00
distance_type = "kernel"
matrix_type = "dense"
kernel_type = "gaussian"
lambda_inv = 1  # regularization parameter

# Compute the kernel matrix
kernel_matrix = kernel_standard(x_train, x_train)
print(kernel_matrix.shape)
kernel_matrix = kernel_matrix.astype("float32")

start_time = time.time()
inverse_GOFMM_obj = Inverse_calculator(executable, problem_size, max_leaf_node_size,
                                       num_of_neighbors, max_off_diagonal_ranks, num_rhs, user_tolerance, computation_budget,
                                       distance_type, matrix_type, kernel_type, kernel_matrix)

# Inverse kernel using GOFMM
inv_gofmm = inverse_GOFMM_obj.matinv(lambda_inv)
end_time = time.time()
execution_time_invGOFMM = end_time - start_time
start_time = time.time()
inv_spd = inv(kernel_matrix + lambda_inv * np.eye(problem_size))
end_time = time.time()
execution_time_invNumpy = end_time - start_time
rse = inverse_GOFMM_obj.compute_rse(inv_gofmm, inv_spd)
print(f"Relative Standard Error: {rse:.4e}")
print("\n")
print("-----------------------------------------------------------")

# Compute kernel evaluations between test and training points
k_star = kernel_standard(x_train, x_test)
k_star_star = kernel_standard(x_test, x_test)

# Compute predictive mean
mu_star = k_star.T @ inv_gofmm @ y_train
mu_star_np = k_star.T @ inv_spd @ y_train
sigma_star = k_star_star - (k_star.T @ inv_gofmm @ k_star)

print("Problem size =", problem_size)
print("\n")
print("-----------------------------------------------------------")
print("Execution time: {:.6f} seconds for global GOFMM Inverse".format(execution_time_invGOFMM))
print("\n")
print("-----------------------------------------------------------")
print("Execution time: {:.6f} seconds for global NUMPY Inverse".format(execution_time_invNumpy))
print("\n")

# Evaluate the model
mse_gofmm = mean_squared_error(y_test, mu_star)
mse_np = mean_squared_error(y_test, mu_star_np)
print(f"Mean Squared Error using GOFMM-inverted kernel: {mse_gofmm:.4f}")
print(f"Mean Squared Error using numpy-inverted kernel: {mse_np:.4f}")
