import matplotlib.pyplot as plt
import os
import numpy as np
import tools  # gofmm shared lib stuff
from math import sqrt
from scipy.linalg import inv
import time
import sys
sys.path.insert(1, '../python')

from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split

# GOFMM Kernel inverse class
class Inverse_calculator:
    def __init__(self, executable, problem_size, max_leaf_node_size, num_of_neighbors, max_off_diagonal_ranks, num_rhs, user_tolerance, computation_budget,
                 distance_type, matrix_type, kernel_type, spd_matrix):
        """
        Initializes the GOFMM inverse calculator with the given parameters.
        :param executable: Path to the GOFMM executable.
        :param problem_size: Size of the problem (number of data points).
        :param max_leaf_node_size: Maximum size of leaf nodes in the GOFMM tree.
        :param num_of_neighbors: Number of neighbors for the GOFMM algorithm.
        :param max_off_diagonal_ranks: Maximum ranks for off-diagonal blocks.
        :param num_rhs: Number of right-hand sides (for multiple output regression).
        :param user_tolerance: User-defined tolerance for the approximation.
        :param computation_budget: Computation budget for the GOFMM algorithm.
        :param distance_type: Type of distance metric used.
        :param matrix_type: Type of matrix (dense or sparse).
        :param kernel_type: Type of kernel function used.
        :param spd_matrix: Symmetric positive definite matrix for inversion.
        """
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
        # Create GOFMM tree from SPD matrix
        gofmmCalculator = tools.GofmmTree(self.executable, self.problem_size,
                                          self.max_leaf_node_size,
                                          self.num_of_neighbors, self.max_off_diagonal_ranks, self.num_rhs,
                                          self.user_tolerance, self.computation_budget,
                                          self.distance_type, self.matrix_type,
                                          self.kernel_type, self.denseSpd)
        # Use GOFMM function inverse
        c = gofmmCalculator.InverseOfDenseSpdMatrix(lambda_inv, self.matrix_length)
        print("GOFMM Inverse passed")

        # Resize inverse to be n*n matrix
        inv_matrix = np.resize(c, (self.problem_size, self.problem_size))
        return inv_matrix

    def compute_rse(self, matExp, matThe):
        return np.linalg.norm(matExp - matThe) / sqrt(np.sum(matThe ** 2)) * 100

# Data generation functions
def generate_uniform_data(size, low=(-2, -1), high=(2, 1)):
    rng = np.random.default_rng(random_state)
    return rng.uniform(low=low, high=high, size=(size, 2))

def generate_gaussian_data(size, mean=[0, 0], cov=[[1, 0.5], [0.5, 1]]):
    rng = np.random.default_rng(random_state)
    return rng.multivariate_normal(mean, cov, size)

def generate_circular_data(size):
    rng = np.random.default_rng(random_state)
    radius = rng.uniform(0, 1, size)
    angle = rng.uniform(0, 2 * np.pi, size)
    x = radius * np.cos(angle)
    y = radius * np.sin(angle)
    return np.vstack((x, y)).T

def generate_mixture_of_gaussians(size, centers, cluster_std):
    from sklearn.datasets import make_blobs
    return make_blobs(n_samples=size, centers=centers, cluster_std=cluster_std, random_state=random_state)[0]

def generate_high_dimensional_uniform_data(size, dimensions):
    rng = np.random.default_rng(random_state)
    return rng.uniform(low=-1, high=1, size=(size, dimensions))

# Define a nonlinear function for the data
def nonlinear_function(x):
    return np.sin(3 * x).ravel() + 0.5 * np.exp(-0.1 * x) + 0.1 * np.random.randn(len(x))

# Random state
random_state = 42

# Choose data generation method
data_count = int(os.getenv('PROBLEM_SIZE', 4096))
data_choice = 'gaussian'  # Options: 'boston', 'uniform', 'gaussian', 'circular', 'mixture', 'high_dim'

if data_choice == 'uniform':
    X_train = generate_uniform_data(data_count)
    y_train = nonlinear_function(X_train[:, 0])  # Using only the first feature for the nonlinear function
elif data_choice == 'gaussian':
    X_train = generate_gaussian_data(data_count)
    y_train = nonlinear_function(X_train[:, 0])  # Using only the first feature for the nonlinear function
elif data_choice == 'circular':
    X_train = generate_circular_data(data_count)
    y_train = nonlinear_function(X_train[:, 0])  # Using only the first feature for the nonlinear function
elif data_choice == 'mixture':
    X_train = generate_mixture_of_gaussians(data_count, centers=[(-2, -2), (2, 2), (-2, 2), (2, -2)], cluster_std=[0.5, 0.5, 0.5, 0.5])
    y_train = nonlinear_function(X_train[:, 0])  # Using only the first feature for the nonlinear function
elif data_choice == 'high_dim':
    X_train = generate_high_dimensional_uniform_data(data_count, dimensions=10)
    y_train = nonlinear_function(X_train[:, 0])  # Using only the first feature for the nonlinear function

# Initialize KernelRidge with Gaussian (RBF) kernel
krr = KernelRidge(kernel='rbf', gamma=0.1)

# Fit the model
krr.fit(X_train, y_train)

# Calculate the Gaussian kernel matrix using pairwise_kernels
K = pairwise_kernels(X_train, metric='rbf', gamma=0.1)

# Save the SPD matrix to a .BIN file
spd_matrix_file = "spd_matrix4096_2048_128_2048.bin"
K.astype(np.float32).tofile(spd_matrix_file)

print(f"SPD matrix saved to {spd_matrix_file}")

# Regularization parameter
alpha = 0.1

# Parameters
executable = "./test_gofmm"
problem_size = X_train.shape[0]
max_leaf_node_size = int(problem_size / 2)
num_of_neighbors = 0
max_off_diagonal_ranks = int(problem_size / 2)
num_rhs = 1
user_tolerance = 1E-3
computation_budget = 0.00
distance_type = "kernel"
matrix_type = "dense"
kernel_type = "gaussian"
lambda_inv = 1.0  # regularization parameter

# Prepare inverse GOFMM calculator
kernel_matrix = K.astype("float32")
start_time = time.time()
inverse_GOFMM_obj = Inverse_calculator(executable, problem_size, max_leaf_node_size,
                                       num_of_neighbors, max_off_diagonal_ranks, num_rhs, user_tolerance, computation_budget,
                                       distance_type, matrix_type, kernel_type, K)

# INVERSE KERNEL using GOFMM
inv_gofmm = inverse_GOFMM_obj.matinv(lambda_inv)
end_time = time.time()
execution_time_invGOFMM = end_time - start_time

# Compute the inverse of the regularized kernel matrix using numpy
start_time = time.time()
K_reg = K + lambda_inv * np.eye(len(X_train))
K_reg_inv = inv(K_reg)
end_time = time.time()
execution_time_invNumpy = end_time - start_time

# Calculate the weights
weights_np = np.dot(K_reg_inv, y_train)
weights_gofmm = np.dot(inv_gofmm, y_train)

# Get the learned weights of the SKLEARN
weights = krr.dual_coef_

# Compute RSE of inverse
rse = inverse_GOFMM_obj.compute_rse(inv_gofmm, K_reg_inv)


# Print results
print("Problem size =", problem_size)
print("\n")
print(f"Relative Standard Error of the weights computed using GOFMM inverse: {inverse_GOFMM_obj.compute_rse(weights_gofmm, weights):.4e}")
print("\n")
print("-----------------------------------------------------------")
print(f"Relative Standard Error of the inverse kernel calculation compared to (np.linalg.inv): {rse:.4e}")
print("\n")
print("-----------------------------------------------------------")
print("Execution time: {:.6f} seconds for global GOFMM Inverse".format(execution_time_invGOFMM))
print("\n")
print("-----------------------------------------------------------")
print("Execution time: {:.6f} seconds for numpy Inverse".format(execution_time_invNumpy))
