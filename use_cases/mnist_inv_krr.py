import sys
import time
import logging
import os
logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import gzip
import numpy as np
import tools  # gofmm shared lib stuff
from math import sqrt
import time
import sys
sys.path.insert(1, '../python')

from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from tensorflow import keras

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

# Loading the MNIST dataset
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Preprocessing the data
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255

# Flattening the images
X_train = x_train.reshape((x_train.shape[0], -1))
X_test = x_test.reshape((x_test.shape[0], -1))

# Reducing dataset size for testing purposes
# Set problem size
problem_size = int(os.getenv('PROBLEM_SIZE', 8192))  # default if not set
X_train = X_train[:problem_size]
y_train = y_train[:problem_size]

# Initialize KernelRidge with Gaussian (RBF) kernel
krr = KernelRidge(kernel='rbf', gamma=0.1)

# Fit the model
krr.fit(X_train, y_train)

# Calculate the Gaussian kernel matrix using pairwise_kernels
K = pairwise_kernels(X_train, metric='rbf', gamma=0.1)

# Regularization parameter
alpha = 0.1

# Parameters for GOFMM
executable = "./test_gofmm"
max_leaf_node_size = int(problem_size / 2)
num_of_neighbors = 0
max_off_diagonal_ranks = int(problem_size / 2)
num_rhs = 1
user_tolerance = 1E-5
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
K_reg_inv = np.linalg.inv(K_reg)
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
