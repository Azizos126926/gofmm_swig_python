import matplotlib.pyplot as plt
import numpy as np
import tools  # gofmm shared lib stuff
from math import sqrt
import time
import sys
sys.path.insert(1, '../python')

from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics.pairwise import pairwise_kernels



random_state = 42
#GOFMM Kernel inverse class 
class Inverse_calculator:
    def __init__( self, executable, problem_size, max_leaf_node_size, num_of_neighbors, max_off_diagonal_ranks, num_rhs, user_tolerance, computation_budget,
                distance_type, matrix_type, kernel_type, spd_matrix):
        #class parameters
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
        self.spd_matrix = np.float32( spd_matrix )  # from input

        # Construct a fix spd matrix and load it into SPDMATRIX_DENSE structure
        self.denseSpd = tools.LoadDenseSpdMatrixFromConsole( self.spd_matrix )
        self.matrix_length=self.problem_size * self.problem_size
        
    def matinv( self , lambda_inv):
        """The operation of matrix inverse. The calculation is based
        on updated instance attributes. Inverse using GOFMM tree.

        @ret: multi-dimensional numpy array
        """
        #Create GOFMM tree from SPD matrix
        gofmmCalculator = tools.GofmmTree( self.executable, self.problem_size,
		                                  self.max_leaf_node_size,
		                                  self.num_of_neighbors, self.max_off_diagonal_ranks, self.num_rhs,
		                                  self.user_tolerance, self.computation_budget,
		                                  self.distance_type, self.matrix_type,
		                                  self.kernel_type, self.denseSpd )
        # Use of GOFMM function inverse  
        c=gofmmCalculator.InverseOfDenseSpdMatrix(lambda_inv, self.matrix_length)
        print("GOFMM Inverse passed")
        
        #resize inverse to be n*n matrix
        inv_matrix=np.resize( c, ( self.problem_size, self.problem_size ) )
        return inv_matrix
    
    def compute_rse(self, matExp, matThe):
        """Compute the relative standard error of the experimental matrix
        gofmm against the benchmark, matThe.

        @matExp[numpy.ndarray]: experimental result

        @matThe[numpy.ndarray]: theoretical result

        @ret: relative std error (unit: in percent)
        """
        return np.linalg.norm(matExp - matThe) / sqrt(np.sum(matThe ** 2)) * 100
    
# Define a nonlinear function for the data
def nonlinear_function(x):
    return np.sin(3 * x).ravel() + 0.5 * np.exp(-0.1 * x) + 0.1 * np.random.randn(len(x))

#Problem size
data_count=16384
# Generate random data
np.random.seed(0)
X_train = np.random.rand(data_count, 1) * 10  # Random values between 0 and 10
y_train = nonlinear_function(X_train)

# Initialize KernelRidge with Gaussian (RBF) kernel
krr = KernelRidge(kernel='rbf', gamma=0.1)

# Fit the model
krr.fit(X_train, y_train)

# Calculate the gaussian kernel matrix using pairwise
K = pairwise_kernels(X_train, metric='rbf', gamma=0.1)

# Regularization parameter
alpha = 0.1


#parameters
executable = "./test_gofmm"
problem_size = X_train.shape[0]
max_leaf_node_size = int(problem_size/2)
num_of_neighbors = 0
max_off_diagonal_ranks = int(problem_size/2)
num_rhs = 1
user_tolerance = 1E-3 
computation_budget = 0.00
distance_type = "kernel"
matrix_type = "dense"
kernel_type = "gaussian"
rng = np.random.default_rng( random_state )
lambda_inv= 1.0 # regularization parameter

# prepare inverse GOFMM caculator
kernel_matrix = K.astype("float32")
start_time = time.time()
inverse_GOFMM_obj = Inverse_calculator( executable, problem_size, max_leaf_node_size,
                                        num_of_neighbors, max_off_diagonal_ranks, num_rhs, user_tolerance, computation_budget,
                                       	distance_type, matrix_type, kernel_type, K)

# INVERSE KERNEL using GOFMM
inv_gofmm= inverse_GOFMM_obj.matinv(lambda_inv)
end_time = time.time()
execution_time_invGOFMM = end_time - start_time
# Compute the inverse of the regularized kernel matrix numpy inverse
# Compute the regularized kernel matrix
start_time = time.time()
K_reg = K + lambda_inv * np.eye(len(X_train))
K_reg_inv = np.linalg.inv(K_reg)
end_time = time.time()
execution_time_invNumpy = end_time - start_time

# Calculate the weights
weights_np = np.dot(K_reg_inv, y_train)
weights_gofmm = np.dot(inv_gofmm, y_train)
print(weights_gofmm.shape)
# Get the learned weights of the SKLEARN 
weights = krr.dual_coef_
# Compute rse of inverse 
rse= inverse_GOFMM_obj.compute_rse(inv_gofmm,K_reg_inv)
#Prints
print("Problem size =", problem_size)
print("\n")
print(f"Relative Standard Error of the weights computed using GOFMM inverse: { inverse_GOFMM_obj.compute_rse(weights_gofmm,weights):.4e}")
print("\n")
print("-----------------------------------------------------------")
print(f"Relative Standard Error of the inverse kernel calculation compared to (np.linalg.in): {rse:.4e}")
print("\n")
print("-----------------------------------------------------------")
print("Execution time: {:.6f} seconds for global GOFMM Inverse".format(execution_time_invGOFMM))
print("\n")
print("-----------------------------------------------------------")
print("Execution time: {:.6f} seconds for numpy Inverse".format(execution_time_invNumpy))