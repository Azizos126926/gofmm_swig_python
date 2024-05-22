import matplotlib.pyplot as plt
import numpy as np
import tools  # gofmm shared lib stuff
from math import sqrt
import time
import sys
sys.path.insert(1, '../python')

#import Gaussian processes tools
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

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

data_count=10240
#  Generate synthetic data
X = np.random.rand(data_count,1)  # Example: 100 samples, 1 feature
y = np.sin(3 * X).ravel() + 0.1 * np.random.randn(data_count)

#  Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#  Train and evaluate GP models
kernel_standard = 1.0 * RBF(length_scale=1.0)  # Standard RBF kernel
gp_standard = GaussianProcessRegressor(kernel=kernel_standard, alpha=0.1)  # Standard GP with scikit-learn
gp_standard.fit(X_train, y_train)
mu_star_sklearn, std_star_sklearn = gp_standard.predict(X_test, return_std=True)

#y_pred_standard, std_standard = gp_standard.predict(X_test, return_std=True)
#mse_standard = mean_squared_error(y_test, y_pred_standard)

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
lambda_inv= 0.01 # regularization parameter

# Compute the kernel matrix
kernel_matrix = kernel_standard(X_train, X_train)
print(kernel_matrix.shape)
# Compute the Kernel with noise
kernel_matrix = kernel_matrix.astype("float32")
start_time = time.time()
inverse_GOFMM_obj = Inverse_calculator( executable, problem_size, max_leaf_node_size,
                                        num_of_neighbors, max_off_diagonal_ranks, num_rhs, user_tolerance, computation_budget,
                                       	distance_type, matrix_type, kernel_type, kernel_matrix)

# INVERSE KERNEL using GOFMM
inv_gofmm= inverse_GOFMM_obj.matinv(lambda_inv)
end_time = time.time()
execution_time_invGOFMM = end_time - start_time
inv_spd= np.linalg.inv(kernel_matrix+lambda_inv * np.eye(problem_size))
rse= inverse_GOFMM_obj.compute_rse(inv_gofmm,inv_spd)
print(f"Relative Standard Error: {rse:.4e}")
print("\n")
print("-----------------------------------------------------------")
# Compute kernel evaluations between test and training points
k_star = kernel_standard(X_train, X_test)
k_star_star= kernel_standard(X_test, X_test)
# Compute predictive mean
mu_star = k_star.T @ inv_gofmm @ y_train
mu_star_np=k_star.T @ inv_spd @ y_train
sigma_star= k_star_star - (k_star.T @ inv_gofmm @ k_star)
print("Problem size =", problem_size)
print("\n")
print(f"mean GOFMM diff with GP sklearn: { inverse_GOFMM_obj.compute_rse(mu_star,mu_star_sklearn):.4e}")
print("\n")
print("-----------------------------------------------------------")
print(f"mean np inverse diff with GP sklearn: { inverse_GOFMM_obj.compute_rse(mu_star_np,mu_star_sklearn):.4e}")
print("\n")
print("-----------------------------------------------------------")
print("Execution time: {:.6f} seconds for global GOFMM Inverse".format(execution_time_invGOFMM))
print("\n")
#print("covariance=", sigma_star)

# Step 5: Analyze and compare results
#print("MSE (Standard GP):", mse_standard)
