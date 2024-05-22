import matplotlib.pyplot as plt
import numpy as np
import tools  # gofmm shared lib stuff
from math import sqrt
import time
import sys
sys.path.insert(1, '../python')


# NOTE: make sure "path/to/datafold" is in sys.path or PYTHONPATH if not ,!installed
import datafold.dynfold as dfold
import datafold.pcfold as pfold
from datafold.utils.plot import plot_pairwise_eigenvector

random_state = 42

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
        
    def modify_parameter(self, paraIdx, paraVal): #This function could be used to test many cases (tolerances, kerneltypes, spdSizes..) in the same run
        """Modify a gofmm parameter according to parameter index and its value.

        @paraIdx[int]: each index corresponds to a parameter in the file.
        Please check the mapping in the file `build/README.md`

        @paraVal[int]: value of this parameter

        @ret: parameters as object fields will be modified
        """
        # # # This function could be used to test many cases (tolerances, kerneltypes, spdSizes..) in the same run by modifying a specific parameters
        
        if (paraIdx == 0):
            self.executable = paraVal
        elif (paraIdx == 1):
            self.problem_size = paraVal
        elif (paraIdx == 2):
            self.max_leaf_node_size = paraVal
        elif (paraIdx == 3):
            self.num_of_neighbors = paraVal
        elif (paraIdx == 4):
            self.max_off_diagonal_ranks = paraVal
        elif (paraIdx == 5):
            self.num_rhs = paraVal
        elif (paraIdx == 6):
            self.user_tolerance = paraVal
        elif (paraIdx == 7):
            self.computation_budget = paraVal
        elif (paraIdx == 8):
            self.distance_type = paraVal
        elif (paraIdx == 9):
            self.matrix_type = paraVal
        elif (paraIdx == 10):
            self.kernel_type = paraVal
        else:
            raise ValueError("The requested parameter index is invalid as it is not equal to any integer from 0 to 10 \n")

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

    
#parameters
executable = "./test_gofmm"
problem_size = 4096
max_leaf_node_size = 2048
num_of_neighbors = 0
max_off_diagonal_ranks = 2048
num_rhs = 1
user_tolerance = 1E-5
computation_budget = 0.00
distance_type = "kernel"
matrix_type = "dense"
kernel_type = "gaussian"
rng = np.random.default_rng( random_state )
lambda_inv= 5.0  # regularization parameter

#diffusionMaps
data = rng.uniform( low = ( -2, -1 ), high = ( 2, 1 ), size = ( problem_size, 2 ) )
pcm = pfold.PCManifold( data )
pcm.optimize_parameters()
dmap = dfold.DiffusionMaps(
    kernel=pfold.GaussianKernel(epsilon=pcm.kernel.epsilon),
    n_eigenpairs=5,
    dist_kwargs=dict(cut_off=pcm.cut_off),
)
dmap.fit(pcm, store_kernel_matrix=True)

K = np.ones((problem_size,problem_size),dtype=np.float32) #dmap.kernel_matrix_
#K = dmap.kernel_matrix_ #generateSPD with diffusionMaps
#K_sparse = K.copy()
#K = K.todense()
K = K.astype("float32")

#execution inverse numpy 
#start time measure
start_time = time.time()
inv_spd= np.linalg.inv(K + lambda_inv * np.eye(problem_size))
#end time measure
end_time = time.time()
execution_time_invNumpy = end_time - start_time

#execution inverse GOFMM 
start_time = time.time()
inverse_GOFMM_obj = Inverse_calculator( executable, problem_size, max_leaf_node_size,
                                        num_of_neighbors, max_off_diagonal_ranks, num_rhs, user_tolerance, computation_budget,
                                       	distance_type, matrix_type, kernel_type, K)
#start time measure
#start_time = time.time()
inv_gofmm= inverse_GOFMM_obj.matinv(lambda_inv)
#end time measure
end_time = time.time()
execution_time_invGOFMM = end_time - start_time

#calculate RSE
rse= inverse_GOFMM_obj.compute_rse(inv_gofmm,inv_spd)

#Prints
print("\n")
print("-----------------------------------------------------------")
print("Execution time: {:.6f} seconds for numpy Inverse".format(execution_time_invNumpy))
print("\n")
print("-----------------------------------------------------------")
print("Execution time: {:.6f} seconds for global GOFMM Inverse".format(execution_time_invGOFMM))
print("\n")
print("-----------------------------------------------------------")
print(f"Relative Standard Error: {rse:.4e}")
print("\n")
print("-----------------------------------------------------------")

