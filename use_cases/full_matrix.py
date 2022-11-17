import numpy as np
import tools
from scipy.sparse.linalg import LinearOperator

class FullMatrix( LinearOperator ):
    def __init__( self, 
                executable, 
                problem_size, 
                max_leaf_node_size, 
                num_of_neighbors, 
                max_off_diagonal_ranks, 
                num_rhs, user_tolerance, 
                computation_budget,
                distance_type, 
                matrix_type, 
                kernel_type, 
                spd_matrix, 
                weight, 
                dtype="float32" ):
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

        self.spd_matrix = np.float32( spd_matrix )  
        self.denseSpd = tools.LoadDenseSpdMatrixFromConsole( self.spd_matrix )
        self.weight = np.float32( weight )  
        self.wData = tools.LoadNumpyMatrixFromConsole( self.weight )
        self.lenMul = self.problem_size * self.num_rhs
        self.shape = self.spd_matrix.shape
        self.dtype = np.dtype( dtype )

    def _matvec( self, matrix ):
        gofmmCalculator = tools.GofmmTree( self.executable, 
                                        self.problem_size,
		                                self.max_leaf_node_size,
		                                self.num_of_neighbors, 
                                        self.max_off_diagonal_ranks, 
                                        self.num_rhs,
		                                self.user_tolerance, 
                                        self.computation_budget,
		                                self.distance_type, 
                                        self.matrix_type,
		                                self.kernel_type, 
                                        self.denseSpd )

        flattened_matrix = matrix.reshape( self.problem_size, 1 )
        weightData = tools.LoadNumpyMatrixFromConsole( np.float32( flattened_matrix ) )
        spdmatrix = gofmmCalculator.MultiplyDenseSpdMatrix( self.wData, self.lenMul )
        spdmatrix = np.resize( spdmatrix, ( self.problem_size, self.num_rhs ) )
        return spdmatrix
