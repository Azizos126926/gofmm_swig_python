-------------------#distributed_inverse.cpp--------------------
/**
 *  HMLP (High-Performance Machine Learning Primitives)
 *  
 *  Copyright (C) 2014-2017, The University of Texas at Austin
 *  
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *  
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 *  GNU General Public License for more details.
 *  
 *  You should have received a copy of the GNU General Public License
 *  along with this program. If not, see the LICENSE file.
 *
 **/  

/** GOFMM templates. */
#include <gofmm_mpi.hpp>
/** Use dense SPD matrices. */
#include <containers/SPDMatrix.hpp>
/** use implicit kernel matrices (only coordinates are stored). */
#include <containers/KernelMatrix.hpp>
/** use implicit matrices */
#include <containers/VirtualMatrix.hpp>
/** use implicit PVFMM kernel matrices. */
//#include <containers/PVFMMKernelMatrix.hpp>
/** Use implicit Gauss-Newton (multilevel perceptron) matrices. */
#include <containers/MLPGaussNewton.hpp>
/** Use an OOC covariance matrices. */
#include <containers/OOCCovMatrix.hpp>
/** Use Gauss Hessian matrices provided by Chao. */
#include <containers/GNHessian.hpp>
/** Use STL and HMLP namespaces. */
using namespace std;
using namespace hmlp;



/** @brief Top level driver that reads arguments from the command line. */ 
int main( int argc, char *argv[] )
{
  try
  {
    /** Parse arguments from the command line. */
    gofmm::CommandLineHelper cmd( argc, argv );

    /** MPI (Message Passing Interface): check for THREAD_MULTIPLE support. */
    int  provided;
    mpi::Init_thread( &argc, &argv, MPI_THREAD_MULTIPLE, &provided );
    if ( provided != MPI_THREAD_MULTIPLE )
    {
      printf( "MPI_THREAD_MULTIPLE is not supported\n" ); fflush( stdout );
      exit( 1 );
    }

    /** MPI (Message Passing Interface): create a specific comm for GOFMM. */
    mpi::Comm CommGOFMM;
    mpi::Comm_dup( MPI_COMM_WORLD, &CommGOFMM );

    /** HMLP API call to initialize the runtime. */
    HANDLE_ERROR( hmlp_init( CommGOFMM ) );

    /** Run the matrix file provided by users. */
    if ( !cmd.spdmatrix_type.compare( "dense" ) )
    {
      using T = float;
      /** Dense spd matrix format. */
      SPDMatrix<T> K( cmd.n, cmd.n, cmd.user_matrix_filename );
      /** Launch self-testing routine. */
      mpigofmm::InverseHelper( K, cmd, CommGOFMM );
    }


    /** Generate a kernel matrix from the coordinates. */
    if ( !cmd.spdmatrix_type.compare( "kernel" ) )
    {
      using T = double;
      /** Read the coordinates from the file. */
      DistData<STAR, CBLK, T> X( cmd.d, cmd.n, CommGOFMM, cmd.user_points_filename );
      /** Setup the kernel object. */
      kernel_s<T, T> kernel;
      kernel.type = GAUSSIAN;
      if ( !cmd.kernelmatrix_type.compare( "gaussian" ) ) kernel.type = GAUSSIAN;
      if ( !cmd.kernelmatrix_type.compare(  "laplace" ) ) kernel.type = LAPLACE;
      kernel.scal = -0.5 / ( cmd.h * cmd.h );
      /** Distributed spd kernel matrix format (implicitly create). */
      DistKernelMatrix<T, T> K( cmd.n, cmd.d, kernel, X, CommGOFMM );
      /** Launch self-testing routine. */
      mpigofmm::InverseHelper( K, cmd, CommGOFMM );
    }

    /** Create a random spd matrix, which is diagonal-dominant. */
    if ( !cmd.spdmatrix_type.compare( "testsuit" ) )
    {
      using T = double;
      /** dense spd matrix format */
      SPDMatrix<T> K( cmd.n, cmd.n );
      /** random spd initialization */
      K.randspd( 0.0, 1.0 );
      /** broadcast K to all other rank */
      mpi::Bcast( K.data(), cmd.n * cmd.n, 0, CommGOFMM );
      /** Launch self-testing routine. */
      mpigofmm::InverseHelper( K, cmd, CommGOFMM );
    }

    /** HMLP API call to terminate the runtime */
    HANDLE_ERROR( hmlp_finalize() );
    /** Message Passing Interface */
    mpi::Finalize();
  }
  catch ( const exception & e )
  {
    cout << e.what() << endl;
    return -1;
  }
  return 0;

}; /** end main() */

--------------#gofmm_mpi.hpp--------------
template<typename TREE>
void InverseTesting( TREE &tree)
{
  /** Derive type T from TREE. */
  using T = typename TREE::T;
  /** MPI Support. */
  int rank; mpi::Comm_rank( tree.GetComm(), &rank );
  int size; mpi::Comm_size( tree.GetComm(), &size );
  /** Size of right hand sides. */
  size_t n = tree.getGlobalProblemSize();

  /** Input and output in RIDS and RBLK. */
  DistData<RIDS, STAR, T> w_rids( n, n, tree.getOwnedIndices(), tree.GetComm() );
  /** Initialize with random N( 0, 1 ). */
  w_rids.randn();

  if ( !tree.setup.SecureAccuracy() )
  {
    /** Factorization */
    T lambda = 1.0;
    mpigofmm::DistFactorize( tree, lambda ); 
    mpigofmm::ComputeErrorInverse(tree, lambda, w_rids);
  }
}; /** end SelfTesting() */
/** @brief Instantiate the splitters here. */ 
template<typename SPDMATRIX>
void InverseHelper( SPDMATRIX &K, gofmm::CommandLineHelper &cmd, mpi::Comm CommGOFMM )
{
  using T = typename SPDMATRIX::T;
  const int N_CHILDREN = 2;
  /** Use geometric-oblivious splitters. */
  using SPLITTER     = mpigofmm::centersplit<SPDMATRIX, N_CHILDREN, T>;
  using RKDTSPLITTER = mpigofmm::randomsplit<SPDMATRIX, N_CHILDREN, T>;
  /** GOFMM tree splitter. */
  SPLITTER splitter( K );
  splitter.Kptr = &K;
  splitter.metric = cmd.metric;
  /** Randomized tree splitter. */
  RKDTSPLITTER rkdtsplitter( K );
  rkdtsplitter.Kptr = &K;
  rkdtsplitter.metric = cmd.metric;
	/** Create configuration for all user-define arguments. */
  gofmm::Configuration<T> config( cmd.metric, 
      cmd.n, cmd.m, cmd.k, cmd.s, cmd.stol, cmd.budget, cmd.secure_accuracy );
  /** (Optional) provide neighbors, leave uninitialized otherwise. */
  DistData<STAR, CBLK, pair<T, size_t>> NN( 0, cmd.n, CommGOFMM );
  /** Compress matrix K. */
  auto *tree_ptr = mpigofmm::Compress( K, NN, splitter, rkdtsplitter, config, CommGOFMM );
  auto &tree = *tree_ptr;

  /** Examine accuracies. */
  mpigofmm::InverseTesting( tree );

	/** Delete tree_ptr. */
  delete tree_ptr;
}; /** end test_gofmm_setup() */



-----------#igofmm_mpi.hpp---------------
template<typename TREE, typename T>
void ComputeErrorInverse( TREE &tree, T lambda, Data<T> weights )
{
  using NODE    = typename TREE::NODE;
  using MPINODE = typename TREE::MPINODE;

  auto comm = tree.GetComm();
  auto size = tree.GetCommSize();
  auto rank = tree.GetCommRank();
  MPI_Win win;
  int lock = 0; // Shared lock variable

  printf( "[BEG] COMPUTE ERROR\n" );
  size_t n    = weights.row();
  size_t nrhs = weights.col();


  /** Shift lambda and make it a column vector. */
  printf( "[BEG] IDENTITY\n" );
  printf("prob_size_locally = %zu\n", n);
  printf("prob_size_general = %zu\n", nrhs);

  Data<T> rhs( n, nrhs );
  printf( "[BEG] LOOP IDENTITY\n" );
  for ( size_t j = 0; j < nrhs; j ++ ){
      for ( size_t i = 0; i < n; i ++ )
      rhs( i, j ) = 0.0;
  }
  for ( size_t i = 0; i < n; i ++ )
   {rhs( i, i ) = 1.0;}
  printf( "[END] LOOP IDENTITY\n" );

  /** Solver. */
  DistSolve( tree, rhs );
   // Create an MPI window for the lock variable
  MPI_Win_create(&lock, sizeof(lock), sizeof(lock), MPI_INFO_NULL, MPI_COMM_WORLD, &win);

    // Barrier to synchronize all processes before starting
  MPI_Barrier(MPI_COMM_WORLD);

    // Ensure mutual exclusion
  int entered_critical_section = 0;
  while (!entered_critical_section) {
        // Lock the window for exclusive access
      MPI_Win_lock(MPI_LOCK_EXCLUSIVE, 0, 0, win);
        
        // Check and set the lock
      if (lock == 0) {
            lock = rank + 1; // Set the lock to the rank that is entering the critical section
            entered_critical_section = 1; // Indicate that this rank has entered the critical section
            /** PRINT RESULTS PER RANK */
            printf( "[BEG] INVERSE PRINT RANK= %d\n", rank );
            for ( size_t j = 0; j < nrhs; j+=100 ){
                for ( size_t i = 0; i < n; i+=100 )
                printf("rhs(%zu, %zu) = %f\n", i, j, rhs(i,j));  }
            printf( "[END] INVERSE PRINT RANK= %d\n", rank );
        }
        
        // Unlock the window
      MPI_Win_unlock(0, win);
        
        // Wait for the lock to be released if this rank did not set it
      if (!entered_critical_section) {
            // Synchronize all processes to ensure that we wait for the next turn
            MPI_Barrier(MPI_COMM_WORLD);
        }
    }

    // Critical section
  printf("Rank %d is in the critical section\n", rank);

    // Exit critical section
    // Lock the window for exclusive access
  MPI_Win_lock(MPI_LOCK_EXCLUSIVE, 0, 0, win);
    
    // Release the lock
  lock = 0; // Indicate that the critical section is free now
    
    // Unlock the window
  MPI_Win_unlock(0, win);
    
    // Barrier to ensure all ranks exit the critical section before proceeding
  MPI_Barrier(MPI_COMM_WORLD);

  // Free the MPI window
  MPI_Win_free(&win);

};

