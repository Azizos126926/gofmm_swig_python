/**
 *  HMLP (High-Performance Machine Learning Primitives)
 *  
 *  Copyright (C) 2014-2018, The University of Texas at Austin
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

/** Use MPI-GOFMM templates. */
#include <gofmm_mpi.hpp>
/** Use dense SPD matrices. */
#include <containers/SPDMatrix.hpp>
/** Use implicit kernel matrices (only coordinates are stored). */
#include <containers/KernelMatrix.hpp>
/** Use STL and HMLP namespaces. */
using namespace std;
using namespace hmlp;

/** 
 *  @brief In this example, we explain how you can compress generic
 *         SPD matrices and kernel matrices using MPIGOFMM. 
 */ 
int main( int argc, char *argv[] )
{
  try
  {
     /** Parse arguments from the command line. */
    gofmm::CommandLineHelper cmd( argc, argv );
    /** Use float as data type. */
    using T = float;
    /** Regularization for the system (K+lambda*I). */
    T lambda = 1.0;

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
    int comm_rank, comm_size;
    mpi::Comm_size( CommGOFMM, &comm_size );
    mpi::Comm_rank( CommGOFMM, &comm_rank );
    /** [Step#0] HMLP API call to initialize the runtime. */
    HANDLE_ERROR( hmlp_init( &argc, &argv, CommGOFMM ) );
    /** Run the matrix file provided by users. */
    // TODO //
    if ( !cmd.spdmatrix_type.compare( "dense" ) )
    {      /** Dense spd matrix format. */
      SPDMatrix<T> K( cmd.n, cmd.n, cmd.user_matrix_filename );
      /** Launch self-testing routine. */
      //SAME AS THE NEXT CONTENT WE WILL MAKE A FUNCTION FOR THAT//
    }
     /** Create a random spd matrix, which is diagonal-dominant. */
    if ( !cmd.spdmatrix_type.compare( "testsuit" ) )
    {
      /** dense spd matrix format */
      SPDMatrix<T> K( cmd.n, cmd.n );
      /** random spd initialization */
      K.randspd( 0.0, 1.0 );
      /** broadcast K to all other rank */
      mpi::Bcast( K.data(), cmd.n * cmd.n, 0, CommGOFMM );
      size_t n= cmd.n;
      /** [Step#1] Create a configuration for generic SPD matrices, 
       * Create configuration for all user-define arguments. */
      gofmm::Configuration<T> config( cmd.metric, 
          cmd.n, cmd.m, cmd.k, cmd.s, cmd.stol, cmd.budget);  
      /** Use geometric-oblivious splitters. */
      const int N_CHILDREN = 2;
      mpigofmm::randomsplit<SPDMatrix<T>, N_CHILDREN, T> rkdtsplitter( K );
      mpigofmm::centersplit<SPDMatrix<T>, N_CHILDREN, T> splitter( K );
      /** GOFMM tree splitter. */
      splitter.Kptr = &K;
      splitter.metric = cmd.metric;
      /** Randomized tree splitter. */
      rkdtsplitter.Kptr = &K;
      rkdtsplitter.metric = cmd.metric;
      /** (Optional) provide neighbors, leave uninitialized otherwise. */
      auto NN = mpigofmm::FindNeighbors( K, rkdtsplitter, config, CommGOFMM );
      printf( "finish find Neighbors\n" ); fflush( stdout );

      /** Compress matrix K. */
      auto *tree_ptr = mpigofmm::Compress( K, NN, splitter, rkdtsplitter, config, CommGOFMM );
      auto &tree = *tree_ptr;
      /** Report the source and destination rank of an index. */
      for ( int gid = comm_rank; gid < n; gid += 200 )
      {
      //printf( "Here before\n" ); fflush( stdout );
      printf( "gid %4d source [BLK] %2d destination [IDS] %2d\n", 
          gid, gid % comm_size, tree.Index2Rank( gid ) ); fflush( stdout );
      //printf( "Here after\n" ); fflush( stdout );
      }
      /** [Step] Split the work to the multiple MPI ranks  */
      size_t n_loc = n / comm_size;
      size_t n_cut = n % comm_size;
      if ( comm_rank < n_cut ) n_loc ++; 
      printf( "Here after GIDS\n" ); 

      /** Input and output in RIDS and RBLK. THIS STEP IS PROBABLY UNNECESSARY */ 
      DistData<RBLK, STAR, T> u_rblk( n, n, CommGOFMM);
      printf( "Here after DISTDATA\n" ); 

      Data<T> u_identity( n_loc , n );      
      for ( size_t i = 0; i < n_loc; i ++ ){
        for ( size_t j = 0; j < n; j ++ ){
          // Fill the diagonal elements with 1
          if (i==j){
              u_identity(i,j)=1.0;
          }
          else{
            /** This is why it is uncessary becsause we could just put it to 0  */
            u_identity(i,j)=0.0;
          }

        }
      }
      printf( "Here after IDENTITY\n" ); 
  
      /** [Step] Factorization (HSS using ULV). */
      mpigofmm::DistFactorize( tree, lambda ); 
      /** [Step] Solve (K+lambda*I)w = u approximately with HSS. Using  */
      mpigofmm::DistSolve( tree, u_identity); 

    }
  
    /** [Step#] HMLP API call to terminate the runtime. */
    HANDLE_ERROR( hmlp_finalize() );
    /** Finalize Message Passing Interface. */
    mpi::Finalize();
  }
  catch ( const exception & e )
  {
    cout << e.what() << endl;
    return -1;
  }
  return 0;
}; /** end main() */

