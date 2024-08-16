# Distributed Matrix Inversion with MPI

The instructions cover the necessary code modifications, file additions, and execution steps.All the additions needed is inside use_cases/distributed_inverse.cpp

## File Modifications and Additions
1. **New File: `distributed_inverse.cpp`:**

    - Add the `distributed_inverse.cpp` file to the `cd/path/to/hmlp/example` directory.

      FROM:  https://github.com/Azizos126926/gofmm_swig_python/blob/40c20e576c56a961d1c4468ed220100c90931fae/use_cases/distributed_inverse.cpp#L22
      TO: https://github.com/Azizos126926/gofmm_swig_python/blob/40c20e576c56a961d1c4468ed220100c90931fae/use_cases/distributed_inverse.cpp#L124
    ```

2. **Additions to `cd /path/to/hmlp/gofmm/gofmm_mpi.hpp`:**
    - Insert the following code snippet into `#gofmm_mpi.hpp`:
    
    FROM: https://github.com/Azizos126926/gofmm_swig_python/blob/40c20e576c56a961d1c4468ed220100c90931fae/use_cases/distributed_inverse.cpp#L126
    TO: https://github.com/Azizos126926/gofmm_swig_python/blob/40c20e576c56a961d1c4468ed220100c90931fae/use_cases/distributed_inverse.cpp#L183
    ```cpp
    // Add any necessary function prototypes or global variables related to the distributed inverse computation here.
    ```

3. **Additions to `cd /path/to/hmlp/gofmm/igofmm_mpi.hpp`:**
    - Insert the following code snippet into `igofmm_mpi.hpp`:

    FROM: https://github.com/Azizos126926/gofmm_swig_python/blob/40c20e576c56a961d1c4468ed220100c90931fae/use_cases/distributed_inverse.cpp#L186
    TO: https://github.com/Azizos126926/gofmm_swig_python/blob/40c20e576c56a961d1c4468ed220100c90931fae/use_cases/distributed_inverse.cpp#L273
    ```



## Build and Install the Project

Follow these steps to build and install the project:

1. **Navigate to the `hmlp` directory:**

    ```bash
    cd /path/to/hmlp
    ```

2. **Build the project:**

    ```bash
    make
    ```

3. **Install the project:**

    ```bash
    make install
    ```

## Running the Distributed Inverse Program

You can execute the distributed matrix inversion with a randomly generated kernel using the following command. This example uses 2 MPI processes:

1. **Allocate resources using `salloc` (if applicable in your environment):**

    ```bash
    salloc --nodes=1 --ntasks-per-node=2 --time=00:10:00
    ```

2. **Run the program:**

    ```bash
    mpirun -n 2 ./distributed_inverse 2048 1024 64 1024 10 1E-5 0.01 kernel testsuit
    `

---

This setup should allow you to successfully compile, install, and run the distributed inverse matrix program using MPI within the `hmlp` framework.
