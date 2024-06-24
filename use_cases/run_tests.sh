#!/bin/bash

# Load necessary modules
module load charliecloud/0.25

# Define problem sizes to test
problem_sizes=(512 1024 2048 4096 8192 16384)

# Set OpenMP environment variables
export OMP_NUM_THREADS=28  # Set the number of OpenMP threads
export OMP_STACKSIZE=512M  # Set the stack size per thread

# Loop over each problem size
for size in "${problem_sizes[@]}"; do
    export PROBLEM_SIZE=$size
    echo "Testing problem size: $PROBLEM_SIZE"

    # Run the Python script with the current problem size
    ch-run --set-env=./gofmm/ch/environment -w ./gofmm -- python3 workspace/gofmm/use_cases/test_inv_gauss.py

    echo "Finished testing problem size: $PROBLEM_SIZE"
done
