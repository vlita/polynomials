#!/bin/bash

# Local machine settings
export OMP_NUM_THREADS=10
export MKL_NUM_THREADS=10

# Activate environment
source /home/vlita3/miniconda3/envs/run_psi4 # Adjust path if needed
conda activate run_psi4

# Run Psi4
psi4 -n 10 "35.dat"  
