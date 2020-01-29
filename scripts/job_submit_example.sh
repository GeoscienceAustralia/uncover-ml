#!/bin/bash

# Submit this on a PBS system by running:
# qsub job_submit_example.sh

#PBS -P ge3
#PBS -q normal
#PBS -l walltime=0:10:00,mem=4gb,ncpus=4,wd

# Load required modules (depending on platform)
#source setup_vdi.sh
source setup_gadi.sh

# Source your virtual environment
# Refer to Uncover-ML installation docs if are unsure of setting up a virtual environment
source /path/to/your/uncover-ml/venv

# Run a job
# Learning
mpiexec -n 4 uncoverml learn random_forest.yaml

# Prediction
mpiexec -n 4 uncoverml predict random_forest.yaml

# Create a map of covariate shift
mpiexec -n 4 uncoverml shiftmap random_forest.yaml

# Get some diagnostics about covariate data
mpiexec -n 4 covdiag ../tests/test_data/sirsam/covariates
