#!/bin/bash
#PBS -P ge3
#PBS -q express 
#PBS -l walltime=00:05:00,mem=32GB,ncpus=32,jobfs=2GB
#PBS -l wd

# setup environment
module unload intel-cc
module unload intel-fc
module load python3/3.4.3 python3/3.4.3-matplotlib 
module load load hdf5/1.8.10 gdal/2.0.0
source $HOME/.profile

# start the virtualenv
workon uncoverml

# run command
mpirun --mca mpi_warn_on_fork 0 $HOME/uncover-ml/demos/demo_prediction_pipeline.py


