#!/bin/bash
#PBS -P ge3
#PBS -q normal
#PBS -l walltime=00:30:00,mem=32GB,ncpus=8,jobfs=2GB
#PBS -l wd
#PBS -j oe

# setup environment
source $HOME/.profile

# start the virtualenv
workon uncoverml

# this initiates 4 jobs per node
mpirun -map-by ppr:4:node --mca mpi_warn_on_fork 0 predictionpipeline $HOME/uncover-ml/pbs/nci_murray.pipeline &> log.murray.prediction.txt


