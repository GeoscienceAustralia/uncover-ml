#!/bin/bash
#PBS -P ge3
#PBS -q normal 
#PBS -l walltime=00:10:00,mem=64GB,ncpus=32,jobfs=2GB
#PBS -l wd
#PBS -j oe

# setup environment
source $HOME/.profile

# start the virtualenv
workon uncoverml

# this initiates 4 jobs per node
mpirun -map-by ppr:4:node --mca mpi_warn_on_fork 0 predictionpipeline $HOME/uncover-ml/pbs/nci_sirsam.pipeline &> log.sirsam.prediction.txt


