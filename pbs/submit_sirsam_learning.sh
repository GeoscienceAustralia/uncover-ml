#!/bin/bash
#PBS -P ge3
#PBS -q express
#PBS -l walltime=00:05:00,mem=64GB,ncpus=32,jobfs=1GB
#PBS -l wd

# setup environment
source $HOME/.profile

# start the virtualenv
workon uncoverml

# run command
mpirun --mca mpi_warn_on_fork 0 learningpipeline $HOME/uncover-ml/pbs/nci_sirsam.pipeline

