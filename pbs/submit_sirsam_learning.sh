#!/bin/bash
#PBS -P ge3
#PBS -q normal
#PBS -l walltime=00:05:00,mem=32GB,ncpus=32,jobfs=1GB
#PBS -l wd
#PBS -j oe

# setup environment
source $HOME/.profile

# start the virtualenv
workon uncoverml

# run command
mpirun --mca mpi_warn_on_fork 0 learningpipeline $HOME/uncover-ml/pbs/nci_sirsam.pipeline &> log.sirsam.learning.txt

