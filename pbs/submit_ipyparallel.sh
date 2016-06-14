#!/bin/bash
#PBS -P ge3
#PBS -q express
#PBS -l walltime=00:10:00,mem=1GB,ncpus=2,jobfs=1GB
#PBS -l wd

# setup environment
source $HOME/.profile

# start the virtualenv
workon uncoverml

mpirun ipyparallel_mpi.sh

