#!/bin/bash
#PBS -P ge3
#PBS -q normal
#PBS -l walltime=00:10:00,mem=1GB,ncpus=2,jobfs=1GB
#PBS -l wd

# setup environment
source $HOME/.profile

# start the virtualenv
workon uncoverml

mpirun ipympi $HOME/uncover-ml/demos/demo_learning_pipeline.py

