#!/bin/bash
#PBS -P ge3
#PBS -q express
#PBS -l walltime=00:02:00,mem=2GB,ncpus=5,jobfs=1GB
#PBS -l wd

# setup environment
source $HOME/.profile

# start the virtualenv
workon uncoverml

py.test $HOME/uncover-ml/tests 2>&1 > test_output.log 

