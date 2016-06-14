#!/bin/bash
#PBS -P ge3
#PBS -q normal
#PBS -l walltime=01:00:00,mem=32GB,ncpus=32,jobfs=10GB
#PBS -l wd

# setup environment
module load zlib atlas python3/3.4.3 hdf5/1.8.10 gdal/2.0.0 zeromq/4.1.3
export PATH=$PATH:$HOME/.local/bin
export PYTHONPATH=$PYTHONPATH:$HOME/.local/lib/python3.4/site-packages
export VIRTUALENVWRAPPER_PYTHON=/apps/python3/3.4.3/bin/python3                 
source $HOME/.local/bin/virtualenvwrapper.sh 

# start the virtualenv
workon uncoverml

mpirun ipympi $HOME/uncover-ml/demos/demo_learning_pipeline.py

