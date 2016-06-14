#!/bin/bash
#PBS -P ge3
#PBS -q express
#PBS -l walltime=00:10:00,mem=1GB,ncpus=2,jobfs=1GB
#PBS -l wd

# setup environment
module load zlib atlas python3/3.4.3 hdf5/1.8.10 gdal/2.0.0 zeromq/4.1.3
export PATH=$HOME/.local/bin:$PATH
export PYTHONPATH=$HOME/.local/lib/python3.4/site-packages:$PYTHONPATH
export VIRTUALENVWRAPPER_PYTHON=/apps/python3/3.4.3/bin/python3                 
source $HOME/.local/bin/virtualenvwrapper.sh 

# start the virtualenv
workon uncoverml

mpirun ipympi $HOME/uncover-ml/demos/demo_prediction_pipeline.py

