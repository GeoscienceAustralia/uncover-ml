#!/bin/bash
#PBS -P ge3
#PBS -q express
#PBS -l walltime=20:00:00,mem=32GB,ncpus=2,jobfs=50GB
#PBS -l wd
#PBS -j oe

module unload intel-cc
module unload intel-fc
module load python3/3.4.3 python3/3.4.3-matplotlib
module load hdf5/1.8.10 gdal/2.0.0
module load openmpi/1.8

export PATH=$HOME/.local/bin:$PATH
export PYTHONPATH=$HOME/.local/lib/python3.4/site-packages:$PYTHONPATH
export VIRTUALENVWRAPPER_PYTHON=/apps/python3/3.4.3/bin/python3
export LC_ALL=en_AU.UTF-8
export LANG=en_AU.UTF-8
source $HOME/.local/bin/virtualenvwrapper.sh

# start the virtualenv
workon uncoverml

gammasensor -o /path/to/convolved/output/dir/ --apply --height 100 --absorption 0.01  /path/to/input/covariates