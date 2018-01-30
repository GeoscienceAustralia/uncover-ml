#!/bin/bash
#PBS -P ge3
#PBS -N AA
#PBS -q megamem
#PBS -l walltime=24:00:00,mem=3072GB,ncpus=32,jobfs=3072GB
#PBS -l wd
#PBS -j oe
#PBS -M rakib.hassan@ga.gov.au
#PBS -m bae

module unload openmpi/1.4.3
module load openmpi/1.6.3
module load python/2.7.3
module load hdf5/1.8.10
module load mpi4py/1.3.1
module load gdal/1.11.1-python

export PATH=$HOME/.local/bin:$PATH
export PYTHONPATH=/g/data/ge3/rakib/raijin/uncover-ml:$PYTHONPATH
export VIRTUALENVWRAPPER_PYTHON=/apps/python/2.7.3/bin/python
export LC_ALL=en_AU.UTF-8
export LANG=en_AU.UTF-8

mpirun -np 32 --mca mpi_warn_on_fork 0 python ../uncover-ml/preprocessing/multiscale.py extendedFilelist.txt /g/data/ge3/covariates/national/temp-multiscale 10 --extrapolate 0 --keep-level 2 --keep-level 9 --keep-level 10 --log-level DEBUG

