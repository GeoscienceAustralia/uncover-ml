#!/usr/bin/env bash
#PBS -P ge3
#PBS -q normal
#PBS -l walltime=20:00:00,mem=64GB,ncpus=16,jobfs=30GB
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

# the python command needs full path of the python script
mpirun -np 4 python batch_job.py -i /g/data/ge3/covariates/national -o /g/data/ge3/covariates/national_masked -m /g/data/ge3/covariates/masks/Mask_National_LL_nan_to_keep.tif -r bilinear