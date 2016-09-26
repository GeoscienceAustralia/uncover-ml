#!/usr/bin/env bash
#PBS -P ge3
#PBS -q express
#PBS -l walltime=00:20:00,mem=20GB,ncpus=1,jobfs=30GB
#PBS -l wd

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
python preprocessing/crop_mask_resample_reproject.py -i slope_fill2.tif -o slope_fill2_out.tif  -m mack_LCC.tif -r bilinear -e '-821597 -4530418 1431287 -4174316'
