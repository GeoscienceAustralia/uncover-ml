#!/usr/bin/env bash
#PBS -P ge3
#PBS -q express
#PBS -l walltime=00:30:00,mem=16GB,ncpus=1,jobfs=10GB
#PBS -l wd

module rm intel-fc intel-cc
module load python/2.7.6
module load python/2.7.6-matplotlib
module load gdal/1.11.1-python
module load openmpi/1.8

python crop_mask_resample_reproject.py -i slope_fill2.tif -o slope_fill2_out.tif  -m mack_LCC.tif -r bilinear -e '-821597 -4530418 1431287 -4174316'
