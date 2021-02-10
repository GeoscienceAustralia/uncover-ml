#!/bin/bash 
#PBS -N uncoverml
#PBS -P ge3 
#PBS -q gpuvolta
#PBS -l walltime=4:00:00
#PBS -l ncpus=12
#PBS -l ngpus=1
#PBS -l mem=96GB
#PBS -l jobfs=100GB
#PBS -l storage=gdata/ge3 

module purge
module load pbs
module load python3/3.7.4
module load gdal/3.0.2
module load openmpi/3.0.4

source ~/venvs/uncoverml_gadi/bin/activate
export UNCOVERML_SRC=~/github/uncover-ml/

uncoverml gridsearch $UNCOVERML_SRC/configs/single_xgboost.yaml

