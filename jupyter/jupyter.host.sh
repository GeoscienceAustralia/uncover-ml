#!/bin/bash 
#PBS -N uncoverml 
#PBS -P ge3 
#PBS -q express 
#PBS -l walltime=12:00:00 
#PBS -l ncpus=48 
#PBS -l mem=192GB 
#PBS -l jobfs=100GB 
#PBS -l storage=gdata/ge3 
module purge
module load pbs
module load python3/3.7.4
module load gdal/3.0.2
$HOME/uncover-ml/jupyter/jupyter.node.sh 
sleep infinity 