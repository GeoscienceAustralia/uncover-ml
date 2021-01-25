  
#!/bin/bash 
#PBS -N uncoverml-oos
#PBS -P ge3 
#PBS -q express 
#PBS -l walltime=2:00:00 
#PBS -l ncpus=4 
#PBS -l mem=192GB 
#PBS -l jobfs=100GB 
#PBS -l storage=gdata/ge3 

module purge
module load pbs

module load python3/3.7.4
module load gdal/3.0.2
module load openmpi/3.0.4

source ~/venvs/uncoverml_gadi/bin/activate

export UNCOVERML_SRC=~/github/uncover-ml/
uncoverml gridsearch $UNCOVERML_SRC/configs/optimisation.yaml


uncoverml gridsearch $UNCOVERML_SRC/configs/optimisation_xgboost.yaml
