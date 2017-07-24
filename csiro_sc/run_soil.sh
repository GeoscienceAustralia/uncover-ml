#!/bin/bash
#SBATCH --time=00:20:00
#SBATCH --mem=64GB
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=14

source $HOME/.bashrc
workon uncoverml
mpirun uncoverml learn $HOME/code/uncover-ml/csiro_sc/soil.yaml
mpirun uncoverml predict -p 5 $DATADIR/scratch/soil.model
