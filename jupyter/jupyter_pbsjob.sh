#!/bin/bash 
#PBS -N uncoverml-jupyter
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

set -e
ulimit -s unlimited

GIT_HOME=$HOME/github  # where to check out the uncover-ml repoitory
VENVS=$HOME/venvs

source $VENVS/uncoverml_gadi/bin/activate

PBS_WORKDIR=$GIT_HOME/uncover-ml/jupyter
cd $PBS_WORKDIR

export jport=8388  # choose a port number

echo "Starting Jupyter lab ..."
jupyter lab --no-browser --ip=`hostname` --port=${jport} &


echo "ssh -N -L ${jport}:`hostname`:${jport} ${USER}@gadi.nci.org.au &" > client_cmd
echo "client_cmd created ..."

sleep infinity 
