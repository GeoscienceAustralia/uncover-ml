#!/bin/bash
set -e
ulimit -s unlimited


if [ -e ${PBS_O_WORKDIR}/client_cmd  ]; then
    rm  ${PBS_O_WORKDIR}/client_cmd
fi

jobdir=$PBS_JOBFS

cd $PBS_JOBFS

module purge
module load pbs
module load python3/3.7.4
module load gdal/3.0.2
source $PBS_O_WORKDIR/venvs/jupyter/bin/activate
cd $PBS_O_WORKDIR


export jport=8388

echo "Jupyter lab started ..."
jupyter lab --no-browser --ip=`hostname` --port=${jport} &

echo "client_cmd created ..."
echo "ssh -N -L ${jport}:`hostname`:${jport} ${USER}@gadi.nci.org.au &" > client_cmd
