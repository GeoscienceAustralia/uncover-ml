#!/bin/bash

if [ -z $OMPI_COMM_WORLD_SIZE ]; then
  echo "Cannot see MPI world, are you using mpirun?"
  exit 1
fi

ID=$OMPI_COMM_WORLD_RANK
NNODES=$OMPI_COMM_WORLD_SIZE

if [ $NNODES -lt 2 ]; then
  echo "Error: 2 cpu minimum (1 controller, 1 engine)"
  exit 1
fi

if [ $ID -eq 0 ]; then
    echo "Running ipcontroller on cpu $ID"
    ipcontroller --ip="*" 2>&1 > ipcontroller.log
else
    ENGINEID=$((ID - 1))
    echo "Running ipengine $ENGINEID on cpu $ID"
    sleep 10
    ipengine 2>&1 > ipengine_${ENGINEID}.log
fi 
