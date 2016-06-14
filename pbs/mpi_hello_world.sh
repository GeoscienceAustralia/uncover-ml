#!/bin/bash

ID=$OMPI_COMM_WORLD_RANK
NNODES=$OMPI_COMM_WORLD_SIZE

echo "hello world, I am rank "$ID" of "$NNODES > log_$ID.log
