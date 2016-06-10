
load the right modules
```
$ module load intel-mkl/16.0.3.210 python3/3.4.3 python3/3.4.3-matplotlib hdf5/1.8.10 gdal/2.0.0 zeromq/4.1.3 
```

create a local directory for local prereqs
```
$ mkdir -p ~/.local/lib/python3.4/site-packages ~/.local/bin
```
Then in your `.profile` file, add the lines
```
export PATH=$PATH:$HOME/.local/bin
export PYTHONPATH=$PYTHONPATH:$HOME/.local/lib/python3.4/site-packages
```

now refresh your environment
```
$ source ~/.profile
```

clone the uncoverml repo
```
$ git clone git@github.com:NICTA/uncover-ml.git
```

then install
```
$ cd uncover-ml
$ python3 setup.py develop --prefix=~/.local
```

Now check all is okay with the test script:

## Possibilities for syncronisation of the 3 jobs

1.python mpi, do it all inside python
2. separate jobs
3. PBS_VNODENUM and PBS_NODENUM environment variables (only with -t?)

qsub -l nodes=4:ppn=2 	Request 2 processors on each of four nodes

# TMUX reference

## Useful OpenMPI environment variables

OMPI_COMM_WORLD_SIZE - the number of processes in this process' MPI_COMM_WORLD
OMPI_COMM_WORLD_RANK - the MPI rank of this process in MPI_COMM_WORLD
OMPI_COMM_WORLD_LOCAL_RANK - the relative rank of this process on this node within its job. For example, if four processes in a job share a node, they will each be given a local rank ranging from 0 to 3.
OMPI_UNIVERSE_SIZE - the number of process slots allocated to this job. Note that this may be different than the number of processes in the job.
OMPI_COMM_WORLD_LOCAL_SIZE - the number of ranks from this job that are running on this node.
OMPI_COMM_WORLD_NODE_RANK - the relative rank of this process on this node looking across ALL jobs.

