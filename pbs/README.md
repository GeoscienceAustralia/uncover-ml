NOTE: these instructions currently only work with gcc not the Intel compiler.
make sure your ~/.profile file is not loading icc.


load the right modules
```
$ module load zlib atlas python3/3.4.3 hdf5/1.8.10 gdal/2.0.0 zeromq/4.1.3 
```

Then in your `.profile` file, add the lines (for local python installation)
```
export PATH=$PATH:$HOME/.local/bin
export PYTHONPATH=$PYTHONPATH:$HOME/.local/lib/python3.4/site-packages
export VIRTUALENVWRAPPER_PYTHON=/apps/python3/3.4.3/bin/python3                 
source $HOME/.local/bin/virtualenvwrapper.sh 
```

Install virtualenv
```
pip3 install  --user virtualenv virtualenvwrapper
```

now refresh your environment
```
$ source ~/.profile
```

create a new virtualenv for uncoverml
```
$ mkvirtualenv --system-site-packages uncoverml
```

make sure the virtualenv is activated
```
$ workon uncoverml
```

clone the uncoverml repo
```
$ git clone git@github.com:NICTA/uncover-ml.git
```

then install (note setup.py develop does not work for some reason)
```
$ cd uncover-ml
$ python setup.py install
```

Now check all is okay with the test script:
```
pip install pytest
py.test tests/
```


-lncpus=32,mem=10GB implies 2 nodes. Memory is across all nodes,
so in this case we're asking for 2 nodes with 5GB of memory each

## Possibilities for syncronisation of the 3 jobs

1.python mpi, do it all inside python
2. separate jobs

using pbsdsh instead of mpirun then utilising PBS_VNODENUM

qsub -l nodes=4:ppn=2 	Request 2 processors on each of four nodes

# TMUX reference

## Useful OpenMPI environment variables

OMPI_COMM_WORLD_SIZE - the number of processes in this process' MPI_COMM_WORLD
OMPI_COMM_WORLD_RANK - the MPI rank of this process in MPI_COMM_WORLD
OMPI_COMM_WORLD_LOCAL_RANK - the relative rank of this process on this node within its job. For example, if four processes in a job share a node, they will each be given a local rank ranging from 0 to 3.
OMPI_UNIVERSE_SIZE - the number of process slots allocated to this job. Note that this may be different than the number of processes in the job.
OMPI_COMM_WORLD_LOCAL_SIZE - the number of ranks from this job that are running on this node.
OMPI_COMM_WORLD_NODE_RANK - the relative rank of this process on this node looking across ALL jobs.

