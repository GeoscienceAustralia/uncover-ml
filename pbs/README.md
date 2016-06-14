# Uncover-ml on the NCI

This README is a quick guide to getting the uncover-ml library up and running
in a PBS batch environment that has MPI support. This setup is common in
HPC systems such as the NCI (raijin).

The instructions below should apply to both single- and multi-node runs
on the NCI. Just set ncpus in the PBS  directives in the job submission
script accordingly (e.g. ncpus=32 for 2 nodes).

## Pre-installation

These instructions currently only work with gcc and not the Intel compiler.
Note that on NCI it appears python is compiled against gcc anyway.

1. Unload the icc compiler from the terminal:
```
$ module unload icc
```
2. Load the modules requried for installation and running:
```
$ module load zlib atlas python3/3.4.3 hdf5/1.8.10 gdal/2.0.0 zeromq/4.1.3
```

2. Now add the following lines to the end of your ~/.profile:
```
export PATH=$HOME/.local/bin:$PATH
export PYTHONPATH=$HOME/.local/lib/python3.4/site-packages:$PYTHONPATH
export VIRTUALENVWRAPPER_PYTHON=/apps/python3/3.4.3/bin/python3                 
source $HOME/.local/bin/virtualenvwrapper.sh 
```

4. Install virtualenv and virtualenvwrapper by running the following command
on the terminal:
```
$ pip3 install  --user virtualenv virtualenvwrapper
```

5. Refresh your environment by reloading your profile:
```
$ source ~/.profile
```

## Installation

1. Create a new virtualenv for uncoverml:
```
$ mkvirtualenv --system-site-packages uncoverml
```

2. Make sure the virtualenv is activated:
```
$ workon uncoverml
```

3. Clone the uncoverml repo into your home directory:
```
$ cd ~
$ git clone git@github.com:NICTA/uncover-ml.git
```

4. Install uncoverml:
```
$ cd uncover-ml
$ python setup.py install
```

5. Once installation has completed, you can run the tests to verify everything
has gone correctly:
```
$ pip install pytest
$ py.test ~/uncover-ml/tests/
```

## Running Batch Jobs

in the `pbs` subfolder of uncover-ml there are some example scripts and a
helper function to assist launching batch jobs over multiple nodes with pbs

### Batch testing

To check everything is working, submit the tests as a batch job:
```
$ cd ~/uncover-ml/pbs
$ qsub submit_tests.sh
```

### ipympi

In the pbs folder there is a helper script called `ipympi`. It takes a single
argument which is a command to run. It will run 1 copy of that command,
along with 1 ipyparallel controller and (n-2) ipyparallel engines, where
n is the total number of processors assigned via mpirun. For example,
on the command line we could do something like
```
mpirun -n 4 ipympi <command>
```

whilst a PBS job submission might look like this:
```
#!/bin/bash
#PBS -P ge3
#PBS -q normal
#PBS -l walltime=00:10:00,mem=1GB,ncpus=2,jobfs=1GB
#PBS -l wd

# setup environment
module load zlib atlas python3/3.4.3 hdf5/1.8.10 gdal/2.0.0 zeromq/4.1.3
export PATH=$PATH:$HOME/.local/bin
export PYTHONPATH=$PYTHONPATH:$HOME/.local/lib/python3.4/site-packages
export VIRTUALENVWRAPPER_PYTHON=/apps/python3/3.4.3/bin/python3                 
source $HOME/.local/bin/virtualenvwrapper.sh 

# start the virtualenv
workon uncoverml

mpirun ipympi $HOME/uncover-ml/demos/demo_learning_pipeline.py
```
where in this case mpirun is able to determine the number of available
cores via PBS.

### Running the demos
In the pbs folder there are two scripts called  `submit_demo_predicion.sh`
and `submit_demo_learning.sh` that will submit a batch job to PBS that uses
mpirun and ipympi to run the demos. Feel free to modify the PBS directives
as needed, or copy these scripts to a more convenient location.






