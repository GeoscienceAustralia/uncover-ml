# Uncover-ml on the NCI

This README is a quick guide to getting the uncover-ml library up and running
in a PBS batch environment that has MPI support. This setup is common in
HPC systems such as the NCI (raijin).

The instructions below should apply to both single- and multi-node runs
on the NCI. Just set ncpus in the PBS  directives in the job submission
script accordingly (e.g. ncpus=32 for 2 nodes).

The instructions assume you are using bash shell.

## Pre-installation

These instructions currently only work with gcc and not the Intel compiler.
Note that on NCI it appears python is compiled against gcc anyway.

1. Load the modules requried for installation and running:
```bash
$ module load python3/3.9.2 gdal/3.0.2 openmpi/4.0.2
```
(Alternatively, you may wish to add the above lines to your ~/.profile)

2. Now add the following lines to the end of your ~/.profile:
```bash
export PATH=$HOME/.local/bin:$PATH
export PYTHONPATH=$HOME/.local/lib/python3.9/site-packages:$PYTHONPATH
export VIRTUALENVWRAPPER_PYTHON=/apps/python3/3.9.2/bin/python3
export LC_ALL=en_AU.UTF-8
export LANG=en_AU.UTF-8
source $HOME/.local/bin/virtualenvwrapper.sh 
``` 

3. Install virtualenv and virtualenvwrapper by running the following command
on the terminal:
```bash
$ pip3 install  --user virtualenv virtualenvwrapper
```

5. Refresh your environment by reloading your profile:
```bash
$ source ~/.bashrc
```

## Installation

1. Create a new virtualenv for uncoverml:
```bash
$ mkvirtualenv --system-site-packages uncoverml
```

2. Make sure the virtualenv is activated:
```bash
$ workon uncoverml
```

3. Clone the uncoverml repo into your home directory:
```bash
$ cd ~
$ git clone git@github.com:GeoscienceAustralia/uncover-ml.git
```

4. Install mpi4py
```bash 
$ pip install --no-cache-dir mpi4py==3.1.3 --no-binary=mpi4py
```

5. Install uncoverml:
```bash
$ cd uncover-ml
$ python setup.py install
```

5. Once installation has completed, you can run the tests to verify everything
has gone correctly:
```bash
$ pip install pytest
$ py.test ~/uncover-ml/tests/
```

## Updating the Code
To update the code, first make sure you are in the `uncoverml` virtual environment:
```bash
$ workon uncoverml
```
Next, pull the latest commit from the main branch, and install:
```bash
$ cd ~/uncover-ml
$ git pull origin
$ python setup.py install
```
If the pull and the installation complete successfully, the code is ready to run!


## Running Batch Jobs

in the `pbs` subfolder of uncover-ml there are some example scripts and a
helper function to assist launching batch jobs over multiple nodes with pbs

### Batch testing

To check everything is working, submit the tests as a batch job:
```bash
$ cd ~/uncover-ml/pbs
$ qsub submit_tests.sh
```

### MPIRun

`uncoverml` uses MPI internally for parallelization. To run a script or demo
simply do

```bash
$ mpirun -n <num_procs> <command>
```

whilst a PBS job submission might look like this:

```bash
#!/bin/bash
#PBS -P ge3
#PBS -q normal
#PBS -l walltime=01:00:00,mem=128GB,ncpus=32,jobfs=20GB
#PBS -l wd

# setup environment
module unload intel-cc
module unload intel-fc
module load python3/3.4.3 python3/3.4.3-matplotlib 
module load load hdf5/1.8.10 gdal/2.0.0
source $HOME/.profile

# start the virtualenv
workon uncoverml

# run command
mpirun --mca mpi_warn_on_fork 0 uncoverml learn national_gamma_no_zeros.yaml -p 10
mpirun --mca mpi_warn_on_fork 0 uncoverml predict national_gamma_no_zeros.model -p 40
```

where in this case mpirun is able to determine the number of available
cores via PBS. This job submits the `learn`ing and `predict`ion jobs one 
after the other. The `-p 10` or `-p 40` options partitions the input 
covariates into 10/40 partitions during learning/prediction jobs.  

### PBS job configuration
[Refer to the NCI user guide](https://opus.nci.org.au/display/Help/Raijin+User+Guide) 
for different cpu and memory configuration options.

Specifically, [this section](https://opus.nci.org.au/display/Help/Raijin+User+Guide#RaijinUserGuide-QueueLimits)
details the various combinations available. We recommend using the `normal`, 
`express`, `normalbw`, or the `expressbw` option with the required memory.

### Running the demos
In the pbs folder there are two scripts called  `submit_demo_predicion.sh`
and `submit_demo_learning.sh` that will submit a batch job to PBS that uses
mpirun and ipympi to run the demos. Feel free to modify the PBS directives
as needed, or copy these scripts to a more convenient location.






