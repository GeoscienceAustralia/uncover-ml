Usage
=====

Running locally
---------------

UncoverML uses MPI for parallelization on localhosts and on clusters/high
performance computers. Here is an example of running the pipeline from the
command line,

.. code:: console

  $ mpirun -n 4 uncoverml learn -p 10 config.yaml

Breaking this down,

- `mpirun -n 4` instructs MPI to use four processors for the pipeline
- `uncoverml learn -p 10 config.yaml` runs the learning pipeline (i.e. learns a
  machine learning model). The `-p 10` flag makes 10 chunks of work for the
  four workers (this is to limit memory usage, more chunks, less memory usage),
  and the `config.yaml` is the configuration file for the pipeline.

Similarly, there are two more options,

.. code:: console

  $ mpirun -n 4 uncoverml predict -p 10 config.yaml

Which uses the learned model from the previous command to predict target values
for all query points, and

.. code:: console

  $ mpirun -n 4 uncoverml cluster config.yaml

Which clusters (unsupervised) all of the data.

Running on HPC
--------------

In the ``pbs`` directory of the repository there are some example scripts and a helper function
to assist launching batch jobs over multiple nodes with PBS.

.. todo::
    
    The PBS scripts and examples are outdated and need to be fixed.

UncoverML uses MPI for parallelization. To run an uncoverml command, use:

.. code:: bash

    mpirun -n <number_of_processors> <command>

An example of PBS job submission script:

.. code:: bash

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

where in this case mpirun is able to determine the number of available cores via PBS. This job 
submits the ``learn`` and ``predict`` jobs one after the other. The `-p 10` or `-p 40` options 
partitions the input covariates into the specificed number of memory partitions.

For more information on configuring PBS jobs on Raijin, view the 
`NCI user documentation <https://opus.nci.org.au/display/Help/Raijin+User+Guide>`_. 

.. include:: workflows.rst

Models
------

For an overview of the models available in UncoverML, view the module
documentation: :mod:`uncoverml.models`.

.. _diagnostics:

Diagnostics
-----------

.. _outputs:

Outputs
-------

Command Line Interface
----------------------

UncoverML has several command line options. Select an option below to 
view its documentation:

.. include:: scripts.rst
