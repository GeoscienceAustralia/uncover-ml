Configuring UncoverML
=====================

UncoverML workflows are controlled by a YAML configuration file.
This section provides some examples and explanations of different 
workflows and possible parameters.

For a reference to all possible config parameters, view the module
documentation: :mod:`uncoverml.config`

Random Forest
-------------

The first example is a configuration file for a Random Forest model.
This file can be found in the repository under `tests/test_data/sirsam/random_forest/sirsam_Na_random_forest.yaml`.

.. code:: yaml
 
  learning:
    algorithm: multirandomforest
    arguments:
      n_estimators: 10
      target_transform: log
      forests: 20

The 'learning' block specifies the algorithm or model to train. 'algorithm'
is the name of the algorithm. 'arguments' specifies a dictionary of
keyword arguments specific to that model. For reference about what
arguments are applicable, refer the documentation for the specific model.

.. code:: yaml

  features:
    - type: ordinal
      files:
        - directory: $UNCOVERML_SRC/tests/test_data/sirsam/covariates/
      transforms:
        - centre
        - standardise
      imputation: mean

The 'features' block 

.. code:: yaml

  targets:
    file: $UNCOVERML_SRC/tests/test_data/sirsam/targets/geochem_sites_log.shp
    property: Na_log

  validation:
    feature_rank: True
    k-fold:
      parallel: True
      folds: 5
      random_seed: 1

  prediction:
    quantiles: 0.95
    outbands: 10

  output:
    directory: $UNCOVERML_SRC/tests/test_data/sirsam/random_forest/out
    model: $UNCOVERML_SRC/tests/test_data/sirsam/random_forest/out/sirsam_Na_randomforest.model
    plot_feature_ranks: True
    plot_intersection: True
    plot_real_vs_pred: True
    plot_correlation: True
    plot_target_scaling: True

  pickling:
    covariates: $UNCOVERML_SRC/tests/test_data/sirsam/random_forest/out/features.pk
    targets: $UNCOVERML_SRC/tests/test_data/sirsam/random_forest/out/targets.pk

 

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

Configuration File
------------------

.. todo:: 

    Need example configurtion file and explanation of parameters.

Models
------

For an overview of the models available in UncoverML, view the module
documentation: :mod:`uncoverml.models`

Command Line Interface
----------------------

UncoverML has several command line options. Select an option below to 
view its documentation:

.. include:: scripts.rst
