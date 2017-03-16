Usage
=====

Running locally
---------------

Uncover-ML uses MPI for parallelization on localhosts and on clusters/high
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

See also:

- :doc:`Scripts <scripts>` for details on the script options
- :doc:`Configuration <config>` for how to create a configuration file
- :doc:`Models <models>` for an overview of the available supervised
  algorithm


Running on NCI
--------------
Please see `The PBS Readme <https://github.com/GeoscienceAustralia/uncover-ml/blob/master/pbs/README.md>`_ .
