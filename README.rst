==========
uncover ML
==========

.. .. image:: https://badge.fury.io/py/uncover-ml.png
..     :target: http://badge.fury.io/py/uncover-ml

.. .. image:: https://travis-ci.org/dsteinberg/uncover-ml.png?branch=master
..     :target: https://travis-ci.org/dsteinberg/uncover-ml

.. .. image:: https://codecov.io/github/dsteinberg/uncover-ml/coverage.svg?branch=master
..     :target: https://codecov.io/github/dsteinberg/uncover-ml?branch=master

.. .. image:: https://pypip.in/d/uncover-ml/badge.png
..     :target: https://pypi.python.org/pypi/uncover-ml


Machine learning tools for the Geoscience Australia uncover project.

Quickstart
----------

Before you start, make sure your system has the following packages installed,

- gdal (libgdal-dev)
- openmpi
- hdf5

We strongly recommond using a virtual environment.
To install, simply run ``setup.py``:

.. code:: console

   $ python setup.py install

or install with ``pip``:

.. code:: console

   $ pip install git+https://github.com/nicta/uncover-ml.git@release

The python requirements should automatically be built and installed.
Have a look at some of the `demos <demos/>`_ for how to use these tools.

Cubist
------

In order to use the cubist regressor, you need to first make sure cubist is
installed. This is easy with our simple installation script, invoke it with:

.. code:: console
    
    $ ./makecubist <installation-path>

Once cubist is installed, it will add a configuration file to the script,
if you like, you can test that it's been installed in the correct place by
checking the contents of `uncover-ml/cubist_config.py`, its presence
indicates that the installation completed successfully.

Next you need to rerun the setup script with:

.. code:: console

    $ python setup.py install

Which will ensure the cubist_config has been added successfully. Now you
should be able to use the cubist regressor in the pipeline file.

Running 
-------

Uncover-ML uses MPI for parallelization on localhosts and on clusters.
*DO NOT RUN ipcluster*, we no longer use ipyparallel. Run the demos and scripts
using mpirun:

.. code:: console

  $ mpirun -n 4 python demo_learning_pipeline.py

Note that demo_learning_pipeline and demo_prediction_pipeline must be run
with the same number of CPUs.

Running on NCI
--------------
Please see `The PBS Readme <pbs/README.md>`_ .

Contract with Geoscience Australia
----------------------------------
This software is being developed under contract with Geoscience Australia.
For a list of features still to be implemented, see the 
`issue tracker <https://github.com/nicta/uncover-ml/issues>`.


Useful Links
------------

Home Page
    http://github.com/nicta/uncover-ml

Documentation
    http://nicta.github.io/uncover-ml

Issue tracking
    https://github.com/nicta/uncover-ml/issues


Bugs & Feedback
---------------

For bugs, questions and discussions, please use 
`Github Issues <https://github.com/NICTA/uncover/issues>`_.
