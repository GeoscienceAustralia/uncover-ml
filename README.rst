uncover ML
==========

.. image:: https://circleci.com/gh/GeoscienceAustralia/uncover-ml/tree/develop.svg?style=svg
    :target: https://circleci.com/gh/GeoscienceAustralia/uncover-ml/tree/develop  
    
.. image:: https://codecov.io/gh/GeoscienceAustralia/uncover-ml/branch/develop/graph/badge.svg
    :target: https://codecov.io/gh/GeoscienceAustralia/uncover-ml

Machine learning tools for the Geoscience Australia uncover project.

Quickstart
----------

Before you start, make sure your system has the following packages installed,

- gdal (libgdal-dev)
- openmpi
- hdf5

And your Python environment has:

- numpy
- scipy
- matplotlib
- Cython

We strongly recommend using a virtual environment.
To install, simply run ``setup.py``:

.. code:: console

   $ python setup.py install

or install with ``pip``:

.. code:: console

   $ pip install git+https://github.com/GeoscienceAustralia/uncover-ml.git@release

The Python requirements should automatically be built and installed.

Installation and Usage
----------------------

UncoverML can be run on a local Linux machine or a HPC platform such as NCI's Raijin. For
detailed installation and usage instructions, see the `documentation <http://GeoscienceAustralia.github.io/uncover-ml>`_.

Useful Links
------------

Home Page
    http://github.com/GeoscienceAustralia/uncover-ml

Documentation
    http://GeoscienceAustralia.github.io/uncover-ml

Issue tracking
    https://github.com/GeoscienceAustralia/uncover-ml/issues


Bugs & Feedback
---------------

For bugs, questions and discussions, please use 
`Github Issues <https://github.com/GeoscienceAustralia/uncover/issues>`_.
