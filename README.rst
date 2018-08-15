==========
uncover ML
==========

.. image:: https://circleci.com/gh/GeoscienceAustralia/uncover-ml/tree/master-cleanup.svg?style=svg
    :target: https://circleci.com/gh/GeoscienceAustralia/uncover-ml/tree/master-cleanup

Machine learning tools for the Geoscience Australia uncover project.

Quickstart
----------

Before you start, make sure your system has the following packages installed,

- gdal (libgdal-dev)
- openmpi
- hdf5

We strongly recommend using a virtual environment.
To install, simply run ``setup.py``:

.. code:: console

   $ python setup.py install

or install with ``pip``:

.. code:: console

   $ pip install git+https://github.com/GeoscienceAustralia/uncover-ml.git@release

The python requirements should automatically be built and installed.

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

See the `usage <http://GeoscienceAustralia.github.io/uncover-ml/usage.html>`_ documentation.

Running on NCI
--------------
Please see `The PBS Readme <pbs/README.md>`_ .

Collaboration
-------------
This software is jointly developed by NICTA and Geoscience Australia.
For a list of features still to be implemented, see the 
`issue tracker <https://github.com/GeoscienceAustralia/uncover-ml/issues>`_.


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
