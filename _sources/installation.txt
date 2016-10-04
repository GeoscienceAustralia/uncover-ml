Installation
============

Before you start, make sure your system has the following packages installed,

- gdal (libgdal-dev)
- openmpi
- hdf5

We **strongly** recommend using a virtual environment.
To install, simply run ``setup.py``:

.. code:: console

   $ python setup.py install

or install with ``pip``:

.. code:: console

   $ pip install git+https://github.com/nicta/uncover-ml.git@release

The python requirements should automatically be built and installed.

Cubist
------

In order to use the cubist regressor, you need to first make sure cubist is
installed. This is easy with our simple installation script, invoke it with:

.. code:: console
    
    $ ./makecubist <installation-path>

Once cubist is installed, it will add a configuration file to the script. If
you like, you can test that it's been installed in the correct place by
checking the contents of `uncover-ml/cubist_config.py`, its presence indicates
that the installation completed successfully.

Next you need to rerun the setup script with:

.. code:: console

    $ python setup.py install

Which will ensure the `cubist_config` has been added successfully. Now you
should be able to use the cubist regressor in the pipeline file.
