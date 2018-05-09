Installation
============

Before you start, make sure your system has the following packages installed,

- gdal (libgdal-dev version >= 2.0.0)
- openmpi
- hdf5

libgdal-dev version 2.0.0 or higher is required to build the required GDAL python package. For Ubuntu, binaries for version 2.0.0 and higher can be found in PPA https://wiki.ubuntu.com/UbuntuGIS.  

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