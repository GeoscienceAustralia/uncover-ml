Installation
============

Ubuntu 18.04
------------

UncoverML supports Ubuntu 18.04 with Python 3.6 or Python 3.7. 

The following instructions may be used for other Linux distributions but the packages and package
manager used may be different. If you require help using uncoverML with a different Linux 
Linux distribution, ask for help on our 
`Github Issues <https://github.com/GeoscienceAustralia/uncover-ml/issues>`_ page.

Before installing UncoverML, ensure your OS has the following packages:

- gdal
- openmpi
- hdf5

These can be installed with the following commands:

.. code:: bash

    sudo apt-get install \
        gdal-bin libgdal-dev \
        libblas-dev liblapack-dev \
        libatlas-base-dev libproj-dev \
        gfortran \
        openmpi-bin libopenmpi-dev

It's recommended to use a Python virtual environment (venv) when installing UncoverML. This will
prevent packages that it requires from conflicting with any exist Python environments.

To create a virtual environment, run in your shell:

.. code:: bash

    python3 -m venv /path/to/your/venv

where ``/path/to/your/venv`` is the directory where the venv will exist.

Once created, activate the venv with:

.. code:: bash

    source /path/to/your/venv/bin/activate

With your environment activate, there are some packages that must be installed before uncoverML.
Install these with pip:

.. code:: bash

    pip install -U pip setuptools
    pip install numpy scipy matplotlib Cython

You are now ready to install uncoverML. The latest stable release can be installed from the 
Python Package Index using:

.. code:: bash
    
    pip install uncoverml

To install from the repository, clone using git:

.. code:: bash

    git clone git@github.com:GeoscienceAustralia/uncover-ml

Once in the cloned repository, checkout the desired branch and install with pip:

.. code:: bash
    
    git checkout branch-to-install
    pip install .

.. todo::
    
    Need to include a simple workflow for testing the installation here.

This completes the installation. Check out the :ref:`Usage` documentation to get started using
UncoverML.

HPC
---

The following instructions refer specifically to NCI's Raijin, but may be applicable to other
HPC environments running PBS and MPI.

The first step is to unload unrequired and load required system modules:

.. code:: bash

    module unload intel-cc
    module unload intel-fc

    module load python3/3.7.2
    module load gdal/2.2.2
    module load openmpi/2.1.1
    moudle load hdf5/1.8.10
    module load geos/3.5.0

It's recommended to use virtualenv on Raijin. Install it with pip:

.. code:: bash

    pip3 install --user virtualenv virtualenvwrapper

Setup virtualenv by exporting some environment variables and activating the virtualenv wrapper:

.. code:: bash

    export PATH=$HOME/.local/bin:$PATH
    export PYTHONPATH=$HOME/.local/lib/python3.4/site-packages:$PYTHONPATH
    export VIRTUALENVWRAPPER_PYTHON=/apps/python3/3.4.3/bin/python3                 
    export LC_ALL=en_AU.UTF-8
    export LANG=en_AU.UTF-8

    source $HOME/.local/bin/virtualenvwrapper.sh 

For convenience, the above commands can be placed in your ``~/.profile``. This will run the above
commands everytime you open a new session on Raijin. Alternatively, if you already have a 
configuration in your path you'd like to preserve but don't want to type the above commands
every time, you can source the ``uncover-ml/pbs/setup_hpc.sh`` to perform the above commands as 
needed.

Create a virtualenv for uncoverML and activate it:

.. code:: bash

    mkvirtualenv --system-site-packages uncoverml
    workon uncoverml

Next, clone and install uncoverml:

.. code:: bash

    git clone git@github.com:geoscienceaustralia/uncover-ml
    cd uncover-ml
    python setup.py install

.. todo::
    
    Need to include a simple workflow for testing the installation here (can be run on login node).
    Tests don't count because they require dev requirements and shouldn't need to be installed
    for an average user.

This completes the installation. Check out the :ref:`Usage` documentation to get started using
uncoverML.

Reusing Shared Virtualenv
+++++++++++++++++++++++++

An alternative to the above installation is to activate the shared uncoverml virtual environment. 
On Raijin, activate by running:

.. code:: bash

    source /g/data/ge3/john/uncover-ml/create_uncoverml_env.sh

