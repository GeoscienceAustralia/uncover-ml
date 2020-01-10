Installation
============

Ubuntu 18.04
------------

UncoverML supports Ubuntu 18.04 with Python 3.6 or Python 3.7. 

The following instructions may be used for other Linux distributions but the packages and package
manager used may be different. If you require help using UncoverML with a different Linux 
distribution, ask for help on our 
`Github Issues <https://github.com/GeoscienceAustralia/uncover-ml/issues>`_ page.

Before installing UncoverML, ensure your OS has the following packages:

- gdal
- openmpi

These can be installed with the following commands:

.. code:: bash

    sudo apt-get install \
        gdal-bin libgdal-dev \
        libblas-dev liblapack-dev \
        libatlas-base-dev libproj-dev \
        gfortran \
        openmpi-bin libopenmpi-dev

It's recommended to use a Python virtual environment (venv) when installing UncoverML. This will
prevent packages that it requires from conflicting with any existing Python environments.

To create a virtual environment, run in your shell:

.. code:: bash

    python3 -m venv /path/to/your/venv

where ``/path/to/your/venv`` is the directory where the venv will exist.

Once created, activate the venv with:

.. code:: bash

    source /path/to/your/venv/bin/activate

You are now ready to install UncoverML. To install from the repository, clone using git:

.. code:: bash

    git clone git@github.com:GeoscienceAustralia/uncover-ml
    cd uncover-ml

Before the UncoverML package can be installed, there are some required packages that need to be installed first.

.. code:: bash

    pip install -U pip setuptools
    pip install -r requirements-pykrige.txt

Once these are done, you can install UncoverML with pip:

.. code:: bash

    pip install .

**Alternatively**, you can install the latest stable release from the Python Package Index.
Note you will have install some prerequisites:

.. code:: bash
    
    pip install -U pip setuptools
    pip install Cython==0.29.13
    pip install numpy==1.17.2
    pip install scipy==1.3.1
    pip install matplotlib==3.1.1
    pip install uncover-ml

To ensure the installation has been successful, try running UncoverML against the test data
included in the repository:

.. code:: bash

   uncoverml learn tests/test_data/sirsam/random_forest/sirsam_Na_randomforest.yaml

This will start training a random forest model on the test data. It may take a few minutes.
If the script completes successfully (there will be a log message saying "Finished!") then 
UncoverML has been correctly installed.

This completes the installation. Check out the :ref:`Usage` documentation to get started using
UncoverML.

When you are finished using UncoverML, don't forget to deactivate your virtual environment using:

.. code:: bash

    deactivate

HPC
---

The following instructions refer specifically to NCI's Gadi and Virtual Desktop (VDI), but may be applicable to other
HPC environments running PBS and MPI.

The first step is to unload unrequired and load required system modules. The below are compatible with the VDI (and Raijin, for legacy purposes):

.. code:: bash

    module unload intel-cc
    module unload intel-fc

    module load python3/3.6.2
    module load gdal/2.2.2
    module load openmpi/2.1.1
    module load geos/3.5.0

If you are running on the new Gadi supercomputer, the modules required are different:

.. code:: bash
 
    module load ptython3/3.7.4
    module load gdal/3.0.2
    module load openmpi/2.1.6

For convenience you can place the above commands in your ``~/.bashrc`` (``~/.profile`` on VDI). 
Alternatively, if you already have a configuration in your profile you'd like to preserve but don't
want to type the above commands every time, you can source the ``uncover-ml/pbs/setup_vdi.sh`` or 
``uncover-ml/pbs/setup_gadi.sh`` scripts depending on your platform. 

When using the NCI, a virtual environment is recommended. To create a virtual environment, run in your shell:

.. code:: bash

    python3 -m venv /path/to/your/venv

where ``/path/to/your/venv`` is the directory where the venv will exist.

Once created, activate the venv with:

.. code:: bash

    source /path/to/your/venv/bin/activate

You are now ready to install UncoverML. To install from the repository, clone using git:

.. code:: bash

    git clone git@github.com:GeoscienceAustralia/uncover-ml
    cd uncover-ml

Before the UncoverML package can be installed, there are some required packages that need to be installed first.

.. code:: bash

    pip install -U pip setuptools
    pip install -r requirements-pykrige.txt

Once these are done, you can install UncoverML with pip:

.. code:: bash

    python setup.py install

To ensure the installation has been successful, try running UncoverML against the test data
included in the repository:

.. code:: bash

   uncoverml learn tests/test_data/sirsam/random_forest/sirsam_Na_randomforest.yaml

This will start training a random forest model on the test data. It may take a few minutes.
If the script completes successfully (there will be a log message saying "Finished!") then 
UncoverML has been correctly installed.

This completes the installation. Check out the :ref:`Usage` documentation to get started using
UncoverML.

When you are finished using UncoverML, don't forget to deactivate your virtual environment using:

.. code:: bash

    deactivate

Reusing Shared Virtualenv
+++++++++++++++++++++++++

An alternative to the above installation is to activate the shared UncoverML virtual environment. 
On Raijin, activate by running:

.. code:: bash

    source /g/data/ge3/john/uncover-ml/create_uncoverml_env.sh

