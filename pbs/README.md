# Uncover-ml on the NCI

This README is a quick guide to getting the uncover-ml library up and running
in a PBS batch environment that has MPI support. This setup is common in
HPC systems such as the NCI (raijin).

## Pre-installation

These instructions currently only work with gcc and not the Intel compiler.
Note that on NCI it appears python is compiled against gcc anyway.

In your `~/.profile` file:

1. In your `~/.profile` file, make sure the line 
```
# module load icc
```
is commented out. 

2. Noww add the following lines to the end of the file:
```
export PATH=$PATH:$HOME/.local/bin
export PYTHONPATH=$PYTHONPATH:$HOME/.local/lib/python3.4/site-packages
export VIRTUALENVWRAPPER_PYTHON=/apps/python3/3.4.3/bin/python3                 
source $HOME/.local/bin/virtualenvwrapper.sh 
```

3. In your terminal session, run the following command to load the needed
modules:
```
$ module load zlib atlas python3/3.4.3 hdf5/1.8.10 gdal/2.0.0 zeromq/4.1.3 
```

4. Install virtualenv and virtualenvwrapper by running the following command
on the terminal:
```
$ pip3 install  --user virtualenv virtualenvwrapper
```

5. Refresh your environment by reloading your profile:
```
$ source ~/.profile
```

## Installation

1. Create a new virtualenv for uncoverml:
```
$ mkvirtualenv --system-site-packages uncoverml
```

2. Make sure the virtualenv is activated:
```
$ workon uncoverml
```

3. Clone the uncoverml repo into your home directory:
```
$ cd ~
$ git clone git@github.com:NICTA/uncover-ml.git
```

4. Install uncoverml:
```
$ cd uncover-ml
$ python setup.py install
```

5. Once installation has completed, you can run the tests to verify everything
has gone correctly:
```
$ pip install pytest
$ py.test ~/uncover-ml/tests/
```

## Running Batch Jobs





