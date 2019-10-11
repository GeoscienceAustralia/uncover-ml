module unload intel-cc
module unload intel-fc

module load python3/3.7.2
module load gdal/2.2.2
module load openmpi/2.1.1
module load hdf5/1.8.10
module load geos/3.5.0

export PATH=$HOME/.local/bin:$PATH
export PYTHONPATH=$HOME/.local/lib/python3.4/site-packages:$PYTHONPATH
export VIRTUALENVWRAPPER_PYTHON=/apps/python3/3.4.3/bin/python3                 
export LC_ALL=en_AU.UTF-8
export LANG=en_AU.UTF-8
source $HOME/.local/bin/virtualenvwrapper.sh 
