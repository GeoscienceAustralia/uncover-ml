# UncoverML on Bracewell


Create a `.bashrc` that looks something like this:

```
module load tensorflow/1.1.0-py35-gpu
module load gdal
module load hdf5
module load openmpi
source $(which virtualenvwrapper.sh)
```

Source the file or re-login so that you get the virtualenvwrapper support.
Now:

```
mkvirtualenv uncoverml
pip install gdal==2.0.1
```

Then download the uncoverml source, and install
```
cd code/uncover-ml
pip install -e .
```

`scontrol show job` shows the actual memory usage of the job.

Bracewell has 14 cores per node

use  --ntasks-per-node=14 and --nodes= whatever

