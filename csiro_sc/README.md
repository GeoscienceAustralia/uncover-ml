# UncoverML on Bracewell

## Installation

login to `bracewell-i1.hpc.csiro.au` which is much more powerful than the login
node (so you can do the compilation). Your home directory etc will be shared.

Create a `.bashrc` that looks something like this:

```
module load tensorflow/1.1.0-py35-gpu
module load gdal
module load hdf5
module load openmpi
source $(which virtualenvwrapper.sh)
```
I didn't have much luck with zsh.

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

# Data

the folder for storing persistent job data (eg image files) is `$DATADIR`, 
which in bracewell is `/data/<your csiro id>`. Note it is not backed up.
It is also shared across the different clusters.
For small transfers, just use scp to the bracewell interactive node above.
For big transfers, you should go via the pearcey data moving node:

```
pearcey-dm.hpc.csiro.au
```
which has its data folder in the same location.

## Running

* Bracewell has 14 cores per node, so use  `--ntasks-per-node=14 --nodes=<whatever>`
* See the example job submission script in this folder.
* Use `sbatch <script>` to submit.
* `scontrol show <job>` shows the actual memory usage of the job.

