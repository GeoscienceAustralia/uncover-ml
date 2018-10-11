# Reuse the Shared Virtualenv in NCI


1. Clone the uncoverml repo into your home directory:

```bash
$ cd ~
$ git clone git@github.com:GeoscienceAustralia/uncover-ml.git
```

2. Copy the following lines and save it in `~/create_uncoverml_env.sh` in your home directory.

```
module unload intel-cc
module unload intel-cc
module load python3/3.5.2 python3/3.5.2-matplotlib
module load hdf5/1.8.10 gdal/2.0.0 geos/3.5.0 gcc/4.9.0
module load openmpi/1.8
source /g/data/ge3/john/venvs/class/bin/activate
```

3. Activate `uncoverml` virtualenv by issuing the following command

```bash
$ source ~/create_uncoverml_env.sh
``` 

4. Install `cubist`

```bash
$ cd ~/uncover-ml
$ ./cubist/makecubist .
```

 
 5. Make sure the tests work:

```bash
$ cd ~/uncover-ml
$ pytest tests
```

If the tests all pass, that's it! You can use `uncoverml` in NCI.
