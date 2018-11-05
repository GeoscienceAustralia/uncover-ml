# Reuse the Shared Virtualenv in NCI

1. Activate `uncoverml` virtualenv by issuing the following command

```bash
$ source /g/data/ge3/john/uncover-ml/create_uncoverml_env.sh
``` 
That is it. You are all set to use `uncoverml` in NCI.

2. If you are using this shared `uncoverml` env, your job submission script can be:

```bash
#!/bin/bash
#PBS -P ge3
#PBS -q express
#PBS -l walltime=0:15:00,mem=32GB,ncpus=16,jobfs=100GB
#PBS -l wd
#PBS -j oe

source /g/data/ge3/john/uncover-ml/create_uncoverml_env.sh

# run command
mpirun --mca mpi_warn_on_fork 0 uncoverml learn your_yaml_file.yaml -p 10
mpirun --mca mpi_warn_on_fork 0 uncoverml predict your_tranined_model_file.model -p 10
``` 

3. To run the demo locally:

First copy the demo files locally.
```bash
cp -r /g/data/ge3/john/jobs/sirsam/demo/ ~/demo
```

Then run the demo locally:

```bash
source /g/data/ge3/john/uncover-ml/create_uncoverml_env.sh
cd ~/demo
uncoverml learn sirsam_Na.yaml -p 4
uncoverml predict sirsam_Na.model -p 10
```
