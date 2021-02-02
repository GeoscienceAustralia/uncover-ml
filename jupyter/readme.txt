## Installation procedure

# load the required modules
module purge
module load pbs
module load python3/3.7.4
module load gdal/3.0.2

# clone the uncover-ml repository
git clone https://github.com/GeoscienceAustralia/uncover-ml.git $HOME/github/uncover-ml

# create virtual environment for uncover-ml
python3 -m venv $HOME/venvs/uncoverml_gadi
source $HOME/venvs/uncoverml_gadi/bin/activate
pip install -r $HOME/github/uncover-ml/jupyter/requirements.txt

# setup password to access jupyter notebook server
# /home/547/sg4953/.jupyter/jupyter_lab_config.py
# /home/547/sg4953/.jupyter/jupyter_server_config.json
jupyter lab --generate-config -y
jupyter lab password

# ensure jupyter_pbsjob script is excuable
chmod +x $HOME/github/uncover-ml/jupyter/jupyter_pbsjob.sh

## Usage

# run the following commands to run jupyter server on NCI Node
cd  $HOME/github # the directory level that will be accessible from jupyter server
qsub $HOME/github/uncover-ml/jupyter/jupyter_pbsjob.sh

# wait for the job to start
nqstat_anu

# copy the port forwarding command
more client_cmd


## Port forwarding on Local machine

# run the following command in bash on local machine to access jupyter notebook
# through browser in local machine
ssh -N -L 8391:gadi-gpu-v100-0103.gadi.nci.org.au:8391 sg4953@gadi.nci.org.au &
