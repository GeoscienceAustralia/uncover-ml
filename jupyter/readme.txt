##Installation procedure

module purge
module load pbs
module load python3/3.7.4
module load gdal/3.0.2




mkdir -p $HOME/github
cd $HOME/github
git clone --single-branch --branch sheece-jupyter https://github.com/GeoscienceAustralia/uncover-ml.git



mkdir -p $HOME/venvs
python3 -m venv $HOME/venvs/uncoverml_gadi
source $HOME/venvs/uncoverml_gadi/bin/activate
pip install -r $HOME/github/uncover-ml/jupyter/requirements.txt

jupyter notebook --generate-config -y
jupyter notebook password


chmod +x $HOME/github/uncover-ml/jupyter/jupyter_pbsjob.sh
cd  $HOME/github
qsub $HOME/github/uncover-ml/jupyter/jupyter_pbsjob.sh
nqstat_anu
more client_cmd