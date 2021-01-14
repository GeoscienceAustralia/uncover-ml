module purge
module load pbs
module load python3/3.7.4
module load gdal/3.0.2

cd $HOME

if [ ! -d $HOME/uncover-ml ]
then
     git clone --single-branch --branch sheece-jupyter https://github.com/GeoscienceAustralia/uncover-ml.git
     cd uncover-ml/jupyter
else
     cd uncover-ml
     git pull
     git checkout sheece-jupyter
     cd jupyter
fi

mkdir -p $HOME/venvs/jupyter
python3 -m venv $HOME/venvs/jupyter
$HOME/venvs/jupyter/bin/python -m pip install -r $HOME/uncover-ml/jupyter/requirements.txt


chmod +x jupyter.node.sh 
chmod +x jupyter.host.sh 
