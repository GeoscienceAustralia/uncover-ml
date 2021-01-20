module purge
module load pbs
module load python3/3.7.4
module load gdal/3.0.2

GIT_HOME=$HOME/github  # where to checkout the uncover-ml repo?
VENVS=$HOME/venvs

cd $GIT_HOME

if [ ! -d $GIT_HOME/uncover-ml ]
then
     git clone --single-branch --branch sheece-jupyter https://github.com/GeoscienceAustralia/uncover-ml.git
     cd uncover-ml/jupyter
else
     cd uncover-ml
     git pull
     git checkout sheece-jupyter
     cd jupyter
fi


# createi virtual environment and install requirements packages
mkdir -p $VENVS 
python3 -m venv $VENVS/jupyter
source $VENVS/jupyter/bin/activate
pip install -r $GIT_HOME/uncover-ml/jupyter/requirements.txt


# After successful installation, you can submit a pbs job to run jupyter-notebook
#$ qsub jupyter_pbsjob.sh

