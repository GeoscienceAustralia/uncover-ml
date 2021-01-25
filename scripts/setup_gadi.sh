cd ~
module load python3/3.7.4
module load gdal/3.0.2
module load openmpi/3.0.4 
python3 -m venv ~/venvs/uncoverml
python3 -m venv ~/venvs/uncoverml

source ~/venvs/uncoverml/bin/activate
git clone https://github.com/GeoscienceAustralia/uncover-ml.git
cd uncover-ml
git checkout develop
pip install -U pip setuptools
pip install -r requirements.txt
pip install -e .
export UNCOVERML_SRC=$(pwd ~)
uncoverml learn tests/test_data/sirsam/random_forest/sirsam_Na_randomforest.yaml
uncoverml gridsearch configs/optimisation.yaml