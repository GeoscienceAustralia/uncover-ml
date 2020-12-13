set -x
cd ~
module load python3/3.7.4
module load gdal/3.0.2
module load openmpi/3.0.4 
python3 -m venv ~/Venvs/uncover-ml
source ~/Venvs/uncover-ml/bin/activate
git clone https://github.com/GeoscienceAustralia/uncover-ml.git
cd uncover-ml
git checkout develop
pip install -U pip setuptools
pip install -r requirements-pykrige.txt
pip install .
export UNCOVERML_SRC=$(pwd ~)
uncoverml learn tests/test_data/sirsam/random_forest/sirsam_Na_randomforest.yaml
