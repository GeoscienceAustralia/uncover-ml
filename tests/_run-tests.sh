#!/bin/bash
set -exuo pipefail

echo "Building conda test environment…"
# move into your project
cd "${CIRCLE_WORKING_DIRECTORY:-$HOME/project}"

# install system deps
sudo apt-get update
sudo apt-get install -y \
    libblas-dev liblapack-dev \
    libatlas-base-dev gfortran libproj-dev \
    openmpi-bin libopenmpi-dev

# install miniconda under $HOME
wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
bash miniconda.sh -b -p $HOME/miniconda
export PATH="$HOME/miniconda/bin:$PATH"
eval "$(conda shell.bash hook)"

echo "Bootstrapping mamba…"
# install mamba into base so you can call it directly
conda install -n base -c conda-forge mamba -y

echo "Creating & activating uncover-ml-env with mamba…"
mamba env create -f environment.yml -y
conda activate uncover-ml-env

echo "Patching mlens..."
sed -i 's|from collections import Sequence|from collections.abc import Sequence|g' \
    ../miniconda/envs/uncover-ml-env/lib/python3.10/site-packages/mlens/externals/sklearn/type_of_target.py

echo "Running tests…"
mkdir -p test-results
pytest .

# make coverage
# codecov
# make partition_test

echo "Tests completed successfully!"
