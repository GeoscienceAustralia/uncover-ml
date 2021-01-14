wget -O jupyter_install.sh https://raw.githubusercontent.com/GeoscienceAustralia/uncover-ml/sheece-jupyter/jupyter/jupyter_install.sh
chmod +x jupyter_install.sh
./jupyter_install.sh
source $HOME/venvs/jupyter/bin/activate
jupyter notebook --generate-config -y
jupyter notebook password
cd  $HOME
qsub $HOME/uncover-ml/jupyter/jupyter.host.sh 
nqstat_anu
more client_cmd