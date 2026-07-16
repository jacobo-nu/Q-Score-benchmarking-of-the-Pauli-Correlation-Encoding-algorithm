#!/bin/bash
#SBATCH -p ilk
#SBATCH --mem=8G
#SBATCH -t 0:45:0

module load qmio/hpc gcc/12.3.0 qiskit/2.2.3-python-3.11.9 qmio-tools/0.2.1-python-3.11.9

pip install --user --quiet networkx xarray

python $1