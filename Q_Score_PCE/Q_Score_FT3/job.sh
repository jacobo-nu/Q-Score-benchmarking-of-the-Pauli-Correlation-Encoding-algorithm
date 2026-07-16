#!/bin/bash
#SBATCH --job-name=qscore_pce
#SBATCH --partition=short
#SBATCH --qos=short
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=32
#SBATCH --time=03:00:00
#SBATCH --mem=16G
#SBATCH --output=%x.o%j
#SBATCH --error=%x.o%j

# job.sh — FT3 (GPU), usado por qrun para el flujo en serie.
# Homólogo del job.sh de QMIO, pero con petición de GPU y el módulo de
# Qiskit con Aer-GPU ya autocontenido (no hace falta cargar cuda/ aparte).

module load cesga/2022 gcc/system qiskit/1.2.4-aer-gpu-cu11

# Le dice a qscore_PCE.py que use device="GPU" en AerSimulator.
export PCE_DEVICE=GPU

python $1