#!/bin/bash
#SBATCH --job-name=qscore_pce_qpu
#SBATCH --output=%x.o%j
#SBATCH --error=%x.o%j
#SBATCH --partition=qpu
#SBATCH --mem=4G
#SBATCH --time=02:00:00

# job_qpu.sh — QMIO, hardware REAL (partición 'qpu').
#
# !! ESTRICTAMENTE EN SERIE !! — nunca lances esto como array ni en paralelo
# con otro job que también pida la partición 'qpu'. QmioBackend.run() es
# síncrono (bloquea hasta tener resultado) y la cola de hardware es un
# recurso compartido muy limitado — un solo job qpu en vuelo a la vez.
#
# Uso (equivalente a qrun, pero fijo a esta partición; a diferencia de qrun,
# SÍ reenvía argumentos extra al script de Python):
#   sbatch --job-name=nombre --output=nombre.o%j --error=nombre.o%j job_qpu.sh script.py --arg1 valor1
#
# NOTA no verificada: la plantilla oficial de CESGA para la partición qpu
# carga "module load qmio-run", DISTINTO del módulo qmio-tools/0.2.1 que
# ya usas en job.sh. Puede que necesites cargar AMBOS a la vez — pruébalo
# primero con test_qmio_backend.py antes de lanzar nada del pipeline PCE
# completo contra esto:
#   sbatch --job-name=test_qmio_backend --output=test_qmio_backend.o%j \
#     --error=test_qmio_backend.o%j job_qpu.sh test_qmio_backend.py --backend real

module load qmio/hpc gcc/12.3.0 qiskit/2.2.3-python-3.11.9 qmio-tools/0.2.1-python-3.11.9
# module load qmio-run   # <- descomentar si el módulo de arriba no basta
#                            para conectar con la QPU (ver nota de arriba)

export PCE_BACKEND=QMIO_REAL
export PCE_SHOTS=2048   # obligatorio >0 — QMIO_REAL no admite modo exacto

python "$1" "${@:2}"