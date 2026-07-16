#!/bin/bash
#SBATCH --time=08:00:00
#SBATCH --partition=ilk
#SBATCH --mem=16G
#SBATCH --cpus-per-task=16

# job.sh — QMIO, usado por la función de shell "qrun" para el flujo en serie:
#   qrun () { sbatch --job-name=${1%.py} --output=${1%.py}.o%j --error=${1%.py}.o%j job.sh $1 }
#
# qrun no acepta flags extra, así que --time/--mem/--cpus-per-task se ajustan
# aquí a mano antes de cada qrun si el script que vas a correr necesita más
# o menos recursos (por ejemplo, para n grandes en run_qscore_PCE.py).
#
# NOTA: este .sh SÍ reenvía argumentos extra al script de Python
# ("${@:2}"), pero la función qrun definida arriba solo acepta $1 (el
# nombre del script) — no reenvía nada más aunque se lo pases. Para
# scripts que necesitan argumentos (como test_qmio_backend.py --backend
# fake), lanza sbatch directamente en vez de usar qrun:
#   sbatch --job-name=nombre --output=nombre.o%j --error=nombre.o%j job.sh script.py --arg1 valor1

module load qmio/hpc gcc/12.3.0 qiskit/2.2.3-python-3.11.9 qmio-tools/0.2.1-python-3.11.9

export PCE_BACKEND=QMIO_FAKE
export PCE_SHOTS=2048   # obligatorio >0 — QMIO_REAL no admite modo exacto

python "$1" "${@:2}"