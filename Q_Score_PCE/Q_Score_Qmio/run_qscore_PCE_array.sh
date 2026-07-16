#!/bin/bash
#SBATCH -J qscore_pce
#SBATCH -o logs/%x/qscore_pce_%A_%a.out
#SBATCH -e logs/%x/qscore_pce_%A_%a.err
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --partition=ilk
#SBATCH --array=0-4   # valor por defecto si se lanza a mano; el wrapper lo sobreescribe

# run_qscore_PCE_array.sh — QMIO
# ================================
# NOTA: este script YA NO lleva N_LIST ni --time fijos pensados para todo
# el rango. Los recibe de submit_qscore_PCE_array.sh vía N_LIST_STR
# (--export) y --time (línea de sbatch, tiene prioridad sobre el #SBATCH
# de arriba) — el wrapper decide el --time correcto según el nº de qubits
# de cada n (ver ese fichero), para caer en el QOS automático adecuado
# (ilk_short/ilk_medium/ilk_long — QMIO NO acepta --qos explícito, lo
# asigna solo a partir de --time).
#
# Si se lanza este .sh directamente con sbatch (sin el wrapper), N_LIST_STR
# no existe y se usa N_LIST_DEFAULT de más abajo — en ese caso el --time
# de arriba (06:00:00, ilk_short) es el que aplica, ajústalo a mano si el
# n que vayas a correr necesita más.

# =============================
# Módulos
# =============================
module load qmio/hpc gcc/12.3.0 qiskit/2.2.3-python-3.11.9 qmio-tools/0.2.1-python-3.11.9

# =============================
# Configuración del simulador Aer
# =============================
export PCE_DEVICE="CPU"
export PCE_SIM_METHOD="statevector"
export PCE_SHOTS="${PCE_SHOTS:-2048}"

# =============================
# Lista de tamaños de nodo a evaluar (una tarea del array por elemento).
# Viene de submit_qscore_PCE_array.sh; N_LIST_DEFAULT solo se usa si se
# lanza este .sh a mano, sin pasar por el wrapper.
# =============================
N_LIST_DEFAULT=(10 25 50 75 100)
if [ -n "$N_LIST_STR" ]; then
    read -ra N_LIST <<< "$N_LIST_STR"
else
    N_LIST=("${N_LIST_DEFAULT[@]}")
fi

# =============================
# Parámetros del experimento PCE
# =============================
NUM_INSTANCES=100
K="${K:-2}"   # viene del wrapper (misma K usada para el reparto); fallback si se lanza a mano
MAXITER=25
OPTIMIZER=DIFFERENTIALEVOLUTION
SEED=42

# =============================
# RUN_ID y carpeta de resultados
# =============================
RUN_ID="${RUN_ID:-manual}"
OUTDIR="Resultados/PCE/${RUN_ID}"
mkdir -p "$OUTDIR"

# =============================
# Seleccionar el n correspondiente a esta tarea del array
# =============================
IDX=$SLURM_ARRAY_TASK_ID
N=${N_LIST[$IDX]}

echo "=============================="
echo "RUN_ID: $RUN_ID"
echo "Array ID: $IDX  (sublista: ${N_LIST[*]})"
echo "num_nodes: $N"
echo "k: $K"
echo "num_instances: $NUM_INSTANCES"
echo "optimizer: $OPTIMIZER"
echo "sim_device: $PCE_DEVICE"
echo "sim_method: $PCE_SIM_METHOD"
echo "shots: $PCE_SHOTS"
echo "time_limit (SLURM): $(squeue -h -j "$SLURM_JOB_ID" -o %l 2>/dev/null || echo desconocido)"
echo "=============================="

# =============================
# Ejecutar script Python para este n
# =============================
python -u run_qscore_PCE_single_n.py \
    --num_nodes "$N" \
    --num_instances "$NUM_INSTANCES" \
    --k "$K" \
    --pce_maxiter "$MAXITER" \
    --pce_optimizer "$OPTIMIZER" \
    --seed "$SEED" \
    --outdir "$OUTDIR"

echo "Fecha fin: $(date)"