#!/bin/bash
#SBATCH -J qscore_pce_chunk
#SBATCH -o logs/%x/qscore_pce_%A_%a.out
#SBATCH -e logs/%x/qscore_pce_%A_%a.err
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --partition=ilk
#SBATCH --array=0-0   # valor por defecto si se lanza a mano; el wrapper lo sobreescribe

# run_qscore_PCE_chunk_task.sh — QMIO
# ================================
# Ejecuta UN chunk (n, chunk_index, num_chunks) de la lista plana de tareas
# construida por submit_qscore_PCE_parallel.sh (TASKS_STR, formato
# "n:chunk_index:num_chunks" separados por espacio). Guarda el resultado
# PARCIAL en Resultados/PCE/<RUN_ID>/chunks/ — hace falta
# merge_chunks_qscore_PCE.py después para combinar todos los chunks de
# cada n en su n_<N>.h5 final.

module load qmio/hpc gcc/12.3.0 qiskit/2.2.3-python-3.11.9 qmio-tools/0.2.1-python-3.11.9

export PCE_DEVICE="CPU"
export PCE_SIM_METHOD="statevector"
export PCE_SHOTS="${PCE_SHOTS:-0}"
export PCE_BACKEND="${PCE_BACKEND:-AER}"

K="${K:-2}"
NUM_WORKERS="${NUM_WORKERS:-4}"
NUM_INSTANCES="${NUM_INSTANCES:-30}"
MAXITER="${MAXITER:-15}"
OPTIMIZER="${OPTIMIZER:-DIFFERENTIALEVOLUTION}"
SEED="${SEED:-42}"

if [ -z "$TASKS_STR" ]; then
    echo "ERROR: TASKS_STR no está definida — este script está pensado para "
    echo "lanzarse desde submit_qscore_PCE_parallel.sh, no directamente."
    exit 1
fi

read -ra TASKS_ARR <<< "$TASKS_STR"
ENTRY="${TASKS_ARR[$SLURM_ARRAY_TASK_ID]}"
IFS=':' read -r N CHUNK_INDEX NUM_CHUNKS <<< "$ENTRY"

RUN_ID="${RUN_ID:-manual}"
OUTDIR="Resultados/PCE/${RUN_ID}/chunks"
mkdir -p "$OUTDIR"

echo "=============================="
echo "RUN_ID: $RUN_ID"
echo "Array ID: $SLURM_ARRAY_TASK_ID  (entrada: $ENTRY)"
echo "num_nodes: $N   chunk: $((CHUNK_INDEX+1))/$NUM_CHUNKS"
echo "num_instances (total del n): $NUM_INSTANCES"
echo "num_workers: $NUM_WORKERS"
echo "optimizer: $OPTIMIZER   maxiter: $MAXITER   k: $K"
echo "backend: $PCE_BACKEND   shots: $PCE_SHOTS"
echo "=============================="

python -u run_qscore_PCE_chunk.py \
    --num_nodes "$N" \
    --num_instances "$NUM_INSTANCES" \
    --chunk_index "$CHUNK_INDEX" \
    --num_chunks "$NUM_CHUNKS" \
    --num_workers "$NUM_WORKERS" \
    --k "$K" \
    --pce_maxiter "$MAXITER" \
    --pce_optimizer "$OPTIMIZER" \
    --seed "$SEED" \
    --outdir "$OUTDIR"

echo "Fecha fin: $(date)"