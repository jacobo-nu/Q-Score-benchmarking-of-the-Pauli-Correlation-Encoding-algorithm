#!/bin/bash
#SBATCH -J qscore_pce_chunk_cpu
#SBATCH -o logs/%x/qscore_pce_cpu_%A_%a.out
#SBATCH -e logs/%x/qscore_pce_cpu_%A_%a.err
#SBATCH --time=2-18:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
#SBATCH --partition=medium
#SBATCH --array=0-0   # valor por defecto si se lanza a mano; el wrapper lo sobreescribe

# run_qscore_PCE_chunk_task_cpu.sh — FT3 (CPU)
# ================================
# Ejecuta UN chunk (n, chunk_index, num_chunks) de la lista plana TASKS_STR
# construida por submit_qscore_PCE_parallel.sh. Guarda el resultado
# PARCIAL en Resultados/PCE/<RUN_ID>/chunks/.
#
# FIX (10/07/2026): se pedían 32 cores pero solo se usaban NUM_WORKERS
# (por defecto 4) procesos. Sin fijar los hilos BLAS/OMP, cada proceso worker
# auto-detectaba los 32 cores del nodo y lanzaba hasta 32 hilos cada uno
# (hasta ~128 hilos compitiendo por 32 cores físicos), generando contención
# y wall-clock mucho mayor de lo esperado (CPU Efficiency ~4.5% observado).
#
# Se mantiene --cpus-per-task=32 (el nodo de la particion medium probablemente
# se asigna de forma exclusiva por job, asi que reservar menos no libera nada)
# y en su lugar se sube NUM_WORKERS para aprovechar los 32 cores realmente,
# dejando un pequeno margen para el proceso orquestador/OS. Cada libreria se
# fija a 1 hilo por proceso para que el paralelismo real lo controle solo
# ProcessPoolExecutor (NUM_WORKERS), evitando la sobre-suscripcion de hilos.
#
# Vigilar --mem al escalar a n grandes (100, 200...): con mas workers activos
# simultaneamente y mas qubits por instancia, 16G puede quedarse corto.

module load cesga/2022 gcc/system qiskit/1.2.4-aer-gpu-cu11

# --- Fix de contención de hilos BLAS/OMP ---
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

export PCE_DEVICE="CPU"
export PCE_SIM_METHOD="statevector"
export PCE_SHOTS="${PCE_SHOTS:-4096}"
export PCE_BACKEND="${PCE_BACKEND:-AER}"

K="${K:-2}"
NUM_WORKERS="${NUM_WORKERS:-28}"   # antes 4; 28 de 32 cores, margen para orquestador/OS
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
echo "num_workers: $NUM_WORKERS   cpus-per-task: ${SLURM_CPUS_PER_TASK:-?}"
echo "OMP_NUM_THREADS=$OMP_NUM_THREADS OPENBLAS_NUM_THREADS=$OPENBLAS_NUM_THREADS MKL_NUM_THREADS=$MKL_NUM_THREADS"
echo "device: $PCE_DEVICE   optimizer: $OPTIMIZER   maxiter: $MAXITER   k: $K"
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