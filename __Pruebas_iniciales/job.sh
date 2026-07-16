#!/bin/bash
#SBATCH -J qscore_pce
#SBATCH -o qscore_pce_%A_%a.out
#SBATCH -e qscore_pce_%A_%a.err
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --partition=ilk
#SBATCH --array=0-8   # una tarea por cada valor de N_LIST (ajustar al tamaño de la lista)

# =============================
# Módulos
# =============================
module load qmio/hpc gcc/12.3.0 qiskit/2.2.3-python-3.11.9 qmio-tools/0.2.1-python-3.11.9

# =============================
# Lista de tamaños de nodo a evaluar (una tarea del array por elemento)
# =============================
N_LIST=(4 5 6 7 8 9 10 11 12)

NUM_INSTANCES=50
K=2
MAXITER=30
OPTIMIZER=DIFFERENTIALEVOLUTION
SEED=42
OUTDIR=Resultados/PCE

# =============================
# Seleccionar el n correspondiente a esta tarea del array
# =============================
IDX=$SLURM_ARRAY_TASK_ID
N=${N_LIST[$IDX]}

echo "=============================="
echo "Array ID: $IDX"
echo "num_nodes: $N"
echo "num_instances: $NUM_INSTANCES"
echo "optimizer: $OPTIMIZER"
echo "=============================="

# =============================
# Ejecutar script Python para este n
# =============================
python -u run_qscore_PCE_paralelo.py \
    --num_nodes "$N" \
    --num_instances "$NUM_INSTANCES" \
    --k "$K" \
    --pce_maxiter "$MAXITER" \
    --pce_optimizer "$OPTIMIZER" \
    --seed "$SEED" \
    --outdir "$OUTDIR"

echo "Fecha fin: $(date)"