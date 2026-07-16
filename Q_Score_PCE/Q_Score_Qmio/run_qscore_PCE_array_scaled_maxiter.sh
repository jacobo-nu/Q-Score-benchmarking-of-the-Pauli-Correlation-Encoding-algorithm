#!/bin/bash
#SBATCH -J qscore_pce_scaled_maxiter
#SBATCH -o logs/%x/qscore_pce_%A_%a.out
#SBATCH -e logs/%x/qscore_pce_%A_%a.err
#SBATCH --time=16:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --partition=ilk
#SBATCH --array=0-7   # una tarea por cada valor de N_LIST (ajustar al tamaño de la lista)

# run_qscore_PCE_array_scaled_maxiter.sh — QMIO
# ================================================
# Variante experimental de run_qscore_PCE_array.sh: en vez de un MAXITER
# fijo para todo el N_LIST, escala MAXITER linealmente con n —
#
#   MAXITER(n) = ceil(MAXITER_INTERCEPT + MAXITER_SLOPE * n)
#
# Motivación: Le Ber et al. (Nat Commun 16, 1219 (2025), arXiv del PCE)
# observan que el número de "epochs" (pasos del optimizador clásico)
# necesario para entrenar escala aproximadamente lineal con m — en tu
# caso, m = num_nodes = n. Con un MAXITER fijo (como en
# run_qscore_PCE_array.sh normal) podrías estar sub-entrenando
# sistemáticamente los n grandes respecto a los pequeños.
#
# !! LOS VALORES DE MAXITER_INTERCEPT/MAXITER_SLOPE SON UN PUNTO DE
# PARTIDA, NO ESTÁN CALIBRADOS !! El paper mide esa escala lineal para
# Adam/SLSQP con su propio ansatz — no hay garantía de que la misma
# pendiente aplique a tu COBYLA/DIFFERENTIALEVOLUTION/ROTOSOLVE con
# PCECircuit. El objetivo de esta tanda es observar SI existe una
# tendencia parecida en tu caso, no reproducir la pendiente exacta del
# paper.

# =============================
# Módulos
# =============================
module load qmio/hpc gcc/12.3.0 qiskit/2.2.3-python-3.11.9 qmio-tools/0.2.1-python-3.11.9

# =============================
# Configuración del simulador Aer
# =============================
export PCE_DEVICE="CPU"
export PCE_SIM_METHOD="statevector"
export PCE_SHOTS=0

# =============================
# Lista de tamaños de nodo — n=10 a 80, paso 10 (tu rango de interés ahora)
# =============================
N_LIST=(10 20 30 40 50 60 70 80)

# =============================
# Fórmula de MAXITER escalado — ajustable sin editar el fichero:
#   MAXITER_INTERCEPT=20 MAXITER_SLOPE=0.2 ./tu_lanzador
# =============================
MAXITER_INTERCEPT="${MAXITER_INTERCEPT:-20}"
MAXITER_SLOPE="${MAXITER_SLOPE:-0.2}"

# =============================
# Resto de parámetros del experimento (fijos — solo MAXITER varía por n,
# para que la comparación entre n aísle ese único factor)
# =============================
NUM_INSTANCES=40
K=2
OPTIMIZER=DIFFERENTIALEVOLUTION
SEED=42

# =============================
# RUN_ID y carpeta de resultados
# =============================
RUN_ID="${RUN_ID:-manual}"
OUTDIR="Resultados/PCE/${RUN_ID}"
mkdir -p "$OUTDIR"

# =============================
# Seleccionar el n de esta tarea y calcular su MAXITER escalado
# =============================
IDX=$SLURM_ARRAY_TASK_ID
N=${N_LIST[$IDX]}
MAXITER=$(python3 -c "import math; print(max(1, math.ceil(${MAXITER_INTERCEPT} + ${MAXITER_SLOPE} * ${N})))")

echo "=============================="
echo "RUN_ID: $RUN_ID"
echo "Array ID: $IDX"
echo "num_nodes: $N"
echo "MAXITER_INTERCEPT: $MAXITER_INTERCEPT  MAXITER_SLOPE: $MAXITER_SLOPE"
echo "maxiter (escalado para este n): $MAXITER"
echo "num_instances: $NUM_INSTANCES"
echo "optimizer: $OPTIMIZER"
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