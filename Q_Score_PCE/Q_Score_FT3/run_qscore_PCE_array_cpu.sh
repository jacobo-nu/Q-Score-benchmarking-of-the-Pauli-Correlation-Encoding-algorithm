#!/bin/bash
#SBATCH -J qscore_pce
#SBATCH -o logs/%x/qscore_pce_cpu_%A_%a.out
#SBATCH -e logs/%x/qscore_pce_cpu_%A_%a.err
#SBATCH --time=2-18:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=16G
#SBATCH --partition=medium
#SBATCH --qos=medium
#SBATCH --array=0-3   # valor por defecto si se lanza a mano; submit_qscore_PCE_array.sh lo sobreescribe

# =============================================================================
# Verificación ya resuelta / pendiente para este script:
#
# 1. Partición/QOS: se usa "medium" (sin prefijo epyc_/clk_/hwl_), igual que
#    la versión GPU pero SIN --gres/--gpus — así SLURM asigna cualquier nodo
#    disponible de esa partición sin pedir la tarjeta, sin competir por ese
#    recurso.
#
# 2. Módulo de Qiskit en un nodo sin GPU física: antes de la primera tanda
#    real, comprueba que qiskit_aer importa y corre en CPU sin depender de
#    libcudart/el driver de NVIDIA, con un test corto:
#
#      srun -p medium -q medium -t 00:02:00 --cpus-per-task=2 --mem=2G \
#        bash -c 'module load cesga/2022 gcc/system qiskit/1.2.4-aer-gpu-cu11 && \
#                 python -c "from qiskit_aer import AerSimulator; s = AerSimulator(device=\"CPU\"); print(\"OK:\", s)"'
#
#    Si imprime "OK: ..." sin errores de import, el módulo ya vale tal cual.
# =============================================================================

# NOTA: --partition/--qos/--time de arriba son solo valores de RESPALDO
# para si se lanza este .sh a mano. Cuando se usa el wrapper
# submit_qscore_PCE_array.sh (versión con reparto por tiempo), estos tres
# se sobreescriben en la línea de sbatch según el nº de qubits del
# sublistado que le toque a este array (short o medium, según
# QUBIT_THRESHOLD_CPU_SHORT) — sbatch da prioridad a los flags de línea de
# comandos sobre el #SBATCH del fichero.
#
# NOTA: este script NO lleva N_LIST hardcodeado. Lo recibe de
# submit_qscore_PCE_array.sh vía la variable de entorno N_LIST_STR
# (--export=ALL,N_LIST_STR="..."), que es quien decide qué tamaños de nodo
# van a CPU (los que NO superan QUBIT_THRESHOLD) y le pasa también --array
# con el rango correcto para ESTA sublista.
#
# Si se lanza este .sh directamente con sbatch (sin el wrapper), N_LIST_STR
# no existe y se usa N_LIST_DEFAULT de más abajo — ajústala a mano si haces
# esto, junto con --array en la línea de sbatch (o edita el #SBATCH de arriba).

# =============================
# Módulos (FT3, nodos sin GPU — ver punto 2 de la nota de arriba)
# =============================
module load cesga/2022 gcc/system qiskit/1.2.4-aer-gpu-cu11

# =============================
# Configuración del simulador Aer (leída por qscore_PCE.py vía
# get_sim_device() / get_sim_method() / get_shots()).
# =============================
export PCE_DEVICE=CPU
export PCE_SIM_METHOD="statevector"   # "statevector" | "matrix_product_state" | ...
export PCE_SHOTS=8192                    # 0 = modo exacto (statevector); >0 = modo shots

# =============================
# Lista de tamaños de nodo a evaluar (una tarea del array por elemento).
# Viene de submit_qscore_PCE_array.sh; N_LIST_DEFAULT solo se usa si se
# lanza este .sh a mano, sin pasar por el wrapper.
# =============================
N_LIST_DEFAULT=(10 25 50 75 100 150)
if [ -n "$N_LIST_STR" ]; then
    read -ra N_LIST <<< "$N_LIST_STR"
else
    N_LIST=("${N_LIST_DEFAULT[@]}")
fi

# =============================
# Parámetros del experimento PCE
# =============================
NUM_INSTANCES=100
K="${K:-2}"   # viene del wrapper (misma K usada para el reparto CPU/GPU); fallback si se lanza a mano
MAXITER=25
OPTIMIZER=DIFFERENTIALEVOLUTION
SEED=42

# =============================
# RUN_ID y carpeta de resultados
# =============================
# RUN_ID llega heredado del wrapper submit_qscore_PCE_array.sh (--export=ALL,RUN_ID=...).
# Si se lanza este .sh directamente con sbatch (sin el wrapper), cae en "manual"
# — en ese caso hay que crear logs/manual/ A MANO antes de mandar el job,
# porque -o/-e ya se resuelven al arrancar la tarea, antes de esta línea.
RUN_ID="${RUN_ID:-manual}"
OUTDIR="Resultados/PCE/${RUN_ID}"
mkdir -p "$OUTDIR"   # esta sí puede crearse aquí, se usa más abajo en el script

# =============================
# Seleccionar el n correspondiente a esta tarea del array
# =============================
IDX=$SLURM_ARRAY_TASK_ID
N=${N_LIST[$IDX]}

echo "=============================="
echo "RUN_ID: $RUN_ID"
echo "Array ID: $IDX  (sublista CPU: ${N_LIST[*]})"
echo "num_nodes: $N"
echo "num_instances: $NUM_INSTANCES"
echo "optimizer: $OPTIMIZER"
echo "device: $PCE_DEVICE"
echo "sim_method: $PCE_SIM_METHOD"
echo "shots: $PCE_SHOTS"
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