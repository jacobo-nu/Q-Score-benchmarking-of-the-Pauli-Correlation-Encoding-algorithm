#!/bin/bash
# submit_qscore_PCE_parallel.sh (FT3)
# ============================
# Igual que la versión de QMIO, pero añadiendo el eje de dispositivo
# (CPU/GPU) igual que en el pipeline normal: qubits < QUBIT_THRESHOLD ->
# CPU, qubits >= QUBIT_THRESHOLD -> GPU (por defecto 17 — GPU reservada
# para los n más grandes, como decidiste en el pipeline normal).
#
# Para cada n:
#   1. n_params (nº de parámetros variacionales) -> num_chunks(n)
#      (ver comentario largo en la versión de QMIO de este mismo fichero).
#   2. qubits -> device (CPU o GPU).
#   3. Se genera una tarea (n, chunk_index) por cada chunk, y se manda a
#      la cola de CPU o de GPU según el device de ese n.
#
# Con troceo por n_params, cada chunk ya tiene un coste acotado — así que,
# igual que en QMIO, aquí NO se reintroduce la escalera de tiempo
# short/medium/long por qubits del pipeline normal; se usa una única
# partición por dispositivo (TIME_LIMIT_CPU / TIME_LIMIT_GPU).
#
# !! NADA DE ESTO ESTÁ CALIBRADO CON TIEMPOS REALES !!
#
# NOTA: esta tubería es solo para simulación. No usar con PCE_BACKEND=QMIO_REAL
# (no aplica en FT3 de todas formas — el hardware real es solo QMIO).
#
# Uso:
#   ./submit_qscore_PCE_parallel.sh
#   QUBIT_THRESHOLD=18 CHUNK_SIZE_PARAMS=40 ./submit_qscore_PCE_parallel.sh

set -e

RUN_ID=$(date +%Y%m%d-%H%M%S)

mkdir -p "logs/${RUN_ID}"
mkdir -p "Resultados/PCE/${RUN_ID}/chunks"
mkdir -p "Imagenes/PCE/${RUN_ID}"

# =============================
# Lista maestra de tamaños de nodo — ÚNICA fuente de verdad.
# =============================
N_LIST_MASTER=(250)

# =============================
# Parámetros de troceo y dispositivo (ajustables sin editar el fichero)
# =============================
K="${K:-3}"
CHUNK_SIZE_PARAMS="${CHUNK_SIZE_PARAMS:-35}"
MAX_CHUNKS="${MAX_CHUNKS:-10}"
QUBIT_THRESHOLD="${QUBIT_THRESHOLD:-19}"        # CPU vs GPU, mismo criterio que el pipeline normal
NUM_WORKERS="${NUM_WORKERS:-16}"                  # procesos en paralelo por chunk, rama CPU
NUM_WORKERS_GPU="${NUM_WORKERS_GPU:-1}"          # ídem, rama GPU — 1 por defecto: una sola A100
                                                   # normalmente sirve una ejecución de AerSimulator
                                                   # a la vez, varios procesos podrían competir por
                                                   # ella en vez de acelerar. Sube esto solo si
                                                   # confirmas empíricamente que ayuda en tu GPU.
TIME_LIMIT_CPU="${TIME_LIMIT_CPU:-18:00:00}"  # partición 'medium' — igual que el pipeline normal
TIME_LIMIT_GPU="${TIME_LIMIT_GPU:-3-00:00:00}"
CPUS_PER_TASK="${CPUS_PER_TASK:-32}"

# =============================
# Parámetros del experimento PCE
# =============================
NUM_INSTANCES="${NUM_INSTANCES:-100}"
MAXITER="${MAXITER:-25}"
OPTIMIZER="${OPTIMIZER:-DIFFERENTIALEVOLUTION}"
SEED="${SEED:-42}"

# =============================
# Módulos
# =============================
module load cesga/2022 gcc/system qiskit/1.2.4-aer-gpu-cu11

# =============================
# Construir las dos listas planas de tareas (CPU y GPU)
# =============================
TASKS_CPU=""
TASKS_GPU=""
echo "=============================="
echo "RUN_ID: ${RUN_ID}"
echo "K=${K}  CHUNK_SIZE_PARAMS=${CHUNK_SIZE_PARAMS}  MAX_CHUNKS=${MAX_CHUNKS}  QUBIT_THRESHOLD=${QUBIT_THRESHOLD}  NUM_WORKERS=${NUM_WORKERS}"
for N in "${N_LIST_MASTER[@]}"; do
    QUBITS=$(python -c "from qscore_PCE import num_qubits; print(num_qubits(${N}, ${K}))")
    NPARAMS=$(python -c "
import math
from qscore_PCE import num_qubits, PCECircuit
q = num_qubits(${N}, ${K})
layers = math.ceil(${N} ** (1 - 1/${K}))
b = PCECircuit(size=q, p=layers); b.compile_circuit()
print(len(b.get_circuit().parameters))
")
    NUM_CHUNKS=$(python -c "import math; print(max(1, min(${MAX_CHUNKS}, math.ceil(${NPARAMS}/${CHUNK_SIZE_PARAMS}))))")
    if [ "$QUBITS" -lt "$QUBIT_THRESHOLD" ]; then DEVICE="CPU"; else DEVICE="GPU"; fi
    echo "  n=${N}: qubits=${QUBITS} n_params=${NPARAMS} -> num_chunks=${NUM_CHUNKS} device=${DEVICE}"
    for ((C=0; C<NUM_CHUNKS; C++)); do
        if [ "$DEVICE" = "CPU" ]; then
            TASKS_CPU="${TASKS_CPU} ${N}:${C}:${NUM_CHUNKS}"
        else
            TASKS_GPU="${TASKS_GPU} ${N}:${C}:${NUM_CHUNKS}"
        fi
    done
done
echo "Total tareas CPU: $(echo $TASKS_CPU | wc -w)   Total tareas GPU: $(echo $TASKS_GPU | wc -w)"
echo "=============================="

# =============================
# Lanzar el array de CPU (si hay algo que lanzar)
# =============================
if [ -n "$(echo $TASKS_CPU)" ]; then
    TASKS_ARR=($TASKS_CPU)
    ARRAY_SPEC="0-$((${#TASKS_ARR[@]}-1))"
    [ -n "$MAX_CONCURRENT_CPU" ] && ARRAY_SPEC="${ARRAY_SPEC}%${MAX_CONCURRENT_CPU}"
    sbatch -J "${RUN_ID}" \
        --partition=medium --qos=medium --time="${TIME_LIMIT_CPU}" \
        --cpus-per-task="${CPUS_PER_TASK}" \
        --array="${ARRAY_SPEC}" \
        --export=ALL,RUN_ID="${RUN_ID}",TASKS_STR="${TASKS_CPU}",K="${K}",NUM_WORKERS="${NUM_WORKERS}",\
NUM_INSTANCES="${NUM_INSTANCES}",MAXITER="${MAXITER}",OPTIMIZER="${OPTIMIZER}",SEED="${SEED}" \
        run_qscore_PCE_chunk_task_cpu.sh
fi

# =============================
# Lanzar el array de GPU (si hay algo que lanzar)
# =============================
if [ -n "$(echo $TASKS_GPU)" ]; then
    TASKS_ARR=($TASKS_GPU)
    ARRAY_SPEC="0-$((${#TASKS_ARR[@]}-1))"
    [ -n "$MAX_CONCURRENT_GPU" ] && ARRAY_SPEC="${ARRAY_SPEC}%${MAX_CONCURRENT_GPU}"
    sbatch -J "${RUN_ID}" \
        --partition=medium --qos=medium --time="${TIME_LIMIT_GPU}" \
        --cpus-per-task="${CPUS_PER_TASK}" \
        --array="${ARRAY_SPEC}" \
        --export=ALL,RUN_ID="${RUN_ID}",TASKS_STR="${TASKS_GPU}",K="${K}",NUM_WORKERS="${NUM_WORKERS_GPU}",\
NUM_INSTANCES="${NUM_INSTANCES}",MAXITER="${MAXITER}",OPTIMIZER="${OPTIMIZER}",SEED="${SEED}" \
        run_qscore_PCE_chunk_task_gpu.sh
fi

echo ""
echo "Cuando terminen TODAS las tareas de AMBOS arrays (squeue -u \$USER), fusiona y agrega:"
echo "  python merge_chunks_qscore_PCE.py --run-id ${RUN_ID}"
echo "  python aggregate_qscore_PCE.py --run-id ${RUN_ID}"