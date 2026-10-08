#!/bin/bash
# submit_qscore_PCE_parallel.sh (QMIO)
# ============================
# Tubería con troceo de instancias por n (Opción B) + multiprocessing
# dentro de cada chunk (Opción A). Para cada n de la lista maestra:
#
#   1. Calcula n_params (nº de parámetros variacionales del ansatz para
#      ese n, con self.k) — es lo que realmente determina el coste de
#      differential_evolution (tamaño de población = popsize * n_params),
#      más fiable que usar qubits directamente como proxy de coste.
#   2. num_chunks(n) = clamp(ceil(n_params / CHUNK_SIZE_PARAMS), 1, MAX_CHUNKS)
#      — más chunks para los n más caros, menos (incluso 1) para los baratos.
#   3. Genera una tarea SLURM por cada (n, chunk_index) — un array donde
#      cada tarea calcula solo una FRACCIÓN de las instancias de un n, en
#      vez de un array con una tarea por n completo como en el pipeline
#      normal (run_qscore_PCE_array.sh).
#
# QMIO no tiene GPU, así que aquí solo hay un eje (nº de chunks), no
# reparto de dispositivo. Tampoco se reintroduce aquí la escalera de
# tiempo short/medium/long por qubits del pipeline normal — con troceo
# por n_params, cada chunk ya tiene un coste acotado y comparable
# independientemente de n, así que un único --time generoso para todas
# las tareas del array debería bastar (ver TIME_LIMIT más abajo).
#
# !! NADA DE ESTO ESTÁ CALIBRADO CON TIEMPOS REALES !! CHUNK_SIZE_PARAMS,
# MAX_CHUNKS, NUM_WORKERS y TIME_LIMIT son puntos de partida razonados,
# no medidos. Ajusta con tus propios duration_seconds de los ficheros de
# chunk (dentro de Resultados/PCE/<RUN_ID>/chunks/ antes de fusionar).
#
# NOTA: esta tubería es solo para simulación (PCE_BACKEND=AER, el valor
# por defecto). PCE_BACKEND=QMIO_REAL NO es compatible con el troceo por
# chunks (un solo chip físico, sin paralelismo posible) — para hardware
# real usa job_qpu.sh / test_qmio_backend.py del pipeline normal, no este.
#
# Uso:
#   ./submit_qscore_PCE_parallel.sh
#   CHUNK_SIZE_PARAMS=30 MAX_CHUNKS=8 ./submit_qscore_PCE_parallel.sh

set -e

RUN_ID=$(date +%Y%m%d-%H%M%S)

mkdir -p "logs/${RUN_ID}"
mkdir -p "Resultados/PCE/${RUN_ID}/chunks"
mkdir -p "Imagenes/PCE/${RUN_ID}"

# =============================
# Lista maestra de tamaños de nodo — ÚNICA fuente de verdad.
# =============================
N_LIST_MASTER=(250 275 300 325 350)

# =============================
# Parámetros de troceo (ajustables sin editar el fichero)
# =============================
K="${K:-2}"
CHUNK_SIZE_PARAMS="${CHUNK_SIZE_PARAMS:-30}"   # nº de parámetros variacionales "objetivo" por chunk
MAX_CHUNKS="${MAX_CHUNKS:-16}"                   # techo de chunks por n, para no explotar el nº de tareas
NUM_WORKERS="${NUM_WORKERS:-12}"                 # procesos en paralelo dentro de cada chunk (Opción A)
TIME_LIMIT="${TIME_LIMIT:-1-00:00:00}"          # --time único para todo el array (ver nota arriba)
CPUS_PER_TASK="${CPUS_PER_TASK:-32}"

# =============================
# Parámetros del experimento PCE (fijos para todo n — solo num_chunks varía)
# =============================
NUM_INSTANCES="${NUM_INSTANCES:-100}"
MAXITER="${MAXITER:-25}"
OPTIMIZER="${OPTIMIZER:-DIFFERENTIALEVOLUTION}"
SEED="${SEED:-42}"

# =============================
# Módulos (necesarios ya aquí: se usa python para calcular n_params/num_chunks)
# =============================
module load qmio/hpc gcc/12.3.0 qiskit/2.2.3-python-3.11.9 qmio-tools/0.2.1-python-3.11.9

# =============================
# Construir la lista plana de tareas (n:chunk_index:num_chunks)
# =============================
TASKS=""
echo "=============================="
echo "RUN_ID: ${RUN_ID}"
echo "K=${K}  CHUNK_SIZE_PARAMS=${CHUNK_SIZE_PARAMS}  MAX_CHUNKS=${MAX_CHUNKS}  NUM_WORKERS=${NUM_WORKERS}"
for N in "${N_LIST_MASTER[@]}"; do
    NPARAMS=$(python -c "
import math
from qscore_PCE import num_qubits, PCECircuit
q = num_qubits(${N}, ${K})
layers = math.ceil(${N} ** (1 - 1/${K}))
b = PCECircuit(size=q, p=layers); b.compile_circuit()
print(len(b.get_circuit().parameters))
")
    NUM_CHUNKS=$(python -c "import math; print(max(1, min(${MAX_CHUNKS}, math.ceil(${NPARAMS}/${CHUNK_SIZE_PARAMS}))))")
    echo "  n=${N}: n_params=${NPARAMS} -> num_chunks=${NUM_CHUNKS}"
    for ((C=0; C<NUM_CHUNKS; C++)); do
        TASKS="${TASKS} ${N}:${C}:${NUM_CHUNKS}"
    done
done
echo "Total de tareas: $(echo $TASKS | wc -w)"
echo "=============================="

TASKS_ARR=($TASKS)
NUM_TASKS=${#TASKS_ARR[@]}

# =============================
# Lanzar el array (throttling opcional con MAX_CONCURRENT, p.ej. si
# chocas con MaxJobsPU/MaxSubmitPU del QOS que te toque según TIME_LIMIT)
# =============================
ARRAY_SPEC="0-$((NUM_TASKS-1))"
if [ -n "$MAX_CONCURRENT" ]; then
    ARRAY_SPEC="${ARRAY_SPEC}%${MAX_CONCURRENT}"
fi

sbatch -J "${RUN_ID}" \
    --time="${TIME_LIMIT}" \
    --cpus-per-task="${CPUS_PER_TASK}" \
    --array="${ARRAY_SPEC}" \
    --export=ALL,RUN_ID="${RUN_ID}",TASKS_STR="${TASKS}",K="${K}",NUM_WORKERS="${NUM_WORKERS}",\
NUM_INSTANCES="${NUM_INSTANCES}",MAXITER="${MAXITER}",OPTIMIZER="${OPTIMIZER}",SEED="${SEED}" \
    run_qscore_PCE_chunk_task.sh

echo ""
echo "Cuando terminen TODAS las tareas del array (squeue -u \$USER), fusiona los chunks y agrega:"
echo "  python merge_chunks_qscore_PCE.py --run-id ${RUN_ID}"
echo "  python aggregate_qscore_PCE.py --run-id ${RUN_ID}"
echo ""
echo "Si merge_chunks_qscore_PCE.py avisa de que algún n tiene chunks incompletos,"
echo "vuelve a ejecutarlo más tarde (no borra nada hasta que un n esté completo)."