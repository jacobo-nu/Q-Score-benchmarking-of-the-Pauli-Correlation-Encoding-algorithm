#!/bin/bash
# submit_qscore_PCE_array.sh (FT3 — CPU hasta 'long', GPU solo para los n más grandes)
# ============================
# Clasifica cada n de la lista maestra según su nº de qubits:
#
#   qubits < QUBIT_THRESHOLD_CPU_SHORT   -> CPU, partición 'short'  (6h)
#   qubits < QUBIT_THRESHOLD_CPU_MEDIUM  -> CPU, partición 'medium' (3-00:00:00)
#   qubits < QUBIT_THRESHOLD             -> CPU, partición 'long'   (5-00:00:00)
#   qubits >= QUBIT_THRESHOLD            -> GPU, partición 'long'   (5-00:00:00)
#
# CPU escala por los tres tramos de tiempo antes de pasar a GPU. La GPU se
# reserva solo para los n más grandes (qubits >= QUBIT_THRESHOLD, por
# defecto 17 — subido desde 14): a esos tamaños (n~300-350, 15-16 qubits)
# la ventaja de la GPU frente a CPU todavía no es tan grande como para
# compensar la espera en cola de un recurso mucho más disputado — cuantas
# menos tareas pidan GPU, antes entran todas en cola.
#
# !! NINGÚN UMBRAL NI VALOR DE --time ESTÁ CALIBRADO CON PRECISIÓN !! Son
# puntos de partida razonados a partir de lo que ya has observado. Ajusta
# con sacctmgr/tus propios duration_seconds.
#
# Uso:
#   ./submit_qscore_PCE_array.sh
#   QUBIT_THRESHOLD=18 ./submit_qscore_PCE_array.sh   # subir aún más el listón de GPU

set -e

RUN_ID=$(date +%Y%m%d-%H%M%S)

mkdir -p "logs/${RUN_ID}"
mkdir -p "Resultados/PCE/${RUN_ID}"
mkdir -p "Imagenes/PCE/${RUN_ID}"

# =============================
# Lista maestra de tamaños de nodo — ÚNICA fuente de verdad.
# =============================
N_LIST_MASTER=(300 325 350 375 400 425 450 475 500)

# =============================
# Umbrales de qubits
# =============================
QUBIT_THRESHOLD_CPU_SHORT="${QUBIT_THRESHOLD_CPU_SHORT:-5}"    # CPU: short vs medium
QUBIT_THRESHOLD_CPU_MEDIUM="${QUBIT_THRESHOLD_CPU_MEDIUM:-10}" # CPU: medium vs long
QUBIT_THRESHOLD="${QUBIT_THRESHOLD:-16}"                        # CPU vs GPU (antes 14 — GPU ahora solo para lo último)
K="${K:-3}"

# =============================
# Clasificar cada n en uno de los 4 grupos
# =============================
module load cesga/2022 gcc/system qiskit/1.2.4-aer-gpu-cu11

declare -A NLIST_BY_BUCKET
for N in "${N_LIST_MASTER[@]}"; do
    QUBITS=$(python -c "from qscore_PCE import num_qubits; print(num_qubits(${N}, ${K}))")
    if [ "$QUBITS" -lt "$QUBIT_THRESHOLD" ]; then
        if [ "$QUBITS" -lt "$QUBIT_THRESHOLD_CPU_SHORT" ]; then
            BUCKET="cpu_short"
        elif [ "$QUBITS" -lt "$QUBIT_THRESHOLD_CPU_MEDIUM" ]; then
            BUCKET="cpu_medium"
        else
            BUCKET="cpu_long"
        fi
    else
        BUCKET="gpu_long"
    fi
    NLIST_BY_BUCKET[$BUCKET]="${NLIST_BY_BUCKET[$BUCKET]} $N"
done

echo "=============================="
echo "RUN_ID: ${RUN_ID}"
echo "K: ${K}"
echo "QUBIT_THRESHOLD_CPU_SHORT: ${QUBIT_THRESHOLD_CPU_SHORT}  QUBIT_THRESHOLD_CPU_MEDIUM: ${QUBIT_THRESHOLD_CPU_MEDIUM}  QUBIT_THRESHOLD (CPU/GPU): ${QUBIT_THRESHOLD}"
echo "cpu_short:  ${NLIST_BY_BUCKET[cpu_short]:-<ninguna>}"
echo "cpu_medium: ${NLIST_BY_BUCKET[cpu_medium]:-<ninguna>}"
echo "cpu_long:   ${NLIST_BY_BUCKET[cpu_long]:-<ninguna>}"
echo "gpu_long:   ${NLIST_BY_BUCKET[gpu_long]:-<ninguna>}"
echo "=============================="

# =============================
# Tabla bucket -> (script, partición/qos, --time)
# =============================
declare -A BUCKET_SCRIPT=([cpu_short]="run_qscore_PCE_array_cpu.sh" [cpu_medium]="run_qscore_PCE_array_cpu.sh" \
                          [cpu_long]="run_qscore_PCE_array_cpu.sh"   [gpu_long]="run_qscore_PCE_array.sh")
declare -A BUCKET_PARTITION=([cpu_short]="short" [cpu_medium]="medium" [cpu_long]="long" [gpu_long]="long")
declare -A BUCKET_TIME=([cpu_short]="6:00:00" [cpu_medium]="3-00:00:00" [cpu_long]="5-00:00:00" [gpu_long]="5-00:00:00")

# =============================
# Lanzar un array por cada bucket no vacío
# =============================
for BUCKET in cpu_short cpu_medium cpu_long gpu_long; do
    NLIST="${NLIST_BY_BUCKET[$BUCKET]:-}"
    if [ -n "$NLIST" ]; then
        # shellcheck disable=SC2206
        NARR=($NLIST)
        PARTITION="${BUCKET_PARTITION[$BUCKET]}"
        sbatch -J "${RUN_ID}" \
            --partition="${PARTITION}" \
            --qos="${PARTITION}" \
            --time="${BUCKET_TIME[$BUCKET]}" \
            --array="0-$((${#NARR[@]}-1))" \
            --export=ALL,RUN_ID="${RUN_ID}",N_LIST_STR="${NLIST}",K="${K}" \
            "${BUCKET_SCRIPT[$BUCKET]}"
    fi
done

echo ""
echo "Cuando terminen TODOS los arrays (comprueba con squeue -u \$USER), agrega los resultados con:"
echo "  python aggregate_qscore_PCE.py --run-id ${RUN_ID}"
