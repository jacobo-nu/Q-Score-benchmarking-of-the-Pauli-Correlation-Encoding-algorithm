#!/bin/bash
# submit_qscore_PCE_array.sh (QMIO — versión con reparto por tiempo)
# ============================
# Divide la lista maestra de tamaños de nodo en TRES sublistas según el
# número de qubits que necesita cada n, y lanza hasta tres job arrays
# independientes, cada uno con el --time correcto para caer en el QOS
# automático que le corresponde en QMIO (ilk_short/ilk_medium/ilk_long,
# asignados por SLURM a partir de --time, no se pide --qos a mano):
#
#   qubits < QUBIT_THRESHOLD_SHORT   -> --time=06:00:00   (-> ilk_short)
#   qubits < QUBIT_THRESHOLD_MEDIUM  -> --time=3-00:00:00 (-> ilk_medium)
#   qubits >= QUBIT_THRESHOLD_MEDIUM -> --time=7-00:00:00 (-> ilk_long)
#
# Motivación: antes, TODO el array pedía el mismo --time (normalmente el
# más largo que pudiera necesitar el n más grande), desperdiciando cupo de
# la cola larga en tareas de n pequeño que en realidad caben de sobra en
# ilk_short — cada QOS tiene sus propios límites de MaxJobsPU/MaxSubmitPU,
# así que repartir bien aprovecha mejor los tres cupos en vez de saturar
# uno solo.
#
# !! LOS UMBRALES DE QUBITS NO ESTÁN CALIBRADOS !! Son un punto de partida
# razonable. Ajústalos con sacctmgr/tus propios duration_seconds ya
# guardados en tus .h5 — ver instrucciones al final de este fichero.
#
# Uso:
#   ./submit_qscore_PCE_array.sh
#   QUBIT_THRESHOLD_SHORT=8 QUBIT_THRESHOLD_MEDIUM=14 ./submit_qscore_PCE_array.sh

set -e

RUN_ID=$(date +%Y%m%d-%H%M%S)

mkdir -p "logs/${RUN_ID}"
mkdir -p "Resultados/PCE/${RUN_ID}"
mkdir -p "Imagenes/PCE/${RUN_ID}"

# =============================
# Lista maestra de tamaños de nodo — ÚNICA fuente de verdad.
# =============================
N_LIST_MASTER=(150)

# =============================
# Umbrales de qubits para elegir el --time (y por tanto el QOS automático)
# =============================
QUBIT_THRESHOLD_SHORT="${QUBIT_THRESHOLD_SHORT:-6}"
QUBIT_THRESHOLD_MEDIUM="${QUBIT_THRESHOLD_MEDIUM:-18}"
K="${K:-2}"

# =============================
# Clasificar cada n en short/medium/long según su nº de qubits real
# =============================
module load qmio/hpc gcc/12.3.0 qiskit/2.2.3-python-3.11.9 qmio-tools/0.2.1-python-3.11.9

declare -A NLIST_BY_BUCKET
for N in "${N_LIST_MASTER[@]}"; do
    QUBITS=$(python -c "from qscore_PCE import num_qubits; print(num_qubits(${N}, ${K}))")
    if [ "$QUBITS" -lt "$QUBIT_THRESHOLD_SHORT" ]; then
        BUCKET="short"
    elif [ "$QUBITS" -lt "$QUBIT_THRESHOLD_MEDIUM" ]; then
        BUCKET="medium"
    else
        BUCKET="long"
    fi
    NLIST_BY_BUCKET[$BUCKET]="${NLIST_BY_BUCKET[$BUCKET]} $N"
done

echo "=============================="
echo "RUN_ID: ${RUN_ID}"
echo "K: ${K}"
echo "QUBIT_THRESHOLD_SHORT: ${QUBIT_THRESHOLD_SHORT}  QUBIT_THRESHOLD_MEDIUM: ${QUBIT_THRESHOLD_MEDIUM}"
echo "short  (--time=06:00:00):   ${NLIST_BY_BUCKET[short]:-<ninguna>}"
echo "medium (--time=3-00:00:00): ${NLIST_BY_BUCKET[medium]:-<ninguna>}"
echo "long   (--time=7-00:00:00): ${NLIST_BY_BUCKET[long]:-<ninguna>}"
echo "=============================="

# =============================
# Lanzar un array por cada bucket no vacío, con --time explícito en la
# línea de sbatch (tiene prioridad sobre el #SBATCH --time del fichero).
# =============================
declare -A BUCKET_TIME=([short]="06:00:00" [medium]="3-00:00:00" [long]="6-00:00:00")

for BUCKET in short medium long; do
    NLIST="${NLIST_BY_BUCKET[$BUCKET]:-}"
    if [ -n "$NLIST" ]; then
        # shellcheck disable=SC2206
        NARR=($NLIST)
        sbatch -J "${RUN_ID}" \
            --time="${BUCKET_TIME[$BUCKET]}" \
            --array="0-$((${#NARR[@]}-1))" \
            --export=ALL,RUN_ID="${RUN_ID}",N_LIST_STR="${NLIST}",K="${K}" \
            run_qscore_PCE_array.sh
    fi
done

echo ""
echo "Cuando terminen TODOS los arrays (comprueba con squeue -u \$USER), agrega los resultados con:"
echo "  python aggregate_qscore_PCE.py --run-id ${RUN_ID}"