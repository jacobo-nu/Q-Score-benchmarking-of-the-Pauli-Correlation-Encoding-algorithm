#!/bin/bash
# lanzar_test_nodos.sh
# =====================
# Lanza el mismo test (n=10, seed=42, num_workers=20) en dos nodos
# concretos con --exclusive, para aislar si la diferencia de beta
# (0.8616 vs 0.6680) viene del nodo físico o de contención con otros
# jobs. Crea a mano las carpetas logs/<RUN_ID>/ y Resultados/PCE/<RUN_ID>/
# porque al saltarnos submit_qscore_PCE_parallel.sh (que normalmente las
# crea) sbatch falla si no existen (visto hoy: 8355031 FAILED,
# ExitCode 1:0, por esta misma causa).
#
# Ejecutar desde Q_Score_PCE_Parallel/FT3/

set -e  # cortar si algo falla, para no lanzar el segundo job con un RUN_ID mal formado

TIMESTAMP=$(date +%H%M%S)
RUN_ID_A="test_nodo_ilk244_${TIMESTAMP}"
RUN_ID_B="test_nodo_ilk194_${TIMESTAMP}"

mkdir -p "logs/${RUN_ID_A}"
mkdir -p "logs/${RUN_ID_B}"
mkdir -p "Resultados/PCE/${RUN_ID_A}/chunks"
mkdir -p "Resultados/PCE/${RUN_ID_B}/chunks"

echo "RUN_ID_A (ilk-244): ${RUN_ID_A}"
echo "RUN_ID_B (ilk-194): ${RUN_ID_B}"
echo ""

JOBID_A=$(sbatch --parsable -J "${RUN_ID_A}" \
    --nodelist=ilk-244 --exclusive \
    --partition=short --qos=short --time=00:30:00 \
    --array=0-0 \
    --export=ALL,RUN_ID="${RUN_ID_A}",TASKS_STR="10:0:1",K=2,NUM_WORKERS=20,\
NUM_INSTANCES=100,MAXITER=25,OPTIMIZER=DIFFERENTIALEVOLUTION,SEED=42 \
    run_qscore_PCE_chunk_task_cpu.sh)

JOBID_B=$(sbatch --parsable -J "${RUN_ID_B}" \
    --nodelist=ilk-194 --exclusive \
    --partition=short --qos=short --time=00:30:00 \
    --array=0-0 \
    --export=ALL,RUN_ID="${RUN_ID_B}",TASKS_STR="10:0:1",K=2,NUM_WORKERS=20,\
NUM_INSTANCES=100,MAXITER=25,OPTIMIZER=DIFFERENTIALEVOLUTION,SEED=42 \
    run_qscore_PCE_chunk_task_cpu.sh)

echo "Job A (ilk-244) enviado: ${JOBID_A}"
echo "Job B (ilk-194) enviado: ${JOBID_B}"
echo ""
echo "Seguimiento:"
echo "  squeue -u \$USER"
echo "  tail -f logs/${RUN_ID_A}/qscore_pce_cpu_${JOBID_A}_0.out"
echo "  tail -f logs/${RUN_ID_B}/qscore_pce_cpu_${JOBID_B}_0.out"
echo ""
echo "Cuando terminen, comparar:"
echo "  grep beta_parcial logs/${RUN_ID_A}/qscore_pce_cpu_${JOBID_A}_0.out"
echo "  grep beta_parcial logs/${RUN_ID_B}/qscore_pce_cpu_${JOBID_B}_0.out"