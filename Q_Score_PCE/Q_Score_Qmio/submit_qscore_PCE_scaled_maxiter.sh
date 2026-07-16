#!/bin/bash
# submit_qscore_PCE_scaled_maxiter.sh — QMIO
# ============================
# Wrapper de envío para run_qscore_PCE_array_scaled_maxiter.sh. Mismo
# patrón que submit_qscore_PCE_array.sh: genera un RUN_ID, crea las
# subcarpetas necesarias, y lanza el array.
#
# Uso:
#   ./submit_qscore_PCE_scaled_maxiter.sh
#   MAXITER_INTERCEPT=20 MAXITER_SLOPE=0.2 ./submit_qscore_PCE_scaled_maxiter.sh

set -e

RUN_ID=$(date +%Y%m%d-%H%M%S)
export RUN_ID

mkdir -p "logs/${RUN_ID}"
mkdir -p "Resultados/PCE/${RUN_ID}"
mkdir -p "Imagenes/PCE/${RUN_ID}"

sbatch -J "${RUN_ID}" run_qscore_PCE_array_scaled_maxiter.sh

echo ""
echo "Cuando termine, agrega los resultados con:"
echo "  python aggregate_qscore_PCE.py --run-id ${RUN_ID}"
echo ""
echo "NOTA: el agregador avisará con '⚠ Aviso: los tamaños [...] usaron"
echo "parámetros distintos' porque maxiter varía a propósito por n en"
echo "esta tanda — es el comportamiento esperado, no un error. El valor"
echo "de maxiter usado en CADA n concreto está en la columna 'pce_maxiter'"
echo "del CSV exportado, fila por fila."