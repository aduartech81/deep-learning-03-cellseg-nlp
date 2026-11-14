#!/usr/bin/env bash
# Ejecuta capítulos (nb_src/*.py -> notebooks/*.ipynb ejecutados). Uso: scripts/run_chapters.sh [prefijos...]
# Perfil por variable DL03_PROFILE (full | smoke). Las salidas quedan en notebooks/ y los logs en logs/.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p notebooks logs
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}" TF_CPP_MIN_LOG_LEVEL=3
chapters=("$@")
[ ${#chapters[@]} -eq 0 ] && chapters=(00 01 02 03 06 07 08 04 05 09 10)
for ch in "${chapters[@]}"; do
  for src in nb_src/${ch}_*.py; do
    nb="notebooks/$(basename "${src%.py}").ipynb"
    jupytext --to ipynb --quiet -o "$nb" "$src"
    echo "=== $(date +%H:%M) ejecutando $nb (perfil ${DL03_PROFILE:-full})"
    jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 \
      --ExecutePreprocessor.kernel_name=python3 "$nb" > "logs/${ch}.log" 2>&1 || { sed -E "s/\x1b\[[0-9;]*m//g" "logs/${ch}.log" | grep -E "Error|error|Traceback|^---->|^\s+[0-9]+ " | tail -25; exit 1; }
  done
done
