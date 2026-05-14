#!/usr/bin/env bash
set -euo pipefail

if command -v nvidia-smi >/dev/null 2>&1; then
  CAP="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -n 1 | tr -d '.[:space:]' || true)"

  if [[ -n "${CAP}" ]]; then
    echo "sm_${CAP}"
    exit 0
  fi
fi

# Valor por defecto para Google Colab T4.
echo "sm_75"
