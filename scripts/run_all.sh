#!/usr/bin/env bash
set -euo pipefail

mkdir -p bin


echo "Taller CUDA - ejecucion completa"


echo ""
echo "== Verificacion de entorno =="
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi || true
else
  echo "nvidia-smi no encontrado. Si estas en Colab, activa GPU."
fi

if command -v nvcc >/dev/null 2>&1; then
  nvcc --version
else
  echo ""
  echo "nvcc no encontrado."
  echo "Solo se ejecutara la version CPU si gcc esta disponible."
fi

echo ""
echo "== Ejercicio 1 CPU C =="
if command -v gcc >/dev/null 2>&1; then
  gcc -O2 src/sum_array.c -o bin/sum_array_cpu
  ./bin/sum_array_cpu
else
  echo "gcc no encontrado."
fi

if ! command -v nvcc >/dev/null 2>&1; then
  echo ""
  echo "Fin: no hay nvcc para compilar CUDA."
  exit 0
fi

ARCH="${ARCH:-$(bash scripts/detect_arch.sh)}"

echo ""
echo "Arquitectura CUDA usada: ${ARCH}"

CUDA_SOURCES=(
  src/sum_array.cu
  src/ej02_hello_gpu.cu
  src/ej03_device_info.cu
  src/ej04_vector_add.cu
  src/ej05_matrix_add.cu
  src/ej06_matrix_mul.cu
  src/ej07_shared_reduction.cu
  src/ej08_large_vector_timing.cu
  src/ej09_unified_memory.cu
)

for src in "${CUDA_SOURCES[@]}"; do
  if [[ ! -f "$src" ]]; then
    continue
  fi

  name="$(basename "$src" .cu)"
  out="bin/${name}_cuda"

  echo ""
  echo "Compilando: ${src}"
  echo "Salida: ${out}"
 

  nvcc -O2 -arch="${ARCH}" "$src" -o "$out"

  echo ""
  echo "Ejecutando: ${out}"
  echo "--------------------------------------"
  "./${out}"
done

echo ""
echo "Todos los ejercicios finalizaron."
