#!/bin/bash
# BANG reproduction benchmark
# 输出 results/plain.txt 和 results/engineered.txt，供 plot.py 使用

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$SCRIPT_DIR/.."
BUILD="$ROOT/build"
RESULTS="$ROOT/results"
mkdir -p "$RESULTS"

if [ ! -f "$BUILD/bang_plain" ]; then
  echo "请先在 build/ 目录下编译: cd $ROOT && mkdir -p build && cd build && cmake .. && make -j\$(nproc)"
  exit 1
fi

echo "========================================"
echo " BANG Reproduction Benchmark"
echo "========================================"

run_and_time() {
  local bin="$1"
  local out="$2"
  echo ""
  echo "--- Running: $bin ---"
  local start=$(date +%s%3N)
  "$bin" 2>&1 | tee "$out"
  local end=$(date +%s%3N)
  local elapsed=$(( end - start ))
  echo "TOTAL_TIME_MS=$elapsed" >> "$out"
  echo "Total wall time: ${elapsed} ms"
}

run_and_time "$BUILD/bang_plain"      "$RESULTS/plain.txt"
run_and_time "$BUILD/bang_engineered" "$RESULTS/engineered.txt"

echo ""
echo "========================================"
echo " Results summary"
echo "========================================"
echo ""
echo "[plain]"
grep -E "Recall|build|search|TOTAL" "$RESULTS/plain.txt" || true
echo ""
echo "[engineered]"
grep -E "Recall|build|search|TOTAL" "$RESULTS/engineered.txt" || true

echo ""
echo "Raw results saved to $RESULTS/"
echo "Run: python3 $SCRIPT_DIR/plot.py  to generate figures"
