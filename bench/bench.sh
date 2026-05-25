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

# ── SIFT1M data paths (adjust if different) ──────────────────────────────────
SIFT_BASE="${ROOT}/sift1m_data/sift_base.fvecs"
SIFT_QUERY="${ROOT}/sift1m_data/sift_query.fvecs"
SIFT_GT="${ROOT}/sift1m_data/sift_groundtruth.ivecs"
# Fall back to sibling cagra_repro data directory
if [ ! -f "$SIFT_BASE" ]; then
  SIFT_BASE="${ROOT}/../2024_cagra_repro/data/sift_base.fvecs"
  SIFT_QUERY="${ROOT}/../2024_cagra_repro/data/sift_query.fvecs"
  SIFT_GT="${ROOT}/../2024_cagra_repro/data/sift_groundtruth.ivecs"
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

# ── SIFT1M L-sweep benchmark (vs CAGRA comparison) ──────────────────────────
echo ""
echo "========================================"
echo " BANG SIFT1M L-sweep (vs CAGRA)"
echo "========================================"

SWEEP_CSV="$RESULTS/bang_sweep.csv"
echo "L,recall,qps,search_ms" > "$SWEEP_CSV"

HAS_SIFT=0
if [ -f "$SIFT_BASE" ] && [ -f "$SIFT_QUERY" ] && [ -f "$SIFT_GT" ]; then
  HAS_SIFT=1
  echo "SIFT1M found: $SIFT_BASE"
else
  echo "SIFT1M not found — skipping L-sweep"
  echo "  To enable, place sift_base.fvecs / sift_query.fvecs / sift_groundtruth.ivecs"
  echo "  in $ROOT/sift1m_data/  or  ${ROOT}/../2024_cagra_repro/data/"
fi

if [ "$HAS_SIFT" -eq 1 ]; then
  for L in 16 32 48 64; do
    BIN="$BUILD/bang_bench_L${L}"
    if [ -f "$BIN" ]; then
      echo ""
      echo "--- bang_bench_L${L} ---"
      "$BIN" --base "$SIFT_BASE" --query "$SIFT_QUERY" --gt "$SIFT_GT" \
             >> "$SWEEP_CSV"
    else
      echo "Warning: $BIN not found, skipping L=$L"
    fi
  done
  echo ""
  echo "bang_sweep.csv:"
  cat "$SWEEP_CSV"
fi

echo ""
echo "Raw results saved to $RESULTS/"
echo "Run: python3 $SCRIPT_DIR/plot.py  to generate figures"
echo "Run: python3 $SCRIPT_DIR/plot.py --cagra-compare  to add CAGRA comparison"
