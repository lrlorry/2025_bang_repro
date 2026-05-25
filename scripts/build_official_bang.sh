#!/usr/bin/env bash
# Build the official BANG Base implementation from ../BANG-Billion-Scale-ANN.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
OFFICIAL_ROOT="${BANG_OFFICIAL_ROOT:-$ROOT/../BANG-Billion-Scale-ANN}"
BANG_BASE="$OFFICIAL_ROOT/BANG_Base"
BUILD_DIR="${BANG_OFFICIAL_BUILD:-$BANG_BASE/build}"

if [[ ! -d "$BANG_BASE" ]]; then
  echo "Official BANG_Base not found: $BANG_BASE" >&2
  echo "Set BANG_OFFICIAL_ROOT=/path/to/BANG-Billion-Scale-ANN" >&2
  exit 1
fi

if ! command -v cmake >/dev/null 2>&1; then
  echo "cmake not found" >&2
  exit 1
fi

if ! command -v nvcc >/dev/null 2>&1; then
  echo "nvcc not found; official BANG requires CUDA" >&2
  exit 1
fi

mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

cmake .. -DCMAKE_BUILD_TYPE=Release
make -j"$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)"

echo
echo "Official BANG built:"
echo "  $BUILD_DIR/bang_search"
echo "  $BUILD_DIR/libbang.so"
