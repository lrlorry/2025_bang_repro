#!/usr/bin/env bash
# Convert TexMex fvecs/bvecs/ivecs files to DiskANN/BANG .bin layout.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
BUILD_DIR="${ROOT}/build"
TOOL="${BUILD_DIR}/fvecs_to_bin"

if [[ $# -lt 3 ]]; then
  echo "Usage: $0 input.{fvecs,bvecs,ivecs} output.bin [float|uint8|int]" >&2
  exit 1
fi

if [[ ! -x "$TOOL" ]]; then
  echo "$TOOL not found. Build repo tools first:" >&2
  echo "  cd $ROOT && mkdir -p build && cd build && cmake .. && make fvecs_to_bin" >&2
  exit 1
fi

exec "$TOOL" "$1" "$2" "$3"
