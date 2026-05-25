#!/usr/bin/env bash
# Prepare official BANG Base inputs from a DiskANN build.
#
# This script does not use the teaching graph.bin path. It expects/creates
# DiskANN's *_disk.index, *_pq_compressed.bin, and *_pq_pivots.bin files, then
# runs BANG_Base/bang_preprocess.py to produce *_disk.bin and *_disk_metadata.bin.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
OFFICIAL_ROOT="${BANG_OFFICIAL_ROOT:-$ROOT/../BANG-Billion-Scale-ANN}"
BANG_BASE="$OFFICIAL_ROOT/BANG_Base"

BUILD_DISK_INDEX="${BUILD_DISK_INDEX:-}"
if [[ -z "$BUILD_DISK_INDEX" ]]; then
  BUILD_DISK_INDEX="$(command -v build_disk_index || true)"
fi

DATA_PATH=""
INDEX_PREFIX="$ROOT/results/official/sift1m_index"
DATA_TYPE="float"
DIST_FN="l2"
DIM="128"
DTYPE_ID="2"
R="64"
L_BUILD="200"
B_GB="1"
M="48"
SKIP_BUILD=0

usage() {
  cat <<EOF
Usage: $0 --data DATA.bin --prefix INDEX_PREFIX [options]

Required:
  --data PATH             DiskANN .bin base vectors: [N:int32][D:int32][raw vectors]

Options:
  --prefix PATH           Index prefix (default: $INDEX_PREFIX)
  --data-type TYPE        float|uint8|int8 (default: float)
  --dist-fn NAME          l2|mips (default: l2)
  -R DEGREE               DiskANN graph degree / BANG MAX_R (default: 64)
  -L BUILD_L              DiskANN build L (default: 200)
  -B GB                   DiskANN PQ memory budget in GiB (default: 1)
  -M MEM                  DiskANN RAM budget / build memory param (default: 48)
  --dim D                 Vector dimension if metadata is unavailable (default: 128)
  --skip-build            Only run BANG preprocess; assumes *_disk.index already exists

Environment:
  BUILD_DISK_INDEX=/path/to/DiskANN/build/apps/build_disk_index
  BANG_OFFICIAL_ROOT=/path/to/BANG-Billion-Scale-ANN
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --data) DATA_PATH="$2"; shift 2 ;;
    --prefix) INDEX_PREFIX="$2"; shift 2 ;;
    --data-type) DATA_TYPE="$2"; shift 2 ;;
    --dist-fn) DIST_FN="$2"; shift 2 ;;
    --dim) DIM="$2"; shift 2 ;;
    -R) R="$2"; shift 2 ;;
    -L) L_BUILD="$2"; shift 2 ;;
    -B) B_GB="$2"; shift 2 ;;
    -M) M="$2"; shift 2 ;;
    --skip-build) SKIP_BUILD=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage; exit 1 ;;
  esac
done

case "$DATA_TYPE" in
  float) DTYPE_ID="2" ;;
  uint8) DTYPE_ID="1" ;;
  int8) DTYPE_ID="0" ;;
  *) echo "Unsupported --data-type $DATA_TYPE; use float|uint8|int8" >&2; exit 1 ;;
esac

if [[ ! -d "$BANG_BASE" ]]; then
  echo "Official BANG_Base not found: $BANG_BASE" >&2
  exit 1
fi

if [[ -z "$DATA_PATH" ]]; then
  echo "--data is required" >&2
  usage
  exit 1
fi

mkdir -p "$(dirname "$INDEX_PREFIX")"

if [[ "$SKIP_BUILD" -eq 0 ]]; then
  if [[ -z "$BUILD_DISK_INDEX" || ! -x "$BUILD_DISK_INDEX" ]]; then
    echo "build_disk_index not found/executable." >&2
    echo "Set BUILD_DISK_INDEX=/path/to/DiskANN/build/apps/build_disk_index" >&2
    exit 1
  fi

  "$BUILD_DISK_INDEX" \
    --data_type "$DATA_TYPE" \
    --dist_fn "$DIST_FN" \
    --data_path "$DATA_PATH" \
    --index_path_prefix "$INDEX_PREFIX" \
    -R "$R" -L "$L_BUILD" -B "$B_GB" -M "$M"
fi

DISK_INDEX="${INDEX_PREFIX}_disk.index"
DISK_BIN="${INDEX_PREFIX}_disk.bin"

if [[ ! -f "$DISK_INDEX" ]]; then
  echo "DiskANN disk index not found: $DISK_INDEX" >&2
  exit 1
fi

python3 "$BANG_BASE/bang_preprocess.py" "$DISK_INDEX" "$DISK_BIN" "$DIM" "$DTYPE_ID" "$R"

echo
echo "Official BANG input prefix is ready:"
echo "  $INDEX_PREFIX"
echo
echo "Required files:"
for suffix in _disk.bin _disk_metadata.bin _pq_compressed.bin _pq_pivots.bin; do
  if [[ -f "${INDEX_PREFIX}${suffix}" ]]; then
    ls -lh "${INDEX_PREFIX}${suffix}"
  else
    echo "  MISSING: ${INDEX_PREFIX}${suffix}"
  fi
done
