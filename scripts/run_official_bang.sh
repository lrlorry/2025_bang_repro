#!/usr/bin/env bash
# Run official BANG Base bang_search against a prepared DiskANN/BANG prefix.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
OFFICIAL_ROOT="${BANG_OFFICIAL_ROOT:-$ROOT/../BANG-Billion-Scale-ANN}"
BANG_BASE="$OFFICIAL_ROOT/BANG_Base"
BUILD_DIR="${BANG_OFFICIAL_BUILD:-$BANG_BASE/build}"
BIN="${BANG_SEARCH_BIN:-$BUILD_DIR/bang_search}"

PREFIX=""
QUERY=""
GT=""
NUMQ="10000"
K="10"
DATA_TYPE="float"
DIST_FN="l2"
MODE="auto"

usage() {
  cat <<EOF
Usage: $0 --prefix INDEX_PREFIX --query QUERY.bin --gt GT.bin [options]

Options:
  --numq N                Number of queries (default: 10000)
  --k K                   recall@K / top-K (default: 10)
  --data-type TYPE        float|uint8|int8 (default: float)
  --dist-fn NAME          l2|mips (default: l2)
  --interactive           Let official bang_search prompt for worklist length

Expected files under prefix:
  PREFIX_pq_pivots.bin
  PREFIX_pq_compressed.bin
  PREFIX_disk.bin
  PREFIX_disk_metadata.bin
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --prefix) PREFIX="$2"; shift 2 ;;
    --query) QUERY="$2"; shift 2 ;;
    --gt) GT="$2"; shift 2 ;;
    --numq) NUMQ="$2"; shift 2 ;;
    --k) K="$2"; shift 2 ;;
    --data-type) DATA_TYPE="$2"; shift 2 ;;
    --dist-fn) DIST_FN="$2"; shift 2 ;;
    --interactive) MODE="interactive"; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage; exit 1 ;;
  esac
done

if [[ -z "$PREFIX" || -z "$QUERY" || -z "$GT" ]]; then
  usage
  exit 1
fi

if [[ ! -x "$BIN" ]]; then
  echo "Official bang_search not found: $BIN" >&2
  echo "Run: bash scripts/build_official_bang.sh" >&2
  exit 1
fi

for f in "${PREFIX}_pq_pivots.bin" "${PREFIX}_pq_compressed.bin" "${PREFIX}_disk.bin" "${PREFIX}_disk_metadata.bin" "$QUERY" "$GT"; do
  if [[ ! -f "$f" ]]; then
    echo "Missing required file: $f" >&2
    exit 1
  fi
done

if [[ "$MODE" == "interactive" ]]; then
  exec "$BIN" "$PREFIX" "$QUERY" "$GT" "$NUMQ" "$K" "$DATA_TYPE" "$DIST_FN"
else
  exec "$BIN" "$PREFIX" "$QUERY" "$GT" "$NUMQ" "$K" "$DATA_TYPE" "$DIST_FN" auto
fi
