#!/usr/bin/env bash
# Compute DiskANN/BANG truthset format: [N, K, ids, dists].

set -euo pipefail

COMPUTE_GROUNDTRUTH="${COMPUTE_GROUNDTRUTH:-}"
if [[ -z "$COMPUTE_GROUNDTRUTH" ]]; then
  COMPUTE_GROUNDTRUTH="$(command -v compute_groundtruth || true)"
fi

BASE=""
QUERY=""
OUT=""
DATA_TYPE="float"
DIST_FN="l2"
K="10"

usage() {
  cat <<EOF
Usage: $0 --base BASE.bin --query QUERY.bin --out GT.bin [options]

Options:
  --data-type TYPE        float|uint8|int8 (default: float)
  --dist-fn NAME          l2|mips (default: l2)
  --k K                   Groundtruth K (default: 10)

Environment:
  COMPUTE_GROUNDTRUTH=/path/to/DiskANN/build/apps/compute_groundtruth
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --base) BASE="$2"; shift 2 ;;
    --query) QUERY="$2"; shift 2 ;;
    --out) OUT="$2"; shift 2 ;;
    --data-type) DATA_TYPE="$2"; shift 2 ;;
    --dist-fn) DIST_FN="$2"; shift 2 ;;
    --k) K="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage; exit 1 ;;
  esac
done

if [[ -z "$BASE" || -z "$QUERY" || -z "$OUT" ]]; then
  usage
  exit 1
fi

if [[ -z "$COMPUTE_GROUNDTRUTH" || ! -x "$COMPUTE_GROUNDTRUTH" ]]; then
  echo "compute_groundtruth not found/executable." >&2
  echo "Set COMPUTE_GROUNDTRUTH=/path/to/DiskANN/build/apps/compute_groundtruth" >&2
  exit 1
fi

"$COMPUTE_GROUNDTRUTH" \
  --data_type "$DATA_TYPE" \
  --dist_fn "$DIST_FN" \
  --base_file "$BASE" \
  --query_file "$QUERY" \
  --K "$K" \
  --gt_file "$OUT"

echo "Groundtruth written: $OUT"
