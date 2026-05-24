#!/usr/bin/env bash
# BANG 复现一键入口
# 用法：
#   bash run.sh smoke          # SIFT10K smoke test（需 GPU）
#   bash run.sh build          # 编译 BANG_Base
#   bash run.sh check          # 检查环境
#   bash run.sh estimate       # 估算各规模资源
#   bash run.sh plot           # 从 results/ 生成图表

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPTS="$REPO_ROOT/scripts"

cmd="${1:-help}"

case "$cmd" in
  check)
    bash "$SCRIPTS/check_env.sh"
    ;;
  build)
    bash "$SCRIPTS/build_bang_base.sh"
    ;;
  smoke)
    bash "$SCRIPTS/run_sift10k_smoke.sh"
    ;;
  estimate)
    python3 "$SCRIPTS/estimate_resources.py"
    ;;
  plot)
    python3 "$SCRIPTS/plot_results.py \
      --results_dir "$REPO_ROOT/results" \
      --figures_dir "$REPO_ROOT/figures"
    ;;
  help|*)
    echo "用法: bash run.sh <command>"
    echo ""
    echo "  check     检查环境（nvcc、GPU、cmake、DiskANN）"
    echo "  build     编译 BANG_Base"
    echo "  smoke     运行 SIFT10K smoke test（需 GPU）"
    echo "  estimate  估算各规模资源需求"
    echo "  plot      从 results/*.csv 生成图表"
    ;;
esac
