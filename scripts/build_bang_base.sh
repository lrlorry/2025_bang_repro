#!/usr/bin/env bash
# 编译 BANG_Base
# 用法：bash scripts/build_bang_base.sh
# 说明：
#   从 ../BANG-Billion-Scale-ANN/BANG_Base 编译，使用 cmake。
#   编译完成后，bang_search 二进制在 BANG_Base/build/ 目录下。

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BANG_SRC="$REPO_ROOT/../BANG-Billion-Scale-ANN/BANG_Base"
BUILD_DIR="$BANG_SRC/build"
LOG_FILE="$REPO_ROOT/results/build_bang_base.log"

mkdir -p "$REPO_ROOT/results"

echo "====== 编译 BANG_Base ======"
echo "源码路径：$BANG_SRC"
echo "构建目录：$BUILD_DIR"
echo "日志：$LOG_FILE"
echo ""

# --- 前置检查 ---
if ! command -v nvcc &>/dev/null; then
    echo "[ERROR] nvcc 未找到，无法编译 BANG_Base。"
    echo "原因：nvcc 未在 PATH 中。请安装 CUDA Toolkit >= 11.8。" | tee -a "$LOG_FILE"
    exit 1
fi

if ! command -v cmake &>/dev/null; then
    echo "[ERROR] cmake 未找到。"
    echo "原因：cmake 未在 PATH 中。请安装 cmake >= 3.16。" | tee -a "$LOG_FILE"
    exit 1
fi

if [ ! -f "$BANG_SRC/CMakeLists.txt" ]; then
    echo "[ERROR] 未找到 CMakeLists.txt：$BANG_SRC/CMakeLists.txt"
    echo "原因：BANG_Base 源码目录不存在或路径错误。" | tee -a "$LOG_FILE"
    exit 1
fi

# --- 检查是否有 GPU（警告但不阻止编译）---
if ! command -v nvidia-smi &>/dev/null; then
    echo "[WARN] nvidia-smi 未找到，可能没有 GPU。编译可能成功，但运行会失败。"
fi

# --- cmake configure ---
echo "--- cmake configure ---"
mkdir -p "$BUILD_DIR"
(cd "$BUILD_DIR" && cmake .. -DCMAKE_BUILD_TYPE=Release 2>&1 | tee -a "$LOG_FILE")

cmake_exit=${PIPESTATUS[0]}
if [ $cmake_exit -ne 0 ]; then
    echo ""
    echo "[ERROR] cmake configure 失败（exit code $cmake_exit）。"
    echo "常见原因："
    echo "  1. Boost 未安装：sudo apt install libboost-all-dev"
    echo "  2. CUDA arch 不匹配：编辑 CMakeLists.txt 中的 -arch=sm_XX"
    echo "  3. gcc 版本过低：需要 >= 11.0"
    echo "查看详细日志：$LOG_FILE"
    exit 1
fi

# --- make ---
echo ""
echo "--- make ---"
NPROC=$(nproc 2>/dev/null || echo 4)
(cd "$BUILD_DIR" && make -j"$NPROC" 2>&1 | tee -a "$LOG_FILE")

make_exit=${PIPESTATUS[0]}
if [ $make_exit -ne 0 ]; then
    echo ""
    echo "[ERROR] make 失败（exit code $make_exit）。"
    echo "常见原因："
    echo "  1. MAX_R 与 DiskANN 构图的 -R 不一致（默认 MAX_R=64）"
    echo "  2. CUDA compute capability 不支持：修改 CMakeLists.txt 中 -arch 参数"
    echo "  3. CUB 头文件缺失：通常随 CUDA Toolkit 安装"
    echo "查看详细日志：$LOG_FILE"
    exit 1
fi

# --- 验证输出 ---
BANG_BIN="$BUILD_DIR/bang_search"
if [ -f "$BANG_BIN" ]; then
    echo ""
    echo "[OK] 编译成功：$BANG_BIN"
    echo "下一步：bash scripts/run_sift10k_smoke.sh"
else
    echo "[ERROR] 编译完成但未找到 bang_search 二进制：$BANG_BIN"
    echo "请检查 CMakeLists.txt 中的目标名称。"
    exit 1
fi
