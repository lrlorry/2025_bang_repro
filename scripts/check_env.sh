#!/usr/bin/env bash
# 检查 BANG 复现所需的环境依赖
# 用法：bash scripts/check_env.sh

set -uo pipefail

PASS=0
FAIL=0
WARN=0

ok()   { echo "[OK]   $1"; ((PASS++)); }
fail() { echo "[FAIL] $1"; ((FAIL++)); }
warn() { echo "[WARN] $1"; ((WARN++)); }

echo "====== BANG 环境检查 ======"
echo ""

# --- nvcc ---
if command -v nvcc &>/dev/null; then
    NVCC_VER=$(nvcc --version | grep "release" | awk '{print $6}' | tr -d ',')
    ok "nvcc 已找到：$NVCC_VER"
else
    fail "nvcc 未找到。需要 CUDA >= 11.8。请安装 CUDA Toolkit 或确认 PATH 包含 /usr/local/cuda/bin"
fi

# --- nvidia-smi / GPU ---
if command -v nvidia-smi &>/dev/null; then
    GPU_INFO=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -1)
    if [ -n "$GPU_INFO" ]; then
        ok "GPU 已找到：$GPU_INFO"
        # 检查显存是否满足 smoke test（需要约 500MB）
        MEM_MiB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
        if [ -n "$MEM_MiB" ] && [ "$MEM_MiB" -lt 4096 ]; then
            warn "GPU 显存 ${MEM_MiB}MiB 较小，smoke test 可运行但 1M+ 规模可能 OOM"
        fi
    else
        warn "nvidia-smi 存在但未检测到 GPU（可能在容器/虚拟机中）"
    fi
else
    fail "nvidia-smi 未找到。没有 GPU 则无法运行 BANG（BANG_Base 强依赖 CUDA）"
fi

# --- cmake ---
if command -v cmake &>/dev/null; then
    CMAKE_VER=$(cmake --version | head -1 | awk '{print $3}')
    ok "cmake 已找到：$CMAKE_VER"
else
    fail "cmake 未找到。需要 cmake >= 3.16。安装：sudo apt install cmake 或 pip install cmake"
fi

# --- make ---
if command -v make &>/dev/null; then
    ok "make 已找到：$(make --version | head -1)"
else
    fail "make 未找到"
fi

# --- g++ ---
if command -v g++ &>/dev/null; then
    GXX_VER=$(g++ --version | head -1)
    ok "g++ 已找到：$GXX_VER"
else
    fail "g++ 未找到。需要 gcc/g++ >= 11.0"
fi

# --- python3 ---
if command -v python3 &>/dev/null; then
    PY_VER=$(python3 --version)
    ok "python3 已找到：$PY_VER"
else
    fail "python3 未找到"
fi

# --- Python 包 ---
if command -v python3 &>/dev/null; then
    for pkg in numpy matplotlib; do
        if python3 -c "import $pkg" 2>/dev/null; then
            ok "Python 包：$pkg"
        else
            warn "Python 包缺失：$pkg（运行 pip install $pkg）"
        fi
    done
fi

# --- Boost ---
# 简单检查常见头文件路径
BOOST_FOUND=0
for d in /usr/include/boost /usr/local/include/boost /opt/homebrew/include/boost; do
    if [ -d "$d" ]; then
        ok "Boost 头文件：$d"
        BOOST_FOUND=1
        break
    fi
done
if [ $BOOST_FOUND -eq 0 ]; then
    warn "未找到 Boost 头文件（/usr/include/boost 等）。如编译报错请安装：sudo apt install libboost-all-dev"
fi

# --- DiskANN build_disk_index ---
DISKANN_FOUND=0
# 常见安装路径
for p in \
    "$(command -v build_disk_index 2>/dev/null)" \
    "$HOME/DiskANN/build/apps/build_disk_index" \
    "/usr/local/bin/build_disk_index" \
    "$(find /root /home 2>/dev/null -name build_disk_index -type f 2>/dev/null | head -1)"; do
    if [ -n "$p" ] && [ -x "$p" ]; then
        ok "DiskANN build_disk_index：$p"
        DISKANN_FOUND=1
        break
    fi
done
if [ $DISKANN_FOUND -eq 0 ]; then
    warn "未找到 DiskANN build_disk_index。SIFT10K smoke test 使用预构建文件，不需要 DiskANN；但 SIFT1M+ 构图需要它"
    warn "安装：git clone https://github.com/microsoft/DiskANN && cd DiskANN && mkdir build && cd build && cmake .. && make -j8"
fi

# --- 检查 sift10kfiles 预构建文件 ---
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SIFT10K_DIR="$REPO_ROOT/../BANG-Billion-Scale-ANN/sift10kfiles"
echo ""
echo "--- 预构建文件（SIFT10K smoke test）---"
if [ -d "$SIFT10K_DIR" ]; then
    for f in bang_search sift10k_index_disk.bin sift10k_index_pq_compressed.bin sift10k_index_pq_pivots.bin sift10k_index_disk_metadata.bin siftsmall_query.bin sift10k_groundtruth.bin; do
        if [ -f "$SIFT10K_DIR/$f" ]; then
            SIZE=$(du -sh "$SIFT10K_DIR/$f" 2>/dev/null | cut -f1)
            ok "$f（$SIZE）"
        else
            fail "缺失：$SIFT10K_DIR/$f"
        fi
    done
else
    fail "sift10kfiles 目录不存在：$SIFT10K_DIR"
fi

# --- 检查 sift1m_data ---
SIFT1M_DIR="$REPO_ROOT/../BANG-Billion-Scale-ANN/sift1m_data"
echo ""
echo "--- sift1m_data（SIFT1M 路线）---"
if [ -d "$SIFT1M_DIR" ]; then
    for f in sift1m_base.bin sift1m_query.bin sift1m_groundtruth.bin; do
        if [ -f "$SIFT1M_DIR/$f" ]; then
            SIZE=$(du -sh "$SIFT1M_DIR/$f" 2>/dev/null | cut -f1)
            ok "$f（$SIZE）"
        else
            warn "缺失：$SIFT1M_DIR/$f（SIFT1M 路线跳过）"
        fi
    done
    # 检查 index 文件（如果已构建）
    for f in sift1m_index_disk.bin sift1m_index_pq_compressed.bin; do
        if [ -f "$SIFT1M_DIR/$f" ]; then
            ok "已构建：$f"
        else
            warn "未构建：$f（需先运行 DiskANN build_disk_index + bang_preprocess.py）"
        fi
    done
else
    warn "sift1m_data 目录不存在（SIFT1M 路线跳过）"
fi

echo ""
echo "====== 检查结果 ======"
echo "  通过：$PASS"
echo "  警告：$WARN"
echo "  失败：$FAIL"
echo ""

if [ $FAIL -gt 0 ]; then
    echo "存在 $FAIL 个失败项，请修复后再运行 smoke test。"
    exit 1
elif [ $WARN -gt 0 ]; then
    echo "存在 $WARN 个警告项，smoke test 可能可运行，请确认后继续。"
    exit 0
else
    echo "所有检查通过，可以运行 smoke test。"
    exit 0
fi
