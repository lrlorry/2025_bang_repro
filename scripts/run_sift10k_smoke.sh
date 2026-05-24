#!/usr/bin/env bash
# SIFT10K smoke test
# 使用 ../BANG-Billion-Scale-ANN/sift10kfiles 中的预构建文件
# 用法：bash scripts/run_sift10k_smoke.sh
#
# 预构建文件说明（来自 ReadMe.pdf Section 4）：
#   bang_search          预编译二进制（Linux x86, CUDA）
#   sift10k_index_*      DiskANN 构建 + bang_preprocess.py 处理后的 SIFT10K index
#   siftsmall_query.bin  100 条查询（float32, 128-dim）
#   sift10k_groundtruth.bin
#
# 正确命令（来自 ReadMe.pdf）：
#   ./bang_search ./sift10k_index ./siftsmall_query.bin ./sift10k_groundtruth.bin 100 10 float l2
#
# 参数说明（来自 test_driver.cpp main()）：
#   argv[1] = index prefix（bang_load 自动拼接 _disk.bin 等后缀）
#   argv[2] = query file
#   argv[3] = groundtruth file
#   argv[4] = numQueries
#   argv[5] = recall_k
#   argv[6] = dtype（float / uint8 / int8）
#   argv[7] = dist_fn（l2 / mips）
#   argv[8] = mode（不填则 auto sweep；填 interactive 则手动输入 L）

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SIFT10K_DIR="$REPO_ROOT/../BANG-Billion-Scale-ANN/sift10kfiles"
RESULTS_DIR="$REPO_ROOT/results"
LOG_FILE="$RESULTS_DIR/sift10k_smoke.log"

mkdir -p "$RESULTS_DIR"

echo "====== SIFT10K Smoke Test ======"
echo "预构建文件目录：$SIFT10K_DIR"
echo "结果日志：$LOG_FILE"
echo ""

# --- 检查 GPU ---
if ! command -v nvidia-smi &>/dev/null; then
    MSG="[SKIP] nvidia-smi 未找到，当前机器没有 NVIDIA GPU。
原因：BANG_Base 是 CUDA 程序，必须有 GPU 才能运行。
如在 AutoDL 或本地 GPU 机器上运行，请确认驱动已安装。
此次 smoke test 跳过，但环境检查和文件验证仍可继续。"
    echo "$MSG"
    echo "$MSG" >> "$LOG_FILE"
    # 继续执行文件检查，但最终不运行 bang_search
    GPU_AVAILABLE=0
else
    GPU_INFO=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -1)
    if [ -z "$GPU_INFO" ]; then
        echo "[WARN] nvidia-smi 存在但未检测到 GPU（可能在容器中）"
        GPU_AVAILABLE=0
    else
        echo "[OK] GPU：$GPU_INFO"
        GPU_AVAILABLE=1
    fi
fi

# --- 检查预构建文件 ---
echo ""
echo "--- 检查预构建文件 ---"
MISSING=0
REQUIRED_FILES=(
    "bang_search"
    "sift10k_index_disk.bin"
    "sift10k_index_disk_metadata.bin"
    "sift10k_index_pq_compressed.bin"
    "sift10k_index_pq_pivots.bin"
    "siftsmall_query.bin"
    "sift10k_groundtruth.bin"
)
for f in "${REQUIRED_FILES[@]}"; do
    fp="$SIFT10K_DIR/$f"
    if [ -f "$fp" ]; then
        SIZE=$(du -sh "$fp" 2>/dev/null | cut -f1)
        echo "[OK]   $f（$SIZE）"
    else
        echo "[FAIL] 缺失：$fp"
        ((MISSING++))
    fi
done

if [ $MISSING -gt 0 ]; then
    MSG="[ERROR] 缺少 $MISSING 个预构建文件。请检查 ../BANG-Billion-Scale-ANN/sift10kfiles/ 是否完整。"
    echo "$MSG"
    echo "$MSG" >> "$LOG_FILE"
    exit 1
fi

# --- 检查 bang_search 是否可执行 ---
BANG_BIN="$SIFT10K_DIR/bang_search"
if [ ! -x "$BANG_BIN" ]; then
    chmod +x "$BANG_BIN" 2>/dev/null || true
    if [ ! -x "$BANG_BIN" ]; then
        echo "[WARN] bang_search 不可执行（无法 chmod），将尝试直接运行"
    fi
fi

# --- 检查架构兼容性 ---
ARCH_OK=1
if command -v file &>/dev/null; then
    BIN_ARCH=$(file "$BANG_BIN" 2>/dev/null)
    echo ""
    echo "bang_search 文件类型：$BIN_ARCH"
    if echo "$BIN_ARCH" | grep -qE "ARM|aarch64"; then
        echo "[WARN] bang_search 是 ARM 二进制，当前可能是 x86 机器，无法直接运行"
        ARCH_OK=0
    fi
fi

# --- 如果没有 GPU 或架构不匹配，跳过执行 ---
if [ $GPU_AVAILABLE -eq 0 ] || [ $ARCH_OK -eq 0 ]; then
    echo ""
    echo "====== Smoke Test 跳过（无 GPU 或架构不匹配）======"
    echo "文件检查通过，环境不满足运行条件。"
    echo "在有 NVIDIA GPU 的机器上，手动运行："
    echo ""
    echo "  cd $SIFT10K_DIR"
    echo "  ./bang_search ./sift10k_index ./siftsmall_query.bin ./sift10k_groundtruth.bin 100 10 float l2"
    echo ""
    echo "跳过原因已记录到：$LOG_FILE"
    {
        echo "=== SIFT10K Smoke Test 跳过 ==="
        echo "时间：$(date)"
        echo "GPU 可用：$GPU_AVAILABLE"
        echo "架构匹配：$ARCH_OK"
        echo "文件检查：全部通过"
        echo "手动运行命令："
        echo "  cd $SIFT10K_DIR"
        echo "  ./bang_search ./sift10k_index ./siftsmall_query.bin ./sift10k_groundtruth.bin 100 10 float l2"
    } >> "$LOG_FILE"
    exit 0
fi

# --- 运行 smoke test ---
echo ""
echo "--- 运行 BANG SIFT10K smoke test ---"
echo "命令：./bang_search ./sift10k_index ./siftsmall_query.bin ./sift10k_groundtruth.bin 100 10 float l2"
echo "（auto sweep 模式，自动从 L=10 递增到 MAX_L）"
echo ""

{
    echo "=== SIFT10K Smoke Test ==="
    echo "时间：$(date)"
    echo "命令：cd $SIFT10K_DIR && ./bang_search ./sift10k_index ./siftsmall_query.bin ./sift10k_groundtruth.bin 100 10 float l2"
    echo "---"
} >> "$LOG_FILE"

# 切换到 sift10kfiles 目录运行（bang_search 用相对路径引用 index 文件）
BANG_OUTPUT=$(cd "$SIFT10K_DIR" && \
    LD_LIBRARY_PATH="$SIFT10K_DIR:${LD_LIBRARY_PATH:-}" \
    ./bang_search ./sift10k_index ./siftsmall_query.bin ./sift10k_groundtruth.bin 100 10 float l2 \
    2>&1) || BANG_EXIT=$?

echo "$BANG_OUTPUT"
echo "$BANG_OUTPUT" >> "$LOG_FILE"

if [ "${BANG_EXIT:-0}" -ne 0 ]; then
    echo ""
    echo "[ERROR] bang_search 退出码：$BANG_EXIT"
    echo "常见原因："
    echo "  1. GPU 显存不足（smoke test 需约 500MB）"
    echo "  2. 预构建二进制与当前 CUDA driver 版本不兼容（需重新编译）"
    echo "  3. libbang.so 加载失败（LD_LIBRARY_PATH 问题）"
    echo "详见日志：$LOG_FILE"
    exit 1
fi

# --- 解析输出并保存 CSV ---
echo ""
echo "--- 解析结果 ---"
python3 "$REPO_ROOT/scripts/parse_bang_output.py" \
    --input_text "$BANG_OUTPUT" \
    --output_csv "$RESULTS_DIR/sift10k_smoke.csv" \
    --dataset "sift10k" \
    --dtype "float" \
    --dim 128 \
    --n_queries 100 \
    --recall_k 10

echo ""
echo "====== Smoke Test 完成 ======"
echo "结果已写入：$RESULTS_DIR/sift10k_smoke.csv"
echo "日志：$LOG_FILE"
echo ""
echo "下一步（如果 recall 正常）："
echo "  python3 scripts/plot_results.py --results_dir results --figures_dir figures"
