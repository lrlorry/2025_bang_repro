# BANG 复现项目

> **论文**：BANG: Billion-Scale Approximate Nearest Neighbour Search Using a Single GPU  
> **期刊**：IEEE Transactions on Big Data, 2025, vol.11, no.6, pp.3142–3157  
> **官方源码**：`../BANG-Billion-Scale-ANN/`（本 repo 同级目录）  
> **本项目目标**：可复现框架，从 SIFT10K smoke test 开始，逐步扩展到 1M / 10M / 100M / 1B。  
> **审计基准**：`BANG-Billion-Scale-ANN/BANG_Base/` 源码，含 `bang_search.cu`、`bang_search.cuh`、`test_driver.cpp`，以及 `ReadMe.pdf`。

---

## 1. BANG 的核心是什么

BANG 解决的是 **单 GPU HBM 容量不足以存放 billion-scale graph + full vectors** 的问题，而不是单纯的 kernel 性能优化。

```
CPU host RAM（数十 GB ~ 640 GB）：
  pIndex           ← Vamana/DiskANN graph + full vectors
                     格式：每条 entry = [full vector coords | neighbor_count | neighbor_ids]
                     来源：bang_preprocess.py 处理 DiskANN 生成的 disk.index

GPU HBM（~40–80 GB）：
  d_compressedVectors  ← 所有向量的 PQ compressed codes（N × uChunks bytes）
  d_pqTable            ← PQ pivot table（转置为 D×256，改善 coalescing）
  d_centroid           ← dataset centroid（用于零中心化）
  d_pqDistTables       ← per-query PQ distance table（numQueries × uChunks × 256）
  d_BestLSets*         ← worklist（top-L 候选）
  d_processed_bit_vec  ← per-query Bloom visited set（399887 bool/query）
  d_parents            ← 下一轮传回 CPU 的 parent ids
  d_FPSetCoordsList    ← 候选 full vectors（异步 H2D，用于 exact rerank）
  d_nearestNeighbours  ← 最终 top-k 输出
```

**CPU-GPU pipeline** 是 BANG 的核心性能机制：
每轮迭代，GPU 返回 parent ids → CPU 用 OpenMP 从 host graph 取 neighbors → H2D neighbors → GPU filter/PQ distance/sort/merge → GPU 提前取下轮 parent（eager prefetch）→ 循环。  
最终用 exact rerank（full vectors）补偿 PQ 近似误差。

---

## 2. 为什么不需要一开始跑 1B

1B 实验依赖：~640 GB host RAM、A100 80GB GPU、完整 DiskANN 构图（需数小时）、~1 TB 磁盘。  
但 **验证代码链路的正确性** 与数据规模无关。本项目采用增量策略：

```
SIFT10K  →  SIFT1M  →  SIFT10M  →  SIFT100M  →  SIFT1B
  ↑                       ↑              ↑            ↑
已有预构建 index        需构建 index    需更多 RAM   论文规模
本地可测（需 GPU）      几十分钟        ~50 GB RAM   640 GB RAM
```

---

## 3. 依赖

| 依赖 | 版本 | 用途 |
|---|---|---|
| CUDA / nvcc | ≥ 11.8 | 编译 BANG_Base |
| NVIDIA GPU | A100 80GB（推荐）；V100/RTX 可用于小规模 | 运行 BANG |
| gcc/g++ | ≥ 11.0（C++11） | 编译 |
| cmake | ≥ 3.16 | CMakeLists.txt |
| Boost C++ | ≥ 1.74 | bang_search.cu 依赖 |
| DiskANN | 最新 main | 构建 graph / PQ 文件 |
| Python | ≥ 3.8 | bang_preprocess.py、estimate_resources.py 等 |
| numpy | 任意 | Python 脚本 |
| matplotlib | 任意 | plot_results.py |

**AutoDL 注意**：`/root/autodl-tmp/` 为数据盘，不要把大文件放系统盘。

---

## 4. 快速开始：SIFT10K smoke test（使用预构建 index）

```bash
# 确保在 2025_bang_repro 目录下运行
cd /path/to/2025_bang_repro

# Step 1：检查环境
bash scripts/check_env.sh

# Step 2：运行 smoke test（使用 ../BANG-Billion-Scale-ANN/sift10kfiles 中的预构建文件）
bash scripts/run_sift10k_smoke.sh
```

**预构建文件清单**（`../BANG-Billion-Scale-ANN/sift10kfiles/`）：

| 文件 | 大小 | 说明 |
|---|---|---|
| `bang_search` | 357 KB | 预编译二进制（Linux x86） |
| `libbang.so` | 1.2 MB | 共享库 |
| `sift10k_index_disk.bin` | 7.4 MB | host graph（Vamana，bang_preprocess 后） |
| `sift10k_index_disk_metadata.bin` | 32 B | medoid、degree、dim 等元数据 |
| `sift10k_index_pq_compressed.bin` | 1.2 MB | GPU PQ compressed vectors |
| `sift10k_index_pq_pivots.bin` | 133 KB | PQ pivots + centroid + chunk offsets |
| `siftsmall_query.bin` | 50 KB | 100 条查询（float32，128-dim） |
| `sift10k_groundtruth.bin` | 781 KB | groundtruth |

**正确命令格式**（来自 ReadMe.pdf）：

```bash
cd ../BANG-Billion-Scale-ANN/sift10kfiles
./bang_search ./sift10k_index ./siftsmall_query.bin ./sift10k_groundtruth.bin 100 10 float l2
```

**注意**：第一个参数是 **index 文件前缀**，不是完整文件名。`bang_search` 会自动拼接 `_disk.bin`、`_pq_compressed.bin`、`_pq_pivots.bin`、`_disk_metadata.bin` 等后缀。

---

## 5. SIFT1M 复现路径

本地已有 `../BANG-Billion-Scale-ANN/sift1m_data/`（base 488MB，已转为 `.bin` 格式），但**尚无对应的 DiskANN graph 文件**。需先构建 index。

### Step 1：安装并编译 DiskANN

```bash
git clone https://github.com/microsoft/DiskANN
cd DiskANN && mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j8
```

### Step 2：构建 Vamana Graph Index

```bash
# sift1m_base.bin 是 float32，128-dim，1M 向量
./apps/build_disk_index \
  --data_type float \
  --dist_fn l2 \
  --data_path ../../BANG-Billion-Scale-ANN/sift1m_data/sift1m_base.bin \
  --index_path_prefix sift1m_index \
  -R 64 -L 200 -B 1 -M 48

# B=1 表示 1 GiB 压缩预算：1GiB/1M = 1024 bytes/vector → uChunks ≈ 1024 / 1 = ...
# 实际 uChunks 由 DiskANN 根据 B 和 D 决定（约 100~128 chunks for D=128）
```

### Step 3：转换 index 格式（bang_preprocess.py）

```bash
python ../BANG-Billion-Scale-ANN/BANG_Base/bang_preprocess.py \
  sift1m_index_disk.index \
  sift1m_index_disk.bin \
  128 1 64
# 参数：<disk.index> <output.bin> <dim> <dtype_flag> <degree>
# 生成：sift1m_index_disk.bin + sift1m_index_disk_metadata.bin
```

### Step 4：构建 BANG_Base

```bash
bash scripts/build_bang_base.sh
```

### Step 5：运行搜索

```bash
./bang_search sift1m_index ./sift1m_query.bin ./sift1m_groundtruth.bin 10000 10 float l2
# 程序会提示输入 worklist length（L），或自动 sweep L=10,22,...,MAX_L
```

---

## 6. 生成 DiskANN/Vamana graph 和 PQ 文件说明

`build_disk_index` 生成以下文件，BANG 需要的是：

| DiskANN 输出 | BANG 使用方式 |
|---|---|
| `<prefix>_disk.index` | 经 `bang_preprocess.py` 转为 `_disk.bin` + `_disk_metadata.bin` |
| `<prefix>_pq_compressed.bin` | 直接使用 |
| `<prefix>_pq_pivots.bin` | 直接使用（包含 pivots + centroid + chunk_offsets 三个 section） |

**BANG 的 `bang_load()` 文件拼接逻辑**（`bang_search.cu:139-145`）：

```cpp
string pqPivots_file    = prefix + "_pq_pivots.bin"
string compressedVector = prefix + "_pq_compressed.bin"
string graphFile        = prefix + "_disk.bin"
string graphMetadata    = prefix + "_disk_metadata.bin"
```

因此 **index_path_prefix** 必须与上述后缀匹配。

---

## 7. 各规模资源估算

详见 [report/bang_resource_plan.md](report/bang_resource_plan.md)。

快速参考（float32，dim=128，R=64，B 参数控制 PQ 压缩比）：

| 规模 | base vectors (host) | graph index (host) | PQ compressed (GPU) | host RAM 总计 | GPU HBM |
|---|---|---|---|---|---|
| SIFT10K | ~5 MB | ~7 MB | ~1 MB | < 1 GB | ~40 GB A100 绰绰有余 |
| SIFT1M | ~488 MB | ~700 MB | ~100 MB | ~2 GB | 同上 |
| SIFT10M | ~4.9 GB | ~7 GB | ~1 GB | ~15 GB | ~10 GB PQ |
| SIFT100M | ~49 GB | ~70 GB | ~10 GB | ~130 GB | ~10 GB PQ |
| SIFT1B | ~490 GB | ~640 GB | ~100 GB | ~650 GB+ | ~100 GB（需 A100 80GB×2 或压缩） |

---

## 8. 关键编译参数与实验宏

以下参数是 **compile-time 约束**，修改后需重新编译（`bang_search.cu`）：

| 宏/常量 | 默认值 | 含义 |
|---|---|---|
| `MAX_R` | 64 | graph degree 上界，必须与 DiskANN 构图的 `-R` 一致 |
| `MAX_L` | （见源码头部）| worklist 上界，限制 `2*L ≤ 1024`（merge kernel 线程数）|
| `BF_ENTRIES` | 399887 | 每 query 的 Bloom filter 大小（bool 数组，非 bit-packed）|
| `NAX_EXTRA_ITERATION` | 50 | search iteration 上界追加量 |
| `MAX_PARENTS_PERQUERY` | `MAX_L + 50` | rerank candidate 数量上界 |
| `_TIMERS` | 注释 | 启用详细 kernel 计时 |
| `_NO_PRETECH` | 注释 | 禁用 eager parent prefetch（用于 ablation）|
| `_NO_ASYNC_FP` | 注释 | 禁用异步 full-vector H2D（用于 ablation）|

**numCPUthreads**（`bang_search.cu:413`）硬编码为 64，TODO 注释说明应从平台动态获取，AutoDL 上需确认实际 CPU 核数。

---

## 9. 目录结构

```text
2025_bang_repro/
├── README.md                          # 本文件
├── run.sh                             # 一键运行入口（从 smoke test 到 sweep）
├── Makefile                           # 快捷任务
├── scripts/
│   ├── check_env.sh                   # 检查 nvcc、GPU、cmake、DiskANN 等
│   ├── build_bang_base.sh             # 编译 ../BANG-Billion-Scale-ANN/BANG_Base
│   ├── run_sift10k_smoke.sh           # 用预构建文件跑 SIFT10K smoke test
│   ├── estimate_resources.py          # 估算不同规模的内存/磁盘/HBM 需求
│   ├── parse_bang_output.py           # 解析 BANG stdout → CSV
│   └── plot_results.py                # 读取 results/*.csv → figures/
├── results/
│   └── .gitkeep
├── figures/
│   └── .gitkeep
├── report/
│   ├── bang_source_map.md             # 核心调用链和数据结构索引
│   ├── bang_audit_report.md           # 源码审计结论（论文 vs 源码差异）
│   ├── bang_repro_tests.md            # 测试设计（部分需要 GPU 执行）
│   ├── bang_resource_plan.md          # 各规模资源计划
│   └── bang_vs_cpu_diskann.md         # BANG vs 传统 CPU DiskANN 对比
└── bang_cagra_final_technical_report.md  # 综合技术报告（勿删）
```

---

## 10. 已知限制与 TODO

- `BANGSearch<T>` 构造函数（`bang_search.cu:73`）始终 `new BANGSearchInner<int>()`，忽略模板参数 T，存在类型安全隐患。`uint8_t` / `int8_t` 路径是否正确需实验验证。
- `compute_neighborDist_par_cachewarmup` 在 `bang_search.cuh` 中声明但主路径未确认使用，可能是实验残留。
- `numCPUthreads=64` 硬编码，NUMA 效应未量化。
- 所有性能结论（QPS/recall 数字）均需 GPU 实际运行。当前无 GPU 环境，结论均为静态审计推断，标记为 `needs_test`。
