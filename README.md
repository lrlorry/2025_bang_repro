# BANG Reproduction — Billion-scale ANNS on GPU

> IEEE Transactions on Big Data 2025  
> 默认复现路线：官方 BANG Base + DiskANN 产物。`plain/` 和 `engineered/` 仅保留作教学/源码理解用。

---

## 算法概述

BANG 将图索引（Vamana/DiskANN）驻留 CPU RAM，PQ 压缩向量驻留 GPU HBM，通过 CPU-GPU pipeline 实现近似最近邻搜索：

```
CPU RAM                          GPU HBM
┌──────────────────────┐         ┌─────────────────────────┐
│  HostGraph           │         │  PQ codes  [N × M]      │
│  adj[N × R]  ────────┼─────H2D─▶  PQ table [M × 256 × d] │
│  vecs[N × d] ────────┼──FP H2D─▶  Worklist [Q × L]       │
└──────────────────────┘         │  Bloom    [Q × BF]      │
                                 └─────────────────────────┘

Search pipeline (per iteration):
  GPU  ──D2H──▶  parent ids
  CPU  ──fetch──▶ adj[parent]   (OpenMP × 64 threads)
  CPU  ──H2D──▶  neighbor ids
  GPU  ──────▶  Bloom filter → PQ dist → worklist merge
  GPU  ──────▶  exact rerank (full FP, prefetched async)
```

---

## 推荐路线：官方 BANG Base

不要再用本目录的教学版 builder 跑正式结果。正式复现走：

```text
TexMex/BigANN vectors
  -> DiskANN .bin
  -> DiskANN build_disk_index
  -> official BANG bang_preprocess.py
  -> official BANG_Base/build/bang_search
```

构建官方 BANG：

```bash
bash scripts/build_official_bang.sh
```

把 `fvecs/bvecs/ivecs` 转成 DiskANN/BANG `.bin`：

```bash
cd 2025_bang_repro
mkdir -p build && cd build && cmake .. && make fvecs_to_bin
cd ..
bash scripts/convert_vecs_to_diskann_bin.sh sift1m_data/sift_base.fvecs  results/official/sift1m_base.bin  float
bash scripts/convert_vecs_to_diskann_bin.sh sift1m_data/sift_query.fvecs results/official/sift1m_query.bin float
```

计算官方 `bang_search` 需要的 groundtruth：

```bash
COMPUTE_GROUNDTRUTH=/path/to/DiskANN/build/apps/compute_groundtruth \
bash scripts/compute_official_groundtruth.sh \
  --base results/official/sift1m_base.bin \
  --query results/official/sift1m_query.bin \
  --out results/official/sift1m_groundtruth.bin \
  --data-type float \
  --k 10
```

用 DiskANN 原版构图并生成 BANG 需要的 `_disk.bin/_disk_metadata.bin`：

```bash
BUILD_DISK_INDEX=/path/to/DiskANN/build/apps/build_disk_index \
bash scripts/prepare_official_diskann_index.sh \
  --data results/official/sift1m_base.bin \
  --prefix results/official/sift1m_index \
  --data-type float \
  --dim 128 \
  -R 64 -L 200 -B 1 -M 48
```

运行官方 BANG：

```bash
bash scripts/run_official_bang.sh \
  --prefix results/official/sift1m_index \
  --query results/official/sift1m_query.bin \
  --gt results/official/sift1m_groundtruth.bin \
  --numq 10000 \
  --k 10 \
  --data-type float \
  --dist-fn l2
```

注意：groundtruth 也必须是 DiskANN/BANG `.bin` truthset 格式：`[N, K, ids, dists]`。TexMex `.ivecs` 不能直接喂给官方 `bang_search`。

---

## 文件结构

```
plain/
  plain_build.cu/cuh    教学版 Vamana 图构建（CPU 串行，不用于正式复现）
  plain_search.cu/cuh   教学版 GPU PQ 搜索（顺序执行）
  plain_main.cu         主程序
  config.cuh            超参数
engineered/
  engineered_build.cu/cuh  教学版 Vamana 图构建（OpenMP 并行，不用于正式复现）
  engineered_search.cu/cuh 教学版 GPU PQ 搜索（4 streams / 8-thread PQ / shared-mem merge）
  engineered_main.cu    主程序
  config.cuh            超参数 + kTPNbr=8
common/
  cuda_utils.cuh        CUDA_CHECK 宏
bench/
  bench.sh              benchmark 脚本（timing + recall）
  plot.py               生成对比图（需要 matplotlib）
scripts/
  build_official_bang.sh             构建官方 BANG_Base
  prepare_official_diskann_index.sh  调 DiskANN build_disk_index + 官方 bang_preprocess.py
  run_official_bang.sh               运行官方 bang_search
  convert_vecs_to_diskann_bin.sh     TexMex vecs -> DiskANN .bin
  compute_official_groundtruth.sh    调 DiskANN compute_groundtruth
report/
  bang_audit_report.md  源码审计（含 LOAD-01 bug）
  bang_source_map.md    数据结构 + 调用链 + kernel 索引
  bang_vs_cagra.md      BANG vs CAGRA 深度对比
  bang_resource_plan.md 各规模资源估算
bang_cagra_report.md    BANG + CAGRA 综合技术报告
```

---

## 构建

```bash
# 需要：CUDA 10+，cmake 3.3+，OpenMP，支持 CUDA 的 GPU
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

---

## 运行

```bash
# reference 版（CPU Vamana build + GPU PQ search，顺序执行）
./bang_plain

# 工程优化版（OpenMP build + 4-stream GPU search）
./bang_engineered
```

输出示例：
```
BANG plain reproduction (Vamana build + GPU PQ search)
N=8192, dim=128, numQ=16, R=16, M=8, L=32, topK=10

[build] Vamana plain: N=8192 dim=128 R=16 alpha=1.2 L_build=64
[build] medoid = 3142
[build] Vamana plain done.

Running BANG plain search...
Recall@10: 0.756  (121 / 160)
```

---

## plain vs engineered 实现对比

### 图构建（Vamana 算法）

| 阶段 | plain | engineered | 对应 BANG 源码 |
|---|---|---|---|
| 外层循环 | 串行 for | **OpenMP parallel for** | `bang_preprocess.py` |
| 距离计算 | CPU 串行 L2 | CPU L2（`batch_l2_kernel` 预留） | greedy_search |
| α-RNG 裁剪 | 串行 `robust_prune` | 并行收集 + 串行 prune | `robust_prune()` |
| 反向边 pass-2 | 串行 | 串行（正确性优先） | Vamana pass-2 |

### GPU 搜索 Pipeline

| 优化点 | plain | engineered | 对应 BANG 源码行 |
|---|---|---|---|
| PQ table 布局 | `[M×256×d]` 原始 | **`[M×d×256]` 转置** | `pqTable_T`（cu:273）|
| PQ distance | 1 thread/neighbor | **8-thread + warp reduce** | `THREADS_PER_NEIGHBOR=8`（cu:184）|
| Worklist 更新 | 串行插入 O(R·L) | **shared-mem 2-way merge** | `compute_BestLSets_par_merge`（cu:439）|
| CPU-GPU 重叠 | 顺序同步 | **4 CUDA streams** | `streamParent/Children/FP/Kernels` |
| CPU graph fetch | 单线程 | **OpenMP 并行** | `numCPUthreads=64`（cu:413）|
| FP vector 预取 | rerank 前同步 H2D | **search loop 内异步 H2D** | `streamFPTransfers` |

### Kernel 对照表

| kernel | plain | engineered | BANG 原版 |
|---|---|---|---|
| PQ dist table | `populate_pq_dist_table_kernel` | `populate_pq_dist_table_T_kernel` | `populate_pqDist_par` |
| Bloom filter | `bloom_filter_kernel` | `bloom_filter_eng_kernel` | `neighbor_filtering_new` |
| PQ distance | `pq_distance_kernel` | `pq_distance_8thread_kernel` | `compute_neighborDist_par` |
| Worklist | `update_worklist_kernel` | `merge_worklist_kernel` | sort_msort + merge |
| Parent select | `select_next_parent_kernel` | `select_parent_eng_kernel` | `compute_parent2` |
| Exact rerank | `exact_rerank_kernel` | `exact_rerank_eng_kernel` | `compute_L2Dist` + `compute_NearestNeighbours` |

---

## Benchmark & 对比图

```bash
# 运行 benchmark，输出 results/
cd bench && bash bench.sh

# 生成对比图（需要 pip install matplotlib numpy）
python3 plot.py
# 输出：figures/recall_qps.png  figures/build_time.png  figures/speedup.png
```

生成图说明：

| 图 | 内容 |
|---|---|
| `recall_qps.png` | QPS vs Recall@10 曲线（plain / engineered / BANG 原版参考线）|
| `build_time.png` | 构建耗时对比（plain / engineered，按阶段分解）|
| `speedup.png` | engineered 各优化点相对 plain 的加速比 |

---

## 与原论文的关键差异

| 项目 | 本复现 | BANG 原版 |
|---|---|---|
| 数据集 | 随机 Gaussian / SIFT1M | SIFT1B / DEEP1B |
| 图构建 | CPU Vamana（单机） | 分布式预处理 |
| PQ 训练 | 随机 centroid（demo）| OPQ/PQ 完整训练 |
| GPU | 单 GPU | 多 GPU |
| LOAD-01 bug | 已记录（`report/bang_audit_report.md`）| 原版存在类型安全问题 |

---

## 报告

- [`report/bang_audit_report.md`](report/bang_audit_report.md) — 源码 findings（含 LOAD-01 类型安全 bug）
- [`report/bang_source_map.md`](report/bang_source_map.md) — 数据结构 + 调用链 + kernel 索引
- [`report/bang_vs_cagra.md`](report/bang_vs_cagra.md) — BANG vs CAGRA 深度对比（20 个技术细节）
- [`report/bang_resource_plan.md`](report/bang_resource_plan.md) — 各规模资源估算
- [`bang_cagra_report.md`](bang_cagra_report.md) — BANG + CAGRA 综合技术报告

---

## 参考

- BANG paper: *Billion-scale Approximate Nearest Neighbor Search on GPU* (IEEE TBD 2025)  
- DiskANN / Vamana: Jayaram et al., NeurIPS 2019  
- CAGRA: Ootomo et al., IPDPS 2023  
- 相关复现: [2024_cagra_repro](../2024_cagra_repro)
