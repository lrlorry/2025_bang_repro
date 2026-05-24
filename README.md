# BANG Reproduction — Billion-scale ANNS on GPU

> IEEE Transactions on Big Data 2025  
> 复现实现：plain reference + engineered 工程优化版，附 Vamana 图构建与 GPU PQ 搜索全流程

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

## 文件结构

```
plain/
  plain_build.cu/cuh    Vamana 图构建（CPU 串行）
  plain_search.cu/cuh   GPU PQ 搜索（顺序执行）
  plain_main.cu         主程序
  config.cuh            超参数
engineered/
  engineered_build.cu/cuh  Vamana 图构建（OpenMP 并行）
  engineered_search.cu/cuh GPU PQ 搜索（4 streams / 8-thread PQ / shared-mem merge）
  engineered_main.cu    主程序
  config.cuh            超参数 + kTPNbr=8
common/
  cuda_utils.cuh        CUDA_CHECK 宏
bench/
  bench.sh              benchmark 脚本（timing + recall）
  plot.py               生成对比图（需要 matplotlib）
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
