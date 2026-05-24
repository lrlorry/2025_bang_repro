# 2025 BANG 复现

参照 [2024_cagra_repro](../2024_cagra_repro) 结构，对 BANG（Billion-scale ANNS on GPU，IEEE TBD 2025）进行复现。

## 结构

```
plain/              BANG reference 实现（顺序执行，可读性优先）
engineered/         BANG 工程优化实现（4 streams / 8-thread PQ / shared-mem merge / OpenMP）
common/             公共头文件
report/             源码审计与分析报告
CMakeLists.txt
bang_cagra_report.md  BANG + CAGRA 综合技术报告
```

## 构建

```bash
mkdir build && cd build
cmake ..
make -j$(nproc)
./bang_plain        # reference 版
./bang_engineered   # 工程优化版
```

需要：CUDA 12+，cmake 3.18+，OpenMP，支持 CUDA 的 GPU。

## plain vs engineered 对比

| 功能 | plain | engineered | 对应 BANG 源码 |
|---|---|---|---|
| PQ table layout | 原始 [M×256×dim] | **转置 [M×dim×256]** | `pqTable_T`（cu:273）|
| PQ distance | 1 thread/neighbor | **8-thread + warp reduce** | `THREADS_PER_NEIGHBOR=8`（cu:184）|
| Worklist merge | serial insert | **shared-mem 2-way merge** | `compute_BestLSets_par_merge`（cu:439）|
| CPU-GPU overlap | 无（顺序同步）| **4 CUDA streams** | `streamParent/Children/FP/Kernels` |
| CPU graph fetch | 单线程 | **OpenMP 并行** | `numCPUthreads=64`（cu:413）|
| FP vector 预取 | rerank 前同步拉取 | **search loop 内异步 H2D** | `streamFPTransfers` |

两者逻辑完全等价，搜索结果一致。

## Kernel 对照表

| kernel | plain | engineered | BANG 原版 |
|---|---|---|---|
| PQ dist table | `populate_pq_dist_table_kernel` | `populate_pq_dist_table_T_kernel` | `populate_pqDist_par` |
| Bloom filter | `bloom_filter_kernel` | `bloom_filter_eng_kernel` | `neighbor_filtering_new` |
| PQ distance | `pq_distance_kernel` | `pq_distance_8thread_kernel` | `compute_neighborDist_par` |
| Worklist update | `update_worklist_kernel` | `merge_worklist_kernel` | sort_msort + merge |
| Parent select | `select_next_parent_kernel` | `select_parent_eng_kernel` | `compute_parent2` |
| Exact rerank | `exact_rerank_kernel` | `exact_rerank_eng_kernel` | `compute_L2Dist` + `compute_NearestNeighbours` |

## 报告

- `report/bang_audit_report.md` — 源码 findings（含 LOAD-01 类型安全 bug）
- `report/bang_source_map.md` — 数据结构 + 调用链 + kernel 索引
- `report/bang_vs_cagra.md` — BANG vs CAGRA 深度对比
- `report/bang_resource_plan.md` — 各规模资源估算
- `bang_cagra_report.md` — 综合技术报告
