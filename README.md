# 2025 BANG 复现

参照 [2024_cagra_repro](../2024_cagra_repro) 结构，对 BANG（Billion-scale ANNS on GPU，IEEE TBD 2025）进行复现。

## 结构

```
plain/          BANG plain CUDA 实现（host graph + GPU PQ，CPU-GPU pipeline）
common/         公共头文件（CUDA_CHECK 等）
report/         源码审计报告
CMakeLists.txt  构建
```

## 构建

```bash
mkdir build && cd build
cmake ..
make -j$(nproc)
./bang_plain
```

需要：CUDA 12+，cmake 3.18+，支持 CUDA 的 GPU。

## plain 实现说明

`plain/` 是 BANG 核心算法的 reference 实现，对应 `BANG_Base/bang_search.cu` 的主路径：

| BANG 原版 | plain 等价 |
|---|---|
| `populate_pqDist_par` | `populate_pq_dist_table_kernel` |
| `neighbor_filtering_new` | `bloom_filter_kernel` |
| `compute_neighborDist_par` | `pq_distance_kernel` |
| `compute_BestLSets_par_sort_msort` + `compute_BestLSets_par_merge` | `update_worklist_kernel`（串行）|
| `compute_parent2`（eager prefetch）| `select_next_parent_kernel` |
| `compute_L2Dist` + `compute_NearestNeighbours` | `exact_rerank_kernel` |

plain 版与 BANG 原版的差异（性能，逻辑等价）：
- **无 CUDA streams**：原版用 4 streams overlap CPU/GPU，plain 版顺序同步执行
- **无 OpenMP**：原版用 `numCPUthreads=64` 并行 fetch graph，plain 版单线程
- **串行 worklist**：原版用 parallel merge sort + shared memory merge，plain 版 serial insert
- **无异步 FP prefetch**：原版 search loop 内异步 H2D candidate vectors，plain 版 rerank 前同步拉取

## 技术报告

- `report/bang_audit_report.md` — 源码 findings（含 LOAD-01 类型安全 bug）
- `report/bang_source_map.md` — 数据结构 + 调用链 + kernel 索引
- `report/bang_vs_cagra.md` — BANG vs CAGRA 深度对比
- `report/bang_resource_plan.md` — 各规模（SIFT10K ~ SIFT1B）资源估算
- `bang_cagra_final_technical_report.md` — 综合技术报告
