# BANG vs CAGRA 最终技术报告

> 综合范围：覆盖所有论文与源码审计轮次。  
> BANG 以论文主方法 **BANG Base** 为主；CAGRA 以论文设计与当前 cuVS/RAFT 源码为主。  
> 重要边界：CAGRA 当前 cuVS main 已经包含论文之后的工程扩展，例如 GPU graph optimization、VPQ-half、persistent search、MULTI_KERNEL search path、generated distance specialization matrix，这些不能全部等同于论文原始实现。BANG 当前源码也存在实验残留/未走路径，例如 `compute_neighborDist_par_cachewarmup` 声明、`_TIMERS/_NO_PRETECH/_NO_ASYNC_FP` 宏、硬编码 CPU thread 数等。

---

## 1. 一句话总览：BANG 和 CAGRA 的本质区别

**BANG 是"容量突破型系统"，为了在单 GPU 上处理 graph/full vectors 放不进 HBM 的 billion-scale ANNS，它把 Vamana/DiskANN graph 和 full vectors 放在 CPU host RAM，把 PQ compressed vectors 放在 GPU，并通过 CPU-GPU pipeline 完成检索；CAGRA 是"GPU-resident 性能优化型系统"，它设计 fixed out-degree、directional、no-hierarchy 的 GPU-friendly graph，并把 graph/dataset/search state 放在 GPU 内，通过 single-CTA/multi-CTA、warp splitting、shared-memory hash、top-M/candidate buffer 等 GPU search 优化提高。**

这意味着两者不是简单的"调得更快"。它们默认的资源假设不同：

- **BANG 更适合**：1B 级、single GPU、graph + full vectors 无法在进 GPU，但 host RAM 足够。
- **CAGRA 更适合**：graph + dataset 可以 GPU-resident，尤其 100M 级或压缩后能进 HBM 的场景。

---

## 2. BANG 深度解析

### 2.1 问题设定

BANG 解决的是 GPU ANNS 的容量问题，而不是纯粹 kernel 优化问题。传统 GPU graph search 通常要求 graph + dataset 都在 GPU 上，当数据量到 1B 级时，full vectors 与 graph index 的总大小远超显卡 HBM。BANG 的设计是：

```text
CPU host RAM:
  - Vamana / DiskANN graph index
  - full base vectors embedded in graph entry
  - CPU fetch parent node's neighbors

GPU HBM:
  - PQ compressed vectors
  - PQ distance tables
  - per-query Bloom visited set
  - worklist / neighbor buffers
  - candidate full vectors for reranking
```

源码对应：

| 组件 | 文件/函数 | 关键对象 |
|---|---|---|
| Public API | `BANG_Base/bang.h::BANGSearch<T>` | `bang_load`, `bang_alloc`, `bang_init`, `bang_query` |
| Base 主实现 | `BANG_Base/bang_search.cu::BANGSearchInner<T>` | 加载、分配、初始化、查询 |
| 数据结构 | `BANG_Base/bang_search.cuh` | `GraphMedataData`, `IndexLoad`, `GPUInstance`, `HostInstance`, `SearchParams` |
| benchmark | `BANG_Base/test_driver.cpp::main`, `run_anns()` | CLI、QPS、recall、worklist sweep |

BANG 论文主方法对应 **BANG Base**，不是 `BANG_Inmemory` 或 `BANG_Exactdistance`。

---

### 2.2 数据布局

#### 2.2.1 Host graph / full vectors

BANG Base 中 graph 不放到 GPU。源码在：

- `BANG_Base/bang_search.cu::BANGSearchInner<T>::bang_load()`
- host pointer：`IndexLoad::pIndex`
- entry length：`ullIndex_Entry_LEN`

逻辑：

```text
pIndex + ullIndex_Entry_LEN * node_id
  -> full vector coordinates
  -> neighbor count
  -> neighbor ids
```

每轮检索中，CPU 根据 GPU 返回的 `parent` 从 `pIndex` 中读取：

- parent full vector，写入 host `FPSetCoordsList`
- parent neighbor count，写入 `numNeighbors_query`
- parent neighbor ids，写入 `neighbors`

源码位置：

- `BANG_Base/bang_search.cu::bang_query()`
- OpenMP fetch region
- 关键变量：`parents`, `neighbors`, `numNeighbors_query`, `FPSetCoordsList`, `pIndex`, `ullIndex_Entry_LEN`

#### 2.2.2 GPU PQ compressed vectors

BANG Base 在 GPU 上保存 compressed vectors：

- 文件：`BANG_Base/bang_search.cu::bang_load()`
- 关键变量：`compressedVectors`, `d_compressedVectors`, `uChunks`

加载流程：

```text
read *_pq_compressed.bin
  -> host compressedVectors
  -> cudaMalloc(d_compressedVectors, N * uChunks)
  -> cudaMemcpy H2D
  -> free(host compressedVectors)
```

这是 BANG 能跑 1B 的关键：GPU 上不保存 full vectors，只保存每个 vector 的 PQ code。

#### 2.2.3 PQ pivots / centroid / chunk offsets

文件/函数：

- `BANG_Base/bang_search.cu::bang_load()`
- 关键变量：`pqTable`, `pqTable_T`, `centroid`, `chunksOffset`, `d_pqTable`, `d_centroid`, `d_chunksOffset`

重要源码优化：

```text
pqTable 原始布局: 256 centroids × D
pqTable_T 转置布局: D × 256
```

目的：在 `populate_pqDist_par<T>()` 中，线程以 centroid id 连续访问 `d_pqTable_T + 256 * dim`，提高 coalescing。

#### 2.2.4 Per-query GPU search state

分配函数：

- `BANG_Base/bang_search.cu::bang_alloc()`

关键 GPU buffers：

| Buffer | 用途 |
|---|---|
| `d_queriesFP` | query batch |
| `d_pqDistTables` | 每 query 的 `m × 256` PQ distance table |
| `d_neighbors_temp` / `d_neighbors` | CPU H2D neighbor list 与 filtered neighbor list |
| `d_neighborsDist_query` | filtered neighbors 到 query 的 PQ distance |
| `d_BestLSets`, `d_BestLSetsDist`, `d_BestLSets_visited` | worklist |
| `d_processed_bit_vec` | per-query Bloom visited set |
| `d_parents` | 下一轮要返回给 CPU 的 parent |
| `d_FPSetCoordsList` | reranking 的 candidate full vectors |
| `d_L2ParentIds`, `d_L2distances` | rerank candidate ids/distances |
| `d_nearestNeighbours` | 最终 top-k 输出 |

关键 pinned host buffers：

| Buffer | 用途 |
|---|---|
| `neighbors` | CPU fetch 后的 neighbors |
| `numNeighbors_query` | 每 query neighbor count |
| `parents` | GPU D2H parent ids |
| `FPSetCoordsList` | candidate full vectors |

---

### 2.3 算法流程

BANG Base 的 search 可以分为三大阶段：

```text
Stage 1: PQ distance table construction
Stage 2: Greedy graph search with CPU-GPU pipeline
Stage 3: Exact reranking
```

#### 2.3.1 Stage 1：PQDistTable construction

Kernel：

- `BANG_Base/bang_search.cu::populate_pqDist_par<T>`

Launch：

```cpp
<<<numQueries, 256, D * (sizeof(float) + sizeof(T)), streamKernels>>>
```

逻辑：

```text
for each query block:
  load query vector into shared memory
  load centroid into shared memory
  for each PQ chunk:
    for each dimension in chunk:
      256 threads compute distance to 256 centroids
    write d_pqDistTables[query, chunk, centroid]
```

机制：

- 每 query 一个 CUDA block。
- 256 threads 对 256 centroids 并行。
- chunk / dimension 顺序迭代。
- `d_pqTable_T` 转置布局提高 coalescing。

#### 2.3.2 Stage 2：Greedy graph search loop

主函数：

- `BANG_Base/bang_search.cu::BANGSearchInner<T>::bang_query()`

核心循环：

```text
GPU has current parent ids
  -> D2H parent ids
  -> CPU/OpenMP fetch parent neighbors from host graph
  -> H2D neighbors
  -> GPU Bloom filter visited nodes
  -> GPU compute PQ asymmetric distances
  -> GPU eager select next parent
  -> GPU sort new neighbors
  -> GPU merge into worklist
  -> repeat until no next parent
```

注意：源码默认的 eager prefetch 路径中，`compute_parent2()` 会在 sort/merge 前运行，从而让 CPU 尽早开始下一轮 parent fetch。这与 Algorithm 2 的直觉顺序不同，但符合论文的 prefetch optimization。

#### 2.3.3 Stage 3：Exact reranking

Kernels：

- `BANG_Base/bang_search.cu::compute_L2Dist<T>`
- `BANG_Base/bang_search.cu::compute_NearestNeighbours`

流程：

```text
search loop 中 CPU 已经将 candidate full vectors 异步传到 d_FPSetCoordsList
rerank 前 cudaStreamSynchronize(streamFPTransfers)
compute_L2Dist: exact distance from query to candidate full vectors
compute_NearestNeighbours: sort candidates and output top-k
```

目的：BANG 检索阶段用 PQ approximate distance，可能走误路径，最终 exact rerank 补偿 recall。

---

### 2.4 Kernel 级实现

#### 2.4.1 Kernel 总表

| Kernel | 文件/函数 | 功能 | 映射 | 主要瓶颈 |
|---|---|---|---|---|
| PQDistTable | `BANG_Base/bang_search.cu::populate_pqDist_par<T>` | 构造 per-query PQ distance table | 1 query / block, 256 threads / centroid | memory/store-heavy |
| Bloom filter | `neighbor_filtering_new` | visited filtering + compact neighbors | 1 query / block, thread loop over neighbors | random global memory + atomic |
| PQ distance | `compute_neighborDist_par` | PQ asymmetric distance | 1 query / block, 8 threads / neighbor | irregular compressed vector loads |
| Neighbor sort | `compute_BestLSets_par_sort_msort` | sort filtered neighbors | 1 query / block, one thread per neighbor-ish | branch + barrier |
| Worklist merge | `compute_BestLSets_par_merge` | merge new neighbors into top-L worklist | 1 query / block, `2L` threads | shared memory + sync |
| Initial parent | `compute_parent1` | initial parent selection | one thread/query | underutilization, but short |
| Eager parent | `compute_parent2` | next parent prefetch | one thread/query | critical path latency |
| Exact distance | `compute_L2Dist<T>` | rerank exact L2 | 1 query / block, one thread per candidate | full-vector memory load |
| Final sort | `compute_NearestNeighbours` | rerank candidate sort/top-k | 1 query / block | noncoalesced + barrier |
| Cache warmup residual | `compute_neighborDist_par_cachewarmup` | declared, not confirmed in Base main path | unknown | unconfirmed |

#### 2.4.2 `populate_pqDist_par<T>`

**瓶颈**：如果每个 query/centroid/dim 都直接访问 PQ centroid table，重复计算与 memory traffic 高。

**实现**：

- 文件：`BANG_Base/bang_search.cu`
- 函数：`populate_pqDist_par<T>`
- key variables：`d_pqTable_T`, `d_pqDistTables`, `d_queriesFP`, `d_chunksOffset`, `d_centroid`
- shared memory：query vector + centroid
- launch：`numQueries` blocks, `256` threads

**收益**：

- 后续每个 neighbor 的 PQ distance 变为 lookup + sum。
- PQ table transpose 改善 centroid access coalescing。

**代价**：

- `d_pqDistTables` 大小随 `numQueries × m × 256` 增大。
- `d_pqDistTables += diff*diff` 存在重复 read-modify-write，store-heavy。

#### 2.4.3 `neighbor_filtering_new`

**瓶颈**：graph search 会反复遇到同一 node，重复 distance 会浪费 GPU time，并且重复节点会污染 worklist。

**实现**：

- 文件：`BANG_Base/bang_search.cu`
- 函数：`neighbor_filtering_new`
- device hash functions：`hashFn1_d`, `hashFn2_d`
- key variables：`d_processed_bit_vec`, `d_neighbors_temp`, `d_neighbors`, `d_numNeighbors_query`
- 用两个 FNV-1a style hash 检查 per-query Bloom-style bool array。
- 未访问时 `atomicAdd(&d_numNeighbors_query[queryID], 1)` compact 到 `d_neighbors`。

**收益**：

- 去掉大量重复节点。
- 论文表示不加 visited filtering 会导致 SIFT-1B recall 最多下降 10×。

**代价**：

- Bloom filter 是 global bool array，不是 shared hash，也不是 bit-packed bitset。
- Hash access 是 random global memory access，uncoalesced。
- false positive 可能误标未访问 node。
- 无显式同步，存在轻微 race，论文认为 sync overhead 不值得。

#### 2.4.4 `compute_neighborDist_par`

**瓶颈**：对每个 filtered neighbor，需要把 `m` 个 PQ codes 转为 distance sum，neighbor ids 来自 graph traversal，compressed vector access 不连续。

**实现**：

- 文件：`BANG_Base/bang_search.cu`
- 函数：`compute_neighborDist_par`
- key variables：`d_neighbors`, `d_compressedVectors`, `d_pqDistTables`, `d_neighborsDist_query`
- `THREADS_PER_NEIGHBOR = 8`
- `512 threads / block` 覆盖 `64 neighbors`
- CUB primitive：`cub::WarpReduce<float, 8>`

**收益**：

- 8-thread tile 比单线程 per neighbor 更快。
- 比 full-vector distance 大幅降低 memory footprint。

**代价**：

- `d_compressedVectors[neighbor * n_chunks + i]` 不规则访问，coalescing 差。
- 算术强度低，memory-bound。
- 固定 8-thread group 依赖 `m/R` 参数，无法随意调。

#### 2.4.5 Sort / merge kernels

Kernels：

- `compute_BestLSets_par_sort_msort`
- `compute_BestLSets_par_merge`

**瓶颈**：GPU 上对小 list serial sort/insert 会让线程阻塞。

**实现**：

- neighbor list sort 使用 parallel merge sort 逻辑。
- worklist merge 使用 lower/upper bound 计算输出 position。
- `compute_BestLSets_par_merge` 使用 shared memory arrays：
  - `shm_currBestLSetsDist`
  - `shm_BestLSetsDist`
  - `shm_pos`
  - `shm_BestLSets`
  - `shm_BestLSets_visited`

**收益**：

- 避免 CPU/GPU 串行 worklist update。
- 每 query 一个 block，避免 inter-query atomic。

**代价**：

- 小 list 中 branch/barrier 成本明显。
- `MAX_L` 和 `2L <= 1024` 限制 worklist length。
- 不如 CAGRA 的 RAFT/cuVS bitonic/radix primitive 工程化。

#### 2.4.6 Parent kernels

Kernels：

- `compute_parent1`
- `compute_parent2`

**瓶颈**：如果等 sort/merge 完成才把下一轮 parent 给 CPU，CPU fetch 不能与 GPU sort/merge overlap。

**实现**：

- `compute_parent2()` 在默认路径中运行在 sort/merge 之前。
- 从新 neighbor list 和 worklist 中找下一个未访问 parent。
- 写 `d_parents`，供下一轮 D2H 给 CPU。

**收益**：

- 实现论文 eager candidate prefetch。
- 论文给出约 8~12% 或更高 ablation 收益。

**代价**：

- 单线程 per query，GPU 利用率差。
- 不严格 Algorithm 2 顺序不同，属于系统 pipeline 优化。
- `d_nextIter` 多 block 共用一 bool，语义上 true 可接受，但严格说需注意 race。

---

### 2.5 CPU-GPU pipeline

#### 2.5.1 Streams

BANG Base 创建四条 stream：

- 文件：`BANG_Base/bang_search.cu::bang_alloc()`
- variables：
  - `streamFPTransfers`
  - `streamParent`
  - `streamChildren`
  - `streamKernels`

用途：

| Stream | 用途 |
|---|---|
| `streamParent` | parent ids D2H |
| `streamChildren` | neighbor ids/count H2D |
| `streamFPTransfers` | candidate full vectors H2D |
| `streamKernels` | GPU search kernels |

#### 2.5.2 Pipeline 机制

典型一轮：

```text
1. GPU compute_parent2 writes d_parents
2. D2H parents on streamParent
3. GPU sort/merge previous results on streamKernels
4. CPU reads parents and OpenMP fetches host graph neighbors
5. H2D children on streamChildren
6. H2D candidate full vectors on streamFPTransfers
7. GPU filter + PQ distance + next parent
```

机制：

- CPU fetch 与 GPU sort/merge overlap。
- candidate full-vector H2D 与后续 GPU work overlap，直到 rerank 前再同步。
- 不传 graph，不传 full dataset，只传 frontier neighbors 和 candidate full vectors。

#### 2.5.3 同步点

| 同步点 | 文件/函数 | 目的 | 代价 |
|---|---|---|---|
| `cudaStreamSynchronize(streamParent)` | `bang_query()` | CPU 需要 parent ids | parent D2H barrier |
| OpenMP implicit barrier | CPU fetch region | 所有 query neighbors 准备好 | CPU-side barrier |
| `cudaStreamSynchronize(streamChildren)` | `bang_query()` | filter 依赖 H2D children | copy-kernel dependency |
| `cudaStreamSynchronize(streamKernels)` | loop control | nextIter / kernels 完成 | 每轮 barrier |
| `cudaStreamSynchronize(streamFPTransfers)` | before rerank | candidate full vectors available | rerank前等待 |

---

### 2.6 BANG 所有优化点：瓶颈 -> 实现 -> 收益 -> 代价

| 优化点 | 瓶颈 | 实现 | 收益 | 代价 |
|---|---|---|---|---|
| Host graph + GPU compressed vectors | 1B graph/full vectors 超 GPU HBM | `bang_load()` 读 `_disk.bin` 到 host `pIndex`，读 `_pq_compressed.bin` 到 `d_compressedVectors` | 让 GPU 可处理 billion-scale | CPU/PCIe 进入 critical path |
| PQ compressed vectors | full vectors 太大 | `d_compressedVectors` 常驻 GPU，`uChunks` codes/vector | 降低 HBM footprint | PQ 误差，需要 rerank |
| PQDistTable | 每次 distance 计算 centroid distance | `populate_pqDist_par<T>` per-query table | neighbor distance 变 lookup + sum | table memory 随 batch 增 |
| PQ table transpose | centroid table access 不连续 | `pqTable_T[col*256+row]` | 改善 coalescing | load-time transpose |
| 8-thread segmented PQ distance | 单线程 sum m chunks 慢 | `compute_neighborDist_par`, `THREADS_PER_NEIGHBOR=8`, CUB WarpReduce | 降低 per-neighbor span | irregular compressed vector load仍在 |
| Bloom visited filter | 重复节点污染 worklist | `neighbor_filtering_new`, `d_processed_bit_vec`, two hash functions | 减少重复 distance，提高 recall | global random access, false positive |
| Skip Bloom synchronization | Bloom sync overhead高 | filter kernel 无 `__syncthreads()`保护全局hash读写 | 降低 filter latency | race/false behavior风险 |
| Parallel neighbor sort | 小 list serial sort 浪费GPU | `compute_BestLSets_par_sort_msort` | GPU并行排序 | branch/barrier |
| Parallel worklist merge | 串行插入 top-L 慢 | `compute_BestLSets_par_merge` shared memory lower/upper bound | 更新 top-L worklist | `L`和`MAX_L/1024 threads`限制 |
| Eager candidate prefetch | CPU fetch 等待 worklist 完整更新 | `compute_parent2` 在 sort/merge 前取 next parent | CPU fetch 与 GPU sort/merge overlap | 逻辑复杂，非伪代码顺序 |
| Async memcpy + streams | CPU/GPU/copy互等 | 4 streams + `cudaMemcpyAsync` + pinned host buffers | 隐藏部分 PCIe latency | 同步点仍多处 |
| OpenMP CPU fetch | 单CPU线程fetch慢 | `#pragma omp parallel`, `numCPUthreads` | 多query并行访问局部 | NUMA效应，线程数硬编码/TODO |
| Candidate full-vector prefetch | rerank集中传 full vectors会stall | 检索中把 parent full vector写`FPSetCoordsList`并H2D | rerank前数据已部分到GPU | 传输未必最终top-k的vectors |
| Exact reranking | PQ search recall损失 | `compute_L2Dist<T>` + `compute_NearestNeighbours` | recall提升 | 额外full-vector传输和kernel |
| Worklist length tuning | recall/QPS trade-off | `worklist_length`, `d_BestLSets*` | 调recall | L大时merge/sort/iterations增加 |
| Batch size tuning | GPU occupancy不足 | query-level blocks | QPS随batch增大到饱和 | buffers线性增长 |
| Compression ratio tuning | HBM不够 | 调PQ chunks `m`，重新生成PQ files | 40GB级GPU可能可跑 | m太低导致detour/recall下降 |
| Compile-time macros | 实验不同路径 | `_TIMERS`, `_NO_PRETECH`, `_NO_ASYNC_FP`, `BF_ENTRIES`, `MAX_R`, `MAX_L` | 可做ablation/调参数 | 默认注释，需改源码再build |

---

### 2.7 BANG 的代价与限制

1. **CPU 在 critical path**：每轮必须 D2H parent，CPU fetch graph，H2D children。
2. **PCIe latency 无法完全消除**：只能通过 prefetch/async 部分隐藏。
3. **Compressed vector access 不规则**：`compute_neighborDist_par` 仍被 graph neighbor id 的随机性影响。
4. **Bloom filter global random access**：`d_processed_bit_vec` 太大，不能放 shared memory。
5. **PQ 误差**：search path 可能 detour，最终要 exact rerank。
6. **Large batch 才能填满 GPU**：小 batch 下 query/block 数不足。
7. **硬编码/常量限制**：`MAX_R=64`, `MAX_L`, `BF_ENTRIES`, `MAX_PARENTS_PERQUERY` 等影响泛化。
8. **外部依赖 DiskANN/Vamana/PQ 产物**：不是只给 base vectors 就能跑。
9. **当前源码有实验残留/断点**：`compute_neighborDist_par_cachewarmup` 声明、`BANGSearch<T>` 构造有 `new BANGSearchInner<int>()` 类型安全断点、CPU thread TODO。

---

## 3. CAGRA 深度解析

### 3.1 Graph design

CAGRA graph 的核心分型：

```text
fixed out-degree
directional
no hierarchy
```

#### 3.1.1 Fixed out-degree

瓶颈：GPU graph search 中 variable degree 会导致 CTA 工作量不均、load imbalance、低 degree 无法填满 CTA。

实现：

- 最终 graph shape 是 `N × graph_degree`。
- Public 参数：`cpp/include/cuvs/neighbors/cagra.hpp::index_params::graph_degree`
- 默认值：`graph_degree = 64`
- 构图中 `intermediate_graph_degree` 默认 128，最终 prune 到 `graph_degree`。

收益：

- 每个 node 扩展固定 `d` 个 children。
- search buffer candidate length 可直接设为 `search_width * graph_degree`。
- CUDA mapping 更稳定。

代价：

- 不像 variable-degree graph 那样能对不同节点自适应度数。
- degree 太小 recall不足，太大后劲不够。

#### 3.1.2 Directional

固定 out-degree 后，in-degree 自然不固定，所以 CAGRA graph 是 directed graph。反向边通过 reverse edge addition 改善 reachability，但最终仍保持 fixed out-degree。

#### 3.1.3 No hierarchy

瓶颈：HNSW hierarchy 更适合 CPU sequential entry search，GPU 可以随机采样多个起点并并行算 distance。

实现：

- search 初始化时 random sampling。
- 文件/函数：`cpp/src/neighbors/detail/cagra/jit_lto_kernels/device_common_jit.cuh::compute_distance_to_random_nodes_jit()`
- random node 生成可来自 seed 和 `xorshift64(gid ^ rand_xor_mask) % seed_index_limit`

收益：

- 避免 hierarchy traversal 的 sequential bottleneck。
- 更适合 GPU batch parallelism。

代价：

- 起点质量依赖 random sampling 数量和 graph reachability。
- 对 hard dataset 可能需要更大 `itopk_size`, `search_width`, `max_iterations`。

---

### 3.2 Graph construction

#### 3.2.1 调用链

```text
cuvs::neighbors::cagra::build(...)
  -> cpp/src/neighbors/cagra.cuh::detail::build(...)
    -> choose graph_build_params
       - NN-Descent if enough device memory
       - IVF-PQ fallback
       - ACE / iterative_search if user specified
    -> build_knn_graph(...)
       -> cuvs::neighbors::nn_descent::build(...)
       -> graph::sort_knn_graph(...)
    -> graph::optimize(...)
       -> prune_graph_gpu(...)
       -> make_reverse_graph_gpu(...)
       -> merge_graph_gpu(...)
    -> attach graph/dataset or VPQ-compressed dataset
```

#### 3.2.2 Initial kNN graph

文件/函数：

- `cpp/src/neighbors/detail/cagra/cagra_build.cuh::detail::build`
- `cpp/src/neighbors/detail/cagra/cagra_build.cuh::build_knn_graph`
- `cpp/src/neighbors/detail/cagra/graph_core.cuh::sort_knn_graph`
- `cpp/src/neighbors/detail/cagra/graph_core.cuh::kern_sort`

参数：

| 参数 | 源码字段 | 默认 |
|---|---|---|
| final degree `d` | `index_params::graph_degree` | 64 |
| initial degree `dinit` | `index_params::intermediate_graph_degree` | 128 |
| backend | `index_params::graph_build_params` | heuristic |

机制：

1. 构建 initial kNN graph，degree=`dinit`。
2. 对每个 source node 的 neighbor row 按跟距离排序。
3. 排序后的名次就是 rank-based pruning 的 rank。

收益：

- 用较大 `dinit` 保留候选边，再优化为 `d`。
- rank-based optimization 不需要额外计算大量 distances。

代价：

- initial graph 构建后排序占 device memory和时间。
- `dinit` 太小 graph quality差，太大构图成本高。

#### 3.2.3 Rank-based pruning

文件/函数：

- `cpp/src/neighbors/detail/cagra/graph_core.cuh::prune_graph_gpu`
- `cpp/src/neighbors/detail/cagra/graph_core.cuh::kern_fused_prune`

机制：

```text
for each node X:
  for each neighbor D of X with better rank:
    for each neighbor B of X with worse rank:
      if B appears in D's neighbor list:
         detour_count[B]++
select d neighbors with smallest detour_count
tie-break by original rank
```

关键结论：

- pruning 阶段不算 vector distance，只用 sorted kNN graph 的 rank/adjacency。
- 当前源码中有一个重要差异：`// if (kDB < kAB)` 条件被注释，因此源码没有严格检查 `rank(D,B) < rank(X,B)`，而是放宽为"D 的邻居中出现 B"。

收益：

- 避免 distance-based pruning 的大量 distance computation和distance table memory。
- 改善 graph 2-hop reachability。

代价：

- rank 是 distance 的近似，initial graph质量差时可能误判。
- `O(N dinit^3)` adjacency checks，虽然GPU并行，但work大。

#### 3.2.4 Reverse edge addition

文件/函数：

- `cpp/src/neighbors/detail/cagra/graph_core.cuh::make_reverse_graph_gpu`
- `cpp/src/neighbors/detail/cagra/graph_core.cuh::kern_make_rev_graph_k`

机制：

```text
for rank k in pruned graph:
  for each edge src -> dest at column k:
    pos = atomicAdd(rev_graph_count[dest], 1)
    if pos < d:
       rev_graph[dest, pos] = src
```

收益：

- 增强 graph reachability。
- 改善 strong connected components。

代价：

- reverse degree capped at `d`，overflow edges被丢弃。
- 按rank顺序写入顺序不完全稳定。

#### 3.2.5 Merge pruned graph + reverse graph

文件/函数：

- `cpp/src/neighbors/detail/cagra/graph_core.cuh::merge_graph_gpu`
- `cpp/src/neighbors/detail/cagra/graph_core.cuh::kern_merge_graph`

论文说法：从 pruned graph 和 reversed graph 各取 d/2 并 interleave，不足时补。

当前源码更准确说法：

```text
1. 把 pruned output row 放 shared memory。
2. 保护到 max(MST edges, d/2) 个 pruned/MST edges。
3. 从 reverse graph 中插入 reverse edges 到 protected prefix 后面。
4. reverse 不足时，把 pruned tail 保留。
```

收益：

- 保留重要的 pruned outgoing edges。
- 加入 incoming reverse edges 改善可达性。

代价：

- 不是严格子数 interleave。
- reverse overflow和duplicate handling会影响最终图质量。

#### 3.2.6 论文的 vs 当前源码

| 阶段 | 论文描述 | 当前 cuVS |
|---|---|---|
| initial kNN graph | GPU NN-Descent | NN-Descent / IVF-PQ / ACE / iterative |
| sort initial graph | GPU | GPU `sort_knn_graph` |
| pruning/reverse/merge | 论文实验说 optimization on CPU | 当前源码主路径 GPU kernels |
| compression | 论文初始主线无 VPQ build | 当前支持 VPQ-compressed dataset |
| connectivity | 论文主图未强调 | 当前有 `guarantee_connectivity` / MST logic |

---

### 3.3 Search algorithm

CAGRA search 的核心是 Fig.6 buffer model：

```text
internal top-M buffer:
  best M candidates so far

candidate list:
  children of selected parent nodes
  length = search_width * graph_degree
```

一轮 search：

```text
0. random sampling initialization
1. update internal top-M from whole buffer
2. select top-p unparented nodes as parents
3. fill candidate list with their children
4. compute distances only for first-seen nodes
5. repeat until convergence or max_iterations
6. output top-k from internal top-M
```

源码主路径：

- Single-CTA JIT：
  - `cpp/src/neighbors/detail/cagra/jit_lto_kernels/search_single_cta_jit.cuh::search_core`
  - helper：`device_common_jit.cuh`
- Multi-CTA JIT：
  - `cpp/src/neighbors/detail/cagra/jit_lto_kernels/search_multi_cta_jit.cuh::search_core`
- Search plan：
  - `cpp/src/neighbors/detail/cagra/search_plan.cuh`
- Factory：
  - `cpp/src/neighbors/detail/cagra/factory.cuh`

---

### 3.4 Single-CTA

#### 3.4.1 映射

```text
one query -> one CTA
blockIdx.y -> query_id
shared memory -> entire query state
```

Shared memory includes:

- `result_indices_buffer`
- `result_distances_buffer`
- `visited_hash_buffer`
- `parent_list_buffer`
- top-k workspace
- terminate flag
- dataset descriptor workspace

#### 3.4.2 瓶颈 -> 实现 -> 收益 -> 代价

| 项目 | 内容 |
|---|---|
| 瓶颈 | 如果每次 search 都有多个 kernels，kernel launch overhead和global memory state writeback高 |
| 实现 | `search_single_cta_jit.cuh::search_core()` 在一个 kernel loop 内完成 random sampling、top-M、parent pickup、distance、filter、output |
| 收益 | per-query state 保存在 shared/register，large batch下速度高 |
| 代价 | small batch CTA数量不足，shared memory/register pressure高 |

---

### 3.5 Multi-CTA

#### 3.5.1 映射

```text
one query -> multiple CTAs
each CTA:
  local internal top-32
  local candidate list length = graph_degree
  local search_width = 1
final:
  merge all CTA intermediate top-32 results
```

源码：

- `cpp/src/neighbors/detail/cagra/search_multi_cta.cuh::set_params`
- `cpp/src/neighbors/detail/cagra/jit_lto_kernels/search_multi_cta_jit.cuh::search_core`
- final merge：`_cuann_find_topk`

#### 3.5.2 参数

```text
num_cta_per_query = max(search_width, ceil(global_itopk_size / 32))
local itopk_size = 32
local search_width = 1
```

#### 3.5.3 瓶颈 -> 实现 -> 收益 -> 代价

| 项目 | 内容 |
|---|---|
| 瓶颈 | single query 或小 batch 下，一个 query一个CTA无法填满GPU，high recall需要更大搜索宽度 |
| 实现 | 多个CTA处理同一query，每个CTA维护local top-32，global traversed hash避免重复parent，最后top-k merge |
| 收益 | small batch / high recall场景更好 |
| 代价 | global hash atomicCAS contention，额外intermediate buffers和final merge |

---

### 3.6 Warp splitting / team size

#### 3.6.1 机制

文件/函数：

- `cpp/src/neighbors/detail/cagra/compute_distance.hpp`
- `cpp/src/neighbors/detail/cagra/jit_lto_kernels/device_common_jit.cuh::compute_distance_to_child_nodes_jit`
- generated：`compute_distance_standard_matrix.json`, `compute_distance_vpq_matrix.json`

典型逻辑：

```text
team_size_bits = descriptor.team_size_bitshift
team_id = threadIdx.x >> team_size_bits
lane_id = threadIdx.x & ((1 << team_size_bits) - 1)

each team computes one distance
team_sum reduces partial sums
```

#### 3.6.2 取值与 specialization

API 允许：

```text
team_size = 4, 8, 16, 32
0 = auto
```

当前 matrix confirmed：

```text
standard:
  dim 128 -> team 8
  dim 256 -> team 16
  dim 512 -> team 32
```

team=4 是否由 generic fallback 覆盖仍需 build directory确认。

#### 3.6.3 瓶颈 -> 实现 -> 收益 -> 代价

| 项目 | 内容 |
|---|---|
| 瓶颈 | 低维度用整warp算一个distance会浪费lanes |
| 实现 | 一个warp有最多个software teams，每team算一个candidate distance |
| 收益 | 低维度数据使lane utilization变好 |
| 代价 | team太小会增加 register pressure，高维度数据通常team 32更好 |

---

### 3.7 Forgettable hash

#### 3.7.1 机制

文件/函数：

- `cpp/src/neighbors/detail/cagra/hashmap.hpp`
- `cpp/src/neighbors/detail/cagra/search_plan.cuh::calc_hashmap_params`
- `search_single_cta_jit.cuh::search_core`
- `device_common_jit.cuh::hashmap_restore`

Hash implementation：

```text
open addressing
linear probing
atomicCAS
empty key = ~IdxT(0)
```

Forgettable mode：

```text
small hash in shared memory
periodically reset
after reset, restore internal top-M into hash
```

#### 3.7.2 瓶颈 -> 实现 -> 收益 -> 代价

| 项目 | 内容 |
|---|---|
| 瓶颈 | 完整 visited hash太大，放shared会占太多memory，放global太慢 |
| 实现 | small shared hash + reset interval，只保护internal top-M |
| 收益 | 高频visited check走shared memory，降低latency |
| 代价 | 被遗忘的节点可能重复计算distance，hash太小会collision/table-full |

---

### 3.8 Top-M / candidate buffer

#### 3.8.1 数据结构

Single-CTA shared memory：

```text
result_distances_buffer[0:M]
result_indices_buffer[0:M]
  -> internal top-M

result_distances_buffer[M:M+p*d]
result_indices_buffer[M:M+p*d]
  -> candidate list
```

Index 的 MSB 用作 parented flag。

#### 3.8.2 Top-M update

- bitonic path：
  - helpers：`topk_by_bitonic_sort_and_merge`
  - primitives：`bitonic::warp_sort`, `bitonic::warp_merge`
  - typical threshold：`p*d <= 256`
- radix path：
  - helper：`topk_by_radix_sort`
  - used when candidate list larger

#### 3.8.3 Candidate update

- `pickup_next_parents()` 从 internal top-M 中找未 parented nodes。
- parent position 写入 `parent_list_buffer`。
- `compute_distance_to_child_nodes_jit()` 展开 parent children 到 candidate list。

#### 3.8.4 瓶颈 -> 实现 -> 收益 -> 代价

| 项目 | 内容 |
|---|---|
| 瓶颈 | GPU上维护priority queue复杂不稳 |
| 实现 | internal top-M + fixed-length candidate list，配合bitonic/radix top-k |
| 收益 | 固定buffer，shared-memory友好，避免dynamic priority queue |
| 代价 | `M`, `search_width`, `graph_degree` 需要调，buffer大时shared memory和sort成本上升 |

---

### 3.9 CAGRA 所有优化点：瓶颈 -> 实现 -> 收益 -> 代价

| 优化点 | 瓶颈 | 实现 | 收益 | 代价 |
|---|---|---|---|---|
| Fixed out-degree graph | variable degree导致GPU load imbalance | final graph `N × graph_degree` | CTA工作量固定，candidate buffer固定 | 不自适应节点需求 |
| Directional + reverse edges | fixed out-degree下in-degree不固定，可达性不足 | prune graph后加reverse graph并merge | 改善reachability/SCC | reverse overflow被丢弃 |
| No hierarchy | hierarchy entry search sequential | random sampling并行算初始距离 | GPU-friendly initialization | 依赖random coverage |
| NN-Descent initial graph | 构初始kNN成本大 | `build_knn_graph` + NN-Descent/IVF-PQ fallback | 快速initial graph | 仍需显存/构图时间 |
| Rank-based pruning | distance-based pruning内存/计算大 | `kern_fused_prune`用rank/adjacency统计detour | 避免distance table/OOM | rank近似可能误判 |
| GPU graph optimization | CPU优化慢/搬运复杂 | current `prune_graph_gpu`, `make_reverse_graph_gpu`, `merge_graph_gpu` | 更现代GPU路径 | 与论文实验的不同 |
| Internal top-M buffer | priority queue不适合GPU | fixed shared buffers SoA | 高效top-k update | buffer参数敏感 |
| Candidate list `p*d` | 动态frontier复杂 | `search_width * graph_degree`固定长度 | 简单CUDA mapping | p/d大时distance/sort成本高 |
| Single-CTA one query/CTA | multi-kernel launch overhead | `search_core()` single kernel loop | large batch速度高 | small batch下不够SM |
| Multi-CTA one query/many CTAs | small batch underutilization | local top-32 + global traversed hash | single-query/high-recall更强 | global atomic contention |
| Warp splitting/team size | 低维整warp浪费 | team_size 4/8/16/32 API，matrix 8/16/32 buckets | lane utilization更好 | team太小register开销 |
| Shared forgettable hash | global visited慢，完整shared hash太大 | small hash reset + restore internal top-M | low latency visited | 重复distance |
| Open addressing hash | dynamic hash/queue不适合GPU | `hashmap.hpp` atomicCAS linear probing | fixed-size GPU-friendly | collision/probing |
| MSB parented flag | parented hash/bitmap额外开销 | index MSB标记已parented | O(1)状态，无额外结构 | uint32下限制到2^31-1 |
| Bitonic top-M | 小candidate list排序开销 | bitonic warp sort/merge | 小buffer快 | 超阈值无效 |
| Radix top-M | 大candidate bitonic成本高 | `topk_by_radix_sort` | 大buffer更稳 | workspace/复杂性 |
| Descriptor manual dispatch | virtual dispatch慢 | `compute_distance.hpp` descriptor/manual dispatch | 无pointer chasing/register overhead | generated code复杂 |
| FP16/int8/uint8 | HBM bandwidth瓶颈 | generated standard distance matrix | 节带宽/提高吞吐 | 精度/metric限制 |
| VPQ-half | dataset footprint太大 | `vpq_dataset<half>`, VPQ matrix | 降footprint | 论文初始主线外，FP32 VPQ不完整 |
| Persistent search | launch overhead/online latency | `search_params::persistent` | 服务化场景低延迟 | 与alloc/non-pinned copy交互弱 |
| AUTO algorithm selection | single/multi适用场景不同 | `search_plan.cuh` AUTO policy | 自动选path | 与论文阈值不完全一致 |
| Multi-kernel path | 特殊大top-k/filter场景 | global double buffer + host loop | 工程fallback | host sync/多kernel overhead |

---

### 3.10 CAGRA 代价与限制

1. **默认要求 GPU-resident graph/dataset**：超 HBM 时需要 VPQ/multi-GPU/ACE等工程。
2. **Index MSB flag 限制**：`uint32_t` 下可用 node id < `2^31`。
3. **Parameter sensitivity**：`graph_degree`, `itopk_size`, `search_width`, `team_size`, `hash_bitlen`, `max_iterations` 都影响 recall/QPS。
4. **Generated code complexity**：distance specialization由 CMake生成，非 build directory无法完全确认。
5. **single/multi CTA 选择敏感**：batch、recall、dataset hardness都会改变最优路径。
6. **forgettable hash 可能重复计算**：不会必然崩 recall，但可能有损耗。
7. **current source 与论文不完全一致**：构图、bitonic阈值、VPQ、persistent、multi-kernel都是需要区分的工程演进。

---

## 4. BANG vs CAGRA 对比矩阵

| 维度 | BANG | CAGRA | 结论 |
|---|---|---|---|
| 目标 | billion-scale single GPU容量突破 | GPU-resident高吞吐graph search | 默认资源假设不同 |
| 图来源 | 外部 Vamana/DiskANN | 自己构 CAGRA graph | BANG不构图，CAGRA构图+检索 |
| 图位置 | CPU host RAM | GPU memory | BANG绕过HBM限制，CAGRA避开CPU/PCIe |
| full vector位置 | host，candidate异H2D | GPU-resident strided/half/int8/VPQ | CAGRA更快但更占HBM |
| 压缩 | PQ compressed vectors主路径 | VPQ-half是current扩展 | BANG靠PQ解决容量 |
| 检索距离 | PQ asymmetric + exact rerank | exact/half/int8/VPQ descriptor distance | BANG有approx+rerank，CAGRA多为直接search distance |
| visited | global Bloom-style bool array | shared/global open-address hash | CAGRA更GPU-friendly |
| frontier state | worklist `BestLSet` | internal top-M + candidate list | 不同检索状态模型 |
| parent管理 | `d_parents`, `compute_parent1/2` | MSB parented flag + parent_list_buffer | CAGRA无额外parent hash |
| 并行粒度 | query/block，8 threads/neighbor，CPU OpenMP | query/CTA，multi-CTA/query，team/warp | CAGRA粒度更丰富 |
| kernel组织 | 多阶段多kernel | single/multi CTA single kernel loop，另有multi-kernel path | BANG为pipeline，CAGRA为shared-state |
| CPU介入 | 每轮介入 | search loop内不介入 | BANG是heterogeneous pipeline |
| PCIe | 每轮parent/children/vector copies | search loop基本无 PCIe | BANG多处PCIe瓶颈，CAGRA避开PCIe |
| shared memory | sort/merge/query/centroid | result buffer/hash/descriptor/topk | CAGRA更依赖shared |
| global memory | irregular compressed loads + Bloom random | dataset bandwidth + graph rows + hash | CAGRA coalescing通常更好 |
| sort/top-k | hand-written merge sort/merge | bitonic/radix/_cuann_find_topk/select_k | CAGRA top-k体系更完整 |
| rerank | 必须/需要 | 通常无BANG式rerank | BANG补偿PQ误差 |
| billion-scale | 单GPU可行但CPU/PCIe瓶颈 | 需HBM/压缩/多GPU | BANG设计更适合 |
| 100M | Base有额外CPU/PCIe成本 | GPU-resident优势明显 | CAGRA通常更强 |
| 当前风险 | DiskANN/PQ/host RAM/NUMA | version/config/generated code/AUTO | 都必须锁commit和配置 |

---

## 5. 为什么 BANG 适合 billion-scale single GPU

BANG 适合 1B single GPU 的原因不是 GPU kernel 本身比 CAGRA 更强，而是它方向了 memory residency 假设。

### 5.1 容量机制

BANG 避免把以下数据放入 GPU：

```text
full base vectors
Vamana/DiskANN graph
```

只放：

```text
PQ compressed vectors
per-query search buffers
PQDistTables
Bloom/worklist/rerank candidate buffers
```

### 5.2 传输机制

BANG 不做 shard swapping，不把大块 graph/data搬来搬去 GPU，而是每轮只传：

```text
GPU -> CPU:
  parent ids

CPU -> GPU:
  parent neighbors
  neighbor counts
  candidate full vectors for rerank
```

### 5.3 计算机制

search loop 中大多数 distance 用 PQ approximate distance：

```text
query-specific PQDistTable
compressed vector codes
lookup + sum
```

而不是 full-vector distance。

### 5.4 质量机制

因为 PQ search 有误差，BANG 用 exact reranking 补 recall：

```text
candidate full vectors fetched during search
compute_L2Dist<T>
compute_NearestNeighbours
```

### 5.5 代价

这种绕过 1B 的能力不是白费的：

- CPU fetch latency
- PCIe synchronization
- global Bloom random access
- compressed vector uncoalesced load
- rerank full-vector transfer
- batch size/memory tuning

因此 BANG 在 graph+dataset可GPU-resident的100M场景未必优。

---

## 6. 为什么 CAGRA 在 GPU-resident / 100M 场景更强

CAGRA 在 GPU-resident 场景更强，因为它从架构上消除了 BANG 的系统瓶颈：

```text
no CPU fetch per iteration
no parent/children PCIe loop
no PQ detour + exact rerank path
no global Bloom bool array as hot visited structure
```

### 6.1 Graph 适合 GPU

fixed out-degree 让：

```text
candidate list length = search_width * graph_degree
```

天然固定，CTA工作量更均衡。

### 6.2 Search state 适合 shared memory

single-CTA 把 internal top-M、candidate list、small hash、parent list都放在 shared memory，减少global访问。

### 6.3 Distance 适合 warp/team

低维用小 team，高维用大 team，减少 lane浪费和register开销。

### 6.4 Sort/top-k 更工程化

CAGRA使用 bitonic/radix/_cuann_find_topk/select_k 等路径，而不是手写的小list sort。

### 6.5 结果

当 100M graph+dataset可以放入GPU时，CAGRA避免了BANG CPU/PCIe/PQ/rerank的额外成本，通常吞吐更高。

---

## 7. 论文说法与源码实现差异

| 主题 | 论文说法 | 当前源码实现 | 修正结论 |
|---|---|---|---|
| CAGRA graph optimization位置 | initial kNN GPU，optimization CPU | 当前 `prune_graph_gpu`, `make_reverse_graph_gpu`, `merge_graph_gpu` 是GPU path | current cuVS不同于论文实验的 |
| CAGRA rank detour条件 | rank-based detour近似 distance detour | `kern_fused_prune` 中 `kDB < kAB` 条件被注释 | 源码判断更宽松 |
| CAGRA merge | pruned/reverse 各取 d/2 interleave | 保护到 d/2 pruned/MST edges，再插入 reverse edges | 不是严格子数 interleave |
| CAGRA Fig.6 init | random nodes放candidate，internal top-M dummy | single-CTA JIT初始化整个 `M+p*d` buffer | 图示与工程实现不同 |
| CAGRA bitonic阈值 | candidate <=512 bitonic | current常见阈值 `p*d <=256` | 工程调优后值 |
| CAGRA convergence | top-M indices不变 | 找不到未parented top-M entry + min/max iteration | 工程近似 |
| CAGRA search paths | 论文重点 single/multi CTA | current还有 `MULTI_KERNEL` | 必须区分 |
| CAGRA compression | 初始论文主线无PQ | current有 VPQ-half path | 工程扩展 |
| CAGRA distance implementation | 论文描述warp splitting | current由 generated matrix + descriptor dispatch实现 | 需build目录确认细节 |
| BANG README CLI | README给老旧命令 | `test_driver.cpp` 当前CLI不同 | 以源码为准 |
| BANG ablation宏 | 论文做ablation | 源码实际默认注释，Makefile传值未确认 | 需改源码确认build |
| BANG CPU threads | 论文实验机器固定 | `numCPUthreads`有64/TODO动态core | NUMA/CPU影响实测 |
| BANG kernel list | 论文主kernels | header有cachewarmup声明未确认 | 视为残留/未确认 |

---

## 8. 最容易被 GPT/Claude 漏掉的 20 个技术细节

1. **BANG 不构图**：它使用 Vamana/DiskANN graph，不从零构建 graph。
2. **BANG Base graph 不在 GPU**：CPU host `pIndex` 是 search loop 的一部分。
3. **BANG PQ compressed vectors 是主路径**：不是 optional compression trick。
4. **BANG PQ table transpose**：`pqTable_T` 改善 centroid access coalescing。
5. **BANG Bloom 是 global bool array，不是 shared hash，也不是 bit-packed bitmap。**
6. **BANG `compute_parent2` 在 sort/merge 前运行**：这是 eager prefetch，不是 Algorithm 2 直觉顺序。
7. **BANG exact rerank 依赖 search loop 中异步 H2D candidate full vectors。**
8. **BANG multi-kernel 是为了 CPU/GPU/PCIe pipeline，不是因为单kernel搞不定。**
9. **BANG `MAX_R`, `MAX_L`, `BF_ENTRIES` 是 compile-time 宏约束。**
10. **BANG header 有 `compute_neighborDist_par_cachewarmup` 声明但主路径未确认使用。**
11. **CAGRA fixed out-degree 是 GPU implementation-centric 设计，不只是图质量设计。**
12. **CAGRA graph construction 的 rank pruning 不算distance，但 initial sorting 算distance。**
13. **当前 CAGRA detour判断是论文Eq.3放宽，因为 `kDB < kAB` 被注释。**
14. **当前 CAGRA merge 不是严格 interleave，而是保护后d/2再插入reverse edges。**
15. **CAGRA Fig.6 的 buffer在源码里是 SoA arrays，不是 AoS pair。**
16. **当前 CAGRA random initialization 初始化整个 `M+p*d` buffer，而不是只填candidate list。**
17. **CAGRA MSB parented flag 会让 uint32 最大node id限到 `2^31-1`。**
18. **CAGRA bitonic/radix阈值论文是512，current源码常见是256。**
19. **CAGRA current source 还有 `MULTI_KERNEL` path，不只是 single/multi CTA。**
20. **CAGRA distance specialization 是 CMake generated matrix + descriptor dispatch，非 build directory不能确认vectorized load/accumulation。**

---

## 9. 如果重新实现一个系统，应如何借鉴两者

### 9.1 首先判断 memory residency

```text
if graph + dataset fit in GPU:
    prefer CAGRA-style GPU-resident architecture
else:
    prefer BANG-style host graph + GPU compressed vectors
```

不要在 graph/dataset 放不进GPU时硬套 CAGRA，也不要在100M GPU-resident时引入BANG的CPU/PCIe pipeline。

### 9.2 设计建议：混合系统

#### 层 1：数据驻留策略

| 情况 | 策略 |
|---|---|
| dataset/full graph fit HBM | CAGRA-style exact/half distance |
| dataset too large but compressed vectors fit | BANG-style PQ/VPQ compressed vectors |
| graph too large | host graph / graph paging / compressed graph |
| multi-GPU | graph/data partition + CAGRA-style local search + inter-GPU merge |

#### 层 2：Graph design

借鉴 CAGRA：

- fixed out-degree for GPU search
- reverse edge addition for reachability
- rank-based pruning to reduce memory
- no hierarchy if GPU random sampling足够

借鉴 BANG：

- 支持外部 Vamana/DiskANN graph，方便复用已有 index。
- 对超大图保留host-resident graph path。

#### 层 3：Search state

推荐 CAGRA-style：

```text
internal top-M buffer
candidate list p*d
SoA distance/index
MSB parented flag
```

如果 host graph path，则借鉴 BANG：

```text
worklist + next parent prefetch
CPU fetch overlap
candidate full-vector prefetch
```

#### 层 4：Visited set

推荐：

- GPU-resident：CAGRA shared/forgettable hash。
- Host graph / huge batch：BANG Bloom思想可用，但应改为更高效 bitset/hash hybrid。
- 多CTA：local shared hash + global traversed hash。

#### 层 5：Distance

推荐：

- exact/half GPU-resident：CAGRA team-size descriptor。
- compressed billion-scale：BANG PQDistTable + better vector layout。
- VPQ：CAGRA current VPQ-half path值得借鉴。
- rerank：如果 search distance approximate，必须支持 exact rerank。

#### 层 6：Pipeline

若 host graph：

- BANG-style streams：
  - parent D2H
  - children H2D
  - candidate vector H2D
  - kernels
- eager parent prefetch 是关键。
- 需要 Nsys 验证 overlap。

若 GPU-resident：

- CAGRA-style single-kernel loop。
- single/multi CTA auto policy。
- persistent kernel可用于服务化场景。

### 9.3 不建议直接抄的地方

| 不建议抄 | 原因 | 替代 |
|---|---|---|
| BANG global bool Bloom | random global access差 | shared hash / bit-packed Bloom / cache-aware visited |
| BANG fixed macros | 泛化差 | runtime templates + autotune |
| BANG one thread/query parent selection | underutilization | warp-level parent scan |
| CAGRA MSB flag | 限制 index range | separate bitset if >2^31 nodes |
| CAGRA current generated code复杂度 | 维护难 | 明确generated artifact和fallback |
| CAGRA AUTO固定阈值 | 硬件/数据变化大 | runtime calibration / profiling-based tuning |

---

## 10. 附录

### 10.1 函数/文件索引

#### BANG Base

| 文件 | 关键函数/对象 |
|---|---|
| `BANG_Base/bang.h` | `BANGSearch<T>` |
| `BANG_Base/bang_search.cuh` | `GraphMedataData`, `IndexLoad`, `GPUInstance`, `HostInstance`, `SearchParams`, kernel declarations |
| `BANG_Base/bang_search.cu` | `BANGSearchInner<T>::bang_load`, `bang_alloc`, `bang_init`, `bang_query` |
| `BANG_Base/bang_search.cu` | `populate_pqDist_par<T>` |
| `BANG_Base/bang_search.cu` | `neighbor_filtering_new`, `hashFn1_d`, `hashFn2_d` |
| `BANG_Base/bang_search.cu` | `compute_neighborDist_par` |
| `BANG_Base/bang_search.cu` | `compute_parent1`, `compute_parent2` |
| `BANG_Base/bang_search.cu` | `compute_BestLSets_par_sort_msort`, `compute_BestLSets_par_merge` |
| `BANG_Base/bang_search.cu` | `compute_L2Dist<T>`, `compute_NearestNeighbours` |
| `BANG_Base/test_driver.cpp` | `main`, `run_anns`, `calculate_recall` |
| `BANG_Inmemory/parANN.cu/.h` | In-memory variant |
| `BANG_Exactdistance/parANN.cu/.h` | Exact-distance variant |

#### CAGRA / cuVS

| 文件 | 关键函数/对象 |
|---|---|
| `cpp/include/cuvs/neighbors/cagra.hpp` | `index_params`, `search_params`, `index<T,IdxT>` |
| `cpp/src/neighbors/cagra.cuh` | public API detail build/search wrappers |
| `cpp/src/neighbors/detail/cagra/cagra_build.cuh` | `detail::build`, `build_knn_graph`, `iterative_build_graph`, `build_ace` |
| `cpp/src/neighbors/detail/cagra/graph_core.cuh` | `sort_knn_graph`, `kern_sort`, `prune_graph_gpu`, `kern_fused_prune`, `make_reverse_graph_gpu`, `kern_make_rev_graph_k`, `merge_graph_gpu`, `kern_merge_graph` |
| `cpp/src/neighbors/detail/cagra/cagra_search.cuh` | `search_main`, `search_main_core` |
| `cpp/src/neighbors/detail/cagra/search_plan.cuh` | `search_plan_impl_base`, `adjust_search_params`, `calc_hashmap_params` |
| `cpp/src/neighbors/detail/cagra/factory.cuh` | `factory::create` |
| `cpp/src/neighbors/detail/cagra/search_single_cta.cuh` | single-CTA launcher |
| `cpp/src/neighbors/detail/cagra/jit_lto_kernels/search_single_cta_jit.cuh` | `search_core` single-CTA |
| `cpp/src/neighbors/detail/cagra/search_multi_cta.cuh` | multi-CTA launcher, `set_params` |
| `cpp/src/neighbors/detail/cagra/jit_lto_kernels/search_multi_cta_jit.cuh` | `search_core` multi-CTA |
| `cpp/src/neighbors/detail/cagra/search_multi_kernel.cuh` | multi-kernel path |
| `cpp/src/neighbors/detail/cagra/jit_lto_kernels/device_common_jit.cuh` | `compute_distance_to_random_nodes_jit`, `compute_distance_to_child_nodes_jit` |
| `cpp/src/neighbors/detail/cagra/hashmap.hpp` | open-address hash |
| `cpp/src/neighbors/detail/cagra/compute_distance.hpp` | dataset descriptor, distance dispatch |
| `cpp/src/neighbors/detail/cagra/bitonic.hpp` | bitonic helpers |
| `cpp/src/neighbors/detail/cagra/topk_by_radix.cuh` | radix top-k |
| `cpp/src/neighbors/detail/cagra/compute_distance_standard_matrix.json` | standard distance specialization matrix |
| `cpp/src/neighbors/detail/cagra/compute_distance_vpq_matrix.json` | VPQ specialization matrix |
| `python/cuvs_bench/cuvs_bench/config/algos/cuvs_cagra.yaml` | benchmark config |
| `python/cuvs_bench/cuvs_bench/config/algos/faiss_gpu_cagra.yaml` | FAISS CAGRA wrapper config |

---

### 10.2 参数索引

#### BANG

| 参数/宏 | 位置 | 含义 |
|---|---|---|
| `MAX_R=64` | `BANG_Base/bang_search.cu` | max graph degree |
| `MAX_L` | `BANG_Base/bang_search.cu` / headers | max worklist length |
| `BF_ENTRIES=399887` | `BANG_Base/bang_search.cu` | Bloom entries per query |
| `BF_MEMORY` | `BANG_Base/bang_search.cu` | Bloom memory aligned |
| `MAX_PARENTS_PERQUERY` | `BANG_Base/bang_search.cu` | rerank candidate upper bound |
| `THREADS_PER_NEIGHBOR=8` | `compute_neighborDist_par` | PQ distance team |
| `numThreads_K1=256` | `bang_init()` | PQDistTable kernel block size |
| `numThreads_K2=512` | `bang_init()` | PQ distance kernel block size |
| `numThreads_K3=R+1` | `bang_init()` | sort kernel block size |
| `numThreads_K3_merge=2L` | `bang_init()` | merge kernel block size |
| `numThreads_K5=256` | `bang_init()` | Bloom filter kernel |
| `_TIMERS` | source macro | timing instrumentation |
| `_NO_PRETECH` | source macro | disable eager prefetch path |
| `_NO_ASYNC_FP` | source macro | disable async full-vector transfer |

#### CAGRA

| 参数 | 位置 | 含义 |
|---|---|---|
| `graph_degree` | `index_params` | final graph out-degree |
| `intermediate_graph_degree` | `index_params` | initial kNN degree |
| `graph_build_params` | `index_params` | NN-Descent/IVF-PQ/ACE/iterative |
| `guarantee_connectivity` | `index_params` | MST connectivity |
| `attach_dataset_on_build` | `index_params` | attach dataset to index |
| `compression` | `index_params` | VPQ params |
| `itopk_size` | `search_params` | internal top-M length |
| `search_width` | `search_params` | parent count p |
| `team_size` | `search_params` | warp splitting team |
| `max_iterations` | `search_params` | hard stop |
| `min_iterations` | `search_params` | minimum loop count |
| `algo` | `search_params` | AUTO/SINGLE_CTA/MULTI_CTA/MULTI_KERNEL |
| `hashmap_mode` | `search_params` | HASH/SMALL/AUTO |
| `hashmap_min_bitlen` | `search_params` | min hash size |
| `hashmap_max_fill_rate` | `search_params` | max hash fill |
| `num_random_samplings` | `search_params` | random init count |
| `rand_xor_mask` | `search_params` | random seed variation |
| `persistent` | `search_params` | persistent kernel |
| `persistent_lifetime` | `search_params` | persistent lifetime |
| `persistent_device_usage` | `search_params` | device fraction used |

---

### 10.3 Kernel 索引

#### BANG

| Kernel | 文件/函数 | 分类 |
|---|---|---|
| `populate_pqDist_par<T>` | `BANG_Base/bang_search.cu` | PQ table |
| `neighbor_filtering_new` | `BANG_Base/bang_search.cu` | visited filter |
| `compute_neighborDist_par` | `BANG_Base/bang_search.cu` | PQ distance |
| `compute_BestLSets_par_sort_msort` | `BANG_Base/bang_search.cu` | sort |
| `compute_BestLSets_par_merge` | `BANG_Base/bang_search.cu` | worklist merge |
| `compute_parent1` | `BANG_Base/bang_search.cu` | initial parent |
| `compute_parent2` | `BANG_Base/bang_search.cu` | eager parent |
| `compute_L2Dist<T>` | `BANG_Base/bang_search.cu` | exact rerank |
| `compute_NearestNeighbours` | `BANG_Base/bang_search.cu` | rerank top-k |
| `compute_neighborDist_par_cachewarmup` | `BANG_Base/bang_search.cuh` declaration | unconfirmed/residual |

#### CAGRA

| Kernel/helper | 文件/函数 | 分类 |
|---|---|---|
| `kern_sort` | `graph_core.cuh` | graph initial sort |
| `kern_fused_prune` | `graph_core.cuh` | rank pruning |
| `kern_make_rev_graph_k` | `graph_core.cuh` | reverse graph |
| `kern_merge_graph` | `graph_core.cuh` | graph merge |
| `search_core` single-CTA | `search_single_cta_jit.cuh` | search loop |
| `search_core` multi-CTA | `search_multi_cta_jit.cuh` | search loop |
| `compute_distance_to_random_nodes_jit` | `device_common_jit.cuh` | random sampling |
| `compute_distance_to_child_nodes_jit` | `device_common_jit.cuh` | child distance |
| `pickup_next_parents` | single-CTA helper | parent selection |
| `pickup_next_parent` | multi-CTA helper | parent selection |
| `topk_by_bitonic_sort_and_merge` | bitonic helpers | top-M update |
| `topk_by_radix_sort` | radix top-k | top-M update |
| `_cuann_find_topk` | multi-CTA/multi-kernel | final top-k |
| `select_k` | multi-kernel fallback | topK >1024 |

---

### 10.4 Profiler 指标索引

#### Nsight Systems

| 系统 | 看什么 |
|---|---|
| BANG | CPU OpenMP fetch 与 GPU sort/merge overlap |
| BANG | `streamParent`, `streamChildren`, `streamFPTransfers`, `streamKernels` 是否并行 |
| BANG | parent D2H / children H2D / FP H2D timeline |
| BANG | kernel sequence: PQ table -> filter -> distance -> parent -> sort -> merge -> rerank |
| CAGRA | single-CTA/multi-CTA 是否是长kernel loop |
| CAGRA | batch=1是否启multi-CTA |
| CAGRA | multi-kernel path是否出现host terminate flag sync |
| CAGRA | build/search分离 |
| CAGRA | final `_cuann_find_topk` merge |

#### Nsight Compute

| 指标 | 用途 |
|---|---|
| `dram__throughput.avg.pct_of_peak_sustained_elapsed` | memory bandwidth |
| `derived__memory_l2_theoretical_sectors_global_excessive` | uncoalesced / excessive L2 sectors |
| `smsp__stall_long_scoreboard` | memory dependency stalls |
| `smsp__stall_barrier` | barrier stalls |
| `sm__warps_active.avg.pct_of_peak_sustained_active` | achieved occupancy |
| `smsp__warps_eligible.avg` | scheduler pressure |
| `smsp__sass_average_branch_targets_threads_uniform.pct` | branch uniformity |
| `l1tex__t_set_accesses_pipe_lsu_mem_global_op_atom.sum` | atomics |
| shared bank conflict metrics | shared memory conflict |
| source counters on distance load | vector load efficiency |
| source counters on hash insert | hash collision/atomic behavior |

---

### 10.5 未确认项

| 未确认项 | 需要什么 |
|---|---|
| BANG `compute_neighborDist_par_cachewarmup` 是否定义/使用 | 本地完整 grep / 编译链接 |
| BANG Makefile/build flags | 本地 repo tree，确认 NVCC arch、宏传递 |
| BANG In-memory/Exact-distance 行为差异 | 继续审 `BANG_Inmemory`, `BANG_Exactdistance` |
| BANG `BANGSearch<T>` 构造类型安全 | 编译/运行不同 dtype，检查 UB |
| CAGRA generated `compute_distance-ext.cuh` | 本地 build directory |
| CAGRA generated distance `.cu` | build output |
| team_size=4 是否有 generic fallback | build artifacts / runtime dispatch |
| CAGRA persistent kernel实现 | 审 launcher JIT persistent path |
| CAGRA multi-kernel何时被AUTO选择 | 完整 search plan + configs |
| bitonic threshold 256 原因 | commit history / microbenchmark |
| CAGRA current source vs paper artifact | 找论文 artifact commit/tag |
| 所有性能结论 | Nsys + NCU 实测 |
