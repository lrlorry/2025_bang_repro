# BANG vs CAGRA 深度对比

> 基于静态源码审计，2026-05-24。  
> 详细依据：`../bang_cagra_final_technical_report.md`。  
> BANG 源码：`../BANG-Billion-Scale-ANN/BANG_Base/`（Jul 2025 commit）。  
> CAGRA 源码：`../2024_cagra_repro/`（cuVS main branch）。

---

## 1. 一句话定位

**BANG** 是容量突破型系统：把 Vamana/DiskANN graph + full vectors 放 CPU host RAM，只把 PQ compressed vectors 放 GPU，通过 CPU-GPU pipeline 处理 graph/data 无法放入 HBM 的 billion-scale 场景。

**CAGRA** 是 GPU-resident 高吞吐系统：自己构 fixed out-degree directional graph，把 graph/dataset/search state 全放 GPU，通过 single-CTA/multi-CTA、warp splitting、shared-memory hash、bitonic/radix top-k 最大化 GPU 利用率。

两者的**根本区别是 memory residency 假设**，不是"谁的 kernel 更快"。

---

## 2. 对比矩阵

| 维度 | BANG | CAGRA |
|---|---|---|
| 目标 | billion-scale single GPU 容量突破 | GPU-resident 高吞吐 graph search |
| 图来源 | 外部 Vamana/DiskANN | 自己构 CAGRA graph |
| 图位置 | CPU host RAM（`pIndex`）| GPU HBM |
| full vector 位置 | host RAM，candidate 异步 H2D | GPU-resident（strided/half/int8/VPQ）|
| 压缩 | PQ compressed vectors 为主路径 | VPQ-half 是 current 扩展（论文主线无）|
| 检索 distance | PQ asymmetric + exact rerank | exact/half/int8/VPQ descriptor distance |
| visited set | global bool array（399887/query，两 hash 函数）| shared/global open-addressing atomicCAS hash |
| frontier state | worklist `d_BestLSets` + dist + visited | internal top-M buffer + candidate list（SoA）|
| parent 管理 | `d_parents` D2H，`compute_parent1/2` | MSB parented flag + `parent_list_buffer` |
| 并行粒度 | query/block，8 threads/neighbor，CPU OpenMP | query/CTA，multi-CTA/query，warp team 4/8/16/32 |
| kernel 组织 | 多阶段多 kernel（pipeline 模型）| single-CTA single kernel loop（另有 multi-CTA/multi-kernel）|
| CPU 介入 | 每轮 D2H parent → CPU fetch → H2D | search loop 内不介入 |
| PCIe | 每轮 parent D2H + children H2D + FP H2D | search loop 基本无 PCIe |
| sort/top-k | hand-written merge sort/merge | bitonic/radix/`_cuann_find_topk`/`select_k` |
| rerank | 必须（补偿 PQ 误差）| 通常无 BANG 式 rerank |
| billion-scale | 单 GPU 可行，但 CPU/PCIe 为瓶颈 | 需 HBM/压缩/多 GPU |
| 100M 场景 | Base 有额外 CPU/PCIe 成本 | GPU-resident 优势明显 |

---

## 3. 为什么 BANG 适合 billion-scale single GPU

### 3.1 容量：避免 HBM 超限

BANG 不把以下内容放 GPU：
- full base vectors（SIFT1B = ~128 GB）
- Vamana/DiskANN graph（SIFT1B = ~640 GB）

只放 GPU HBM：
- PQ compressed vectors（`N × uChunks` bytes，100 GB 以内可控）
- per-query search buffers（PQ dist tables、worklist、Bloom filter、rerank 候选）

### 3.2 传输：每轮只传 frontier

每个 search 迭代：
```
GPU → CPU:  parent ids（几十 KB）
CPU → GPU:  parent neighbors + counts（几百 KB）
CPU → GPU:  candidate full vectors for rerank（异步，streamFPTransfers）
```
不传整个 graph，不传完整 dataset。

### 3.3 近似 + rerank：补偿 PQ 误差

搜索阶段用 PQ asymmetric distance（快但有误差），最终用 exact rerank 补回 recall：
```
populate_pqDist_par → per-query PQ distance table
compute_neighborDist_par → 查表 + sum = PQ distance
compute_L2Dist<T> + compute_NearestNeighbours → exact rerank top-k
```

### 3.4 代价

- CPU fetch 在 critical path（每轮 barrier）
- PCIe 同步点多（5 个 `cudaStreamSynchronize`）
- global Bloom random access（`d_processed_bit_vec` 400K bool/query）
- compressed vector load 不规则（graph neighbor id 随机）
- 大 batch 才能填满 GPU

---

## 4. 为什么 CAGRA 在 100M GPU-resident 场景更强

CAGRA 从架构上消除了 BANG 的系统瓶颈：
- 无 CPU fetch（no PCIe loop）
- 无 PQ 误差 + exact rerank 路径
- 无 global Bloom bool array 随机访问
- search state 在 shared memory，global memory 访问更 coalesced

关键机制：
- fixed out-degree → candidate list 长度固定 = `search_width × graph_degree`
- single-CTA single kernel loop → per-query state 全在 shared/register，large batch QPS 高
- team-size dispatch → 低维 dataset 减少 lane 浪费
- bitonic/radix top-k → 工程化，不是手写

---

## 5. 关键设计点的源码对照

### 5.1 visited set

| | BANG | CAGRA |
|---|---|---|
| 位置 | global memory | shared memory（forgettable hash）|
| 结构 | `d_processed_bit_vec`: `bool[399887]` per query | `visited_hash_buffer`: open-addressing `uint32` hash |
| hash | 2 × FNV-1a style，写 global bool | atomicCAS linear probing |
| sync | 无 `__syncthreads()`（设计如此，接受 race）| warp-level implicit |
| 代价 | random global access，false positive | shared hash 容量小，reset + restore |

源码位置：
- BANG：`bang_search.cu::neighbor_filtering_new`，`bang_search.cu::hashFn1_d/hashFn2_d`
- CAGRA：`hashmap.hpp`，`search_single_cta_jit.cuh::search_core`

### 5.2 top-k / sort

| | BANG | CAGRA |
|---|---|---|
| 新邻居排序 | `compute_BestLSets_par_sort_msort`（手写）| `topk_by_bitonic_sort_and_merge`（bitonic warp sort）|
| worklist 更新 | `compute_BestLSets_par_merge`（手写 lower/upper bound）| `topk_by_radix_sort` 或 bitonic，取决于 candidate size |
| 限制 | `2*L ≤ 1024`（merge kernel 线程数上限）| bitonic 阈值 `p*d ≤ 256`，radix 做 fallback |

### 5.3 parent 选择

| | BANG | CAGRA |
|---|---|---|
| 机制 | `compute_parent2`（单线程/query，提前于 sort/merge）| MSB flag 标记已 parented node，`pickup_next_parents()` |
| eager prefetch | `compute_parent2` 在 sort/merge 前跑，让 CPU 尽早开始 fetch | 无 CPU fetch，不需要 prefetch 机制 |
| GPU 利用率 | 低（单线程/query）| 较好（warp 级 parent scan）|

### 5.4 distance 计算

| | BANG | CAGRA |
|---|---|---|
| 压缩 | PQ code（uint8 per chunk）| strided float/half/int8/VPQ |
| distance 方法 | PQ lookup + sum（`compute_neighborDist_par`，8 threads/neighbor）| exact（team-size warp splitting，4/8/16/32 threads/distance）|
| coalescing | 差（neighbor id 随机，compressed vector access 跳跃）| 较好（dataset 连续存储，graph row 对齐）|
| rerank | 必须（PQ 误差）| 通常不需要 |

---

## 6. 场景选择指引

| 场景 | 推荐 | 理由 |
|---|---|---|
| 数据集 1B，单 GPU，host RAM ≥ 640 GB | **BANG** | HBM 放不下 graph，BANG 是唯一可行方案 |
| 数据集 100M，A100 80GB 放得下 | **CAGRA** | GPU-resident，无 CPU/PCIe 瓶颈 |
| 数据集 100M，GPU HBM 不足 | BANG（有损）或 CAGRA + VPQ | 取决于精度要求 |
| 低延迟在线 serving，< 1ms | CAGRA persistent kernel 或 CPU DiskANN | BANG 的 PCIe sync 太多 |
| 高吞吐离线 batch | BANG（1B）或 CAGRA（100M 以下）| 取决于数据规模 |

---

## 7. 论文说法 vs 当前源码差异

| 主题 | 论文说法 | 当前源码 | 修正结论 |
|---|---|---|---|
| BANG CLI 格式 | README.md 旧格式（分别传文件路径）| `test_driver.cpp` 新格式（index prefix + 7 参数）| 以 ReadMe.pdf 为准 |
| BANG ablation 宏 | Table 5 ablation 有结果 | `_TIMERS/_NO_PRETECH/_NO_ASYNC_FP` 默认注释 | 需改源码重编 |
| BANG CPU threads | 论文实验机器核数未说明 | `numCPUthreads=64` 硬编码（cu:413 有 TODO）| NUMA 影响实测，AutoDL 需 `nproc` 确认 |
| BANG `BANGSearch<T>` | 支持多种 dtype | 始终 `new BANGSearchInner<int>()`（cu:73）| uint8 路径存在 LOAD-01 类型安全缺陷 |
| CAGRA graph optimization | 论文描述 optimization 在 CPU | 当前 `prune_graph_gpu/make_reverse_graph_gpu/merge_graph_gpu` 是 GPU kernels | current cuVS 与论文实验不同 |
| CAGRA rank detour 条件 | 严格 `rank(D,B) < rank(X,B)` | `kern_fused_prune` 中 `kDB < kAB` 被注释 | 判断更宽松 |
| CAGRA merge interleave | pruned/reverse 各 d/2 | 保护 d/2 pruned/MST 后插入 reverse | 不是严格 interleave |
| CAGRA bitonic 阈值 | 论文 ≤ 512 | 当前常见 ≤ 256 | 工程调优值 |

---

## 8. 容易漏掉的 20 个技术细节

（来自 `bang_cagra_final_technical_report.md` 第 8 节）

1. BANG 不构图：使用外部 DiskANN graph，不从零构建。
2. BANG Base graph 不在 GPU：`pIndex` 在 host，是 search loop 的一部分。
3. BANG PQ compressed vectors 是主路径，不是 optional trick。
4. BANG `pqTable_T` 转置改善 centroid access coalescing（`populate_pqDist_par`）。
5. BANG Bloom 是 global bool array，不是 shared hash，也不是 bit-packed bitset。
6. BANG `compute_parent2` 在 sort/merge **前**运行，是 eager prefetch（非 Algorithm 2 直觉顺序）。
7. BANG exact rerank 依赖 search loop 中异步 H2D 的 candidate full vectors（`streamFPTransfers`）。
8. BANG 多 kernel 是为了 CPU/GPU/PCIe pipeline，不是因为单 kernel 搞不定。
9. BANG `MAX_R=64`, `MAX_L`, `BF_ENTRIES=399887` 是 compile-time 宏约束。
10. BANG header 有 `compute_neighborDist_par_cachewarmup` 声明但主路径未确认使用。
11. CAGRA fixed out-degree 是 GPU implementation-centric 设计，不只是图质量设计。
12. CAGRA rank pruning 不算 distance，但 initial sorting 算 distance。
13. 当前 CAGRA detour 判断是论文 Eq.3 放宽（`kDB < kAB` 被注释）。
14. 当前 CAGRA merge 不是严格 interleave，而是保护后 d/2 再插入 reverse edges。
15. CAGRA Fig.6 buffer 在源码里是 SoA arrays，不是 AoS pair。
16. 当前 CAGRA random init 初始化整个 `M+p*d` buffer，不只是 candidate list。
17. CAGRA MSB parented flag 将 uint32 node id 上限限到 `2^31-1`。
18. CAGRA bitonic/radix 阈值：论文 512，current 常见 256。
19. CAGRA current source 还有 `MULTI_KERNEL` path（不只是 single/multi CTA）。
20. CAGRA distance specialization 是 CMake generated matrix + descriptor dispatch，非 build directory 不能确认 vectorized load/accumulation 细节。
