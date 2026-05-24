# BANG Base 复现测试设计

> **源码基准**：`../BANG-Billion-Scale-ANN/BANG_Base/`，2026-05-24。  
> **说明**：这是测试设计文档，不是执行记录。标注 `needs_gpu` 的测试需要 NVIDIA GPU 才能运行。当前无 GPU 环境，未执行任何 GPU 测试。如在 GPU 机器上运行，请将结果记录到 `../results/` 并更新本文档。  
> **对应 findings**：参见 `bang_audit_report.md`。

---

## 测试 T-01：SIFT10K Smoke Test

**目的**：验证预构建二进制可以加载 index、运行搜索、输出合理的 QPS 和 Recall@10。  
**对应 finding**：CLI-01，PREPROC-01。  
**需要**：`needs_gpu`，SIFT10K 预构建文件（已有）。

**运行方式**：
```bash
bash scripts/run_sift10k_smoke.sh
```

**预期输出**（来自 `test_driver.cpp:402-403,526`）：
```
L       Time    QPS             10-r@10
--      ----    ---             ------
10      xx.x    xxxx.x          xx.xx
22      xx.x    xxxx.x          xx.xx
...
```

**通过条件**：
- 程序正常启动并输出表头
- 至少一行 L/QPS/Recall 数据
- Recall@10 > 0（非零即通过基础正确性）
- 没有 CUDA 错误信息

**当前状态**：SKIP（无 GPU）。原因已记录到 `results/sift10k_smoke.log`（由 `run_sift10k_smoke.sh` 写入）。

---

## 测试 T-02：PQ 文件加载验证

**目的**：确认 `bang_load()` 能正确解析 `_pq_pivots.bin` 的三节结构（pivots + centroid + chunk_offsets），uChunks 值合理。  
**对应 finding**：PREPROC-01。  
**需要**：`needs_gpu`（`bang_load` 内部有 `cudaMalloc/cudaMemcpy`）。

**方法**：
1. 在 `bang_load()` 完成后打印 `m_objInputData.uChunks`、`m_objInputData.D`、`m_objInputData.N`。
2. 验证：`uChunks == PQ compressed file 大小 / (N * 1)`。

**SIFT10K 预期值**（从文件大小反推）：
```
sift10k_index_pq_compressed.bin = 1.2 MB = 1,258,xxx bytes
N = 10,000
→ uChunks = 1,258,xxx / 10,000 ≈ 125-127
（sift10kfiles 中 uChunks 实际值需运行确认）
```

**当前状态**：needs_test。

---

## 测试 T-03：Host Graph 不在 GPU

**目的**：从源码层面确认 `pIndex`（graph + full vectors）只在 host RAM，没有对应的 `d_pIndex` 被 `cudaMalloc`。  
**对应 finding**：LOAD-01 相关。  
**需要**：静态审计（已完成）。

**方法**：grep `cudaMalloc` in `bang_search.cu`，确认无 `pIndex` 相关的 GPU 分配。

```bash
grep -n "cudaMalloc" ../BANG-Billion-Scale-ANN/BANG_Base/bang_search.cu | grep -v "d_compress\|d_pqTable\|d_chunk\|d_centroid\|d_queries\|d_nearest\|d_pqDist\|d_BestL\|d_parent\|d_neighbor\|d_processed\|d_nextIter\|d_iter\|d_mark\|d_FPSet\|d_L2"
```

**预期**：无额外 cudaMalloc（`pIndex` 只用 `malloc`）。

**已验证（静态）**：`bang_search.cu:313` 使用 `malloc(m_objInputData.size_indexfile)` 分配 host RAM，无对应 GPU 分配。

---

## 测试 T-04：CPU-GPU Transfer 重叠验证

**目的**：确认 4 streams 实现了 parent D2H + CPU fetch + children H2D + FP H2D 的重叠。  
**对应 finding**：STREAM-01。  
**需要**：`needs_gpu` + Nsight Systems。

**方法**：
1. 启用 `_TIMERS` 宏重新编译。
2. 用 `nsys profile ./bang_search ./sift10k_index ...` 生成 nsys-rep。
3. 在 Nsight Systems Timeline 中检查：
   - `streamParent` D2H 与 `streamKernels` sort/merge 是否时间重叠
   - `streamFPTransfers` H2D 是否与后续 filter/distance 重叠
   - `cudaStreamSynchronize(streamChildren)` 是否是瓶颈

**关键 timeline 标志**：
- `compute_BestLSets_par_sort_msort` 与 `cudaMemcpyAsync parent D2H` overlap
- `cudaMemcpyAsync FPSet H2D` 贯穿多个 iterations

**当前状态**：needs_gpu。

---

## 测试 T-05：Worklist Length Sweep

**目的**：复现论文 Figure 6/7 的 worklist length vs recall/QPS 曲线。  
**对应 finding**：QUERY-01，KERNEL-03。  
**需要**：`needs_gpu`，SIFT1M 数据 + DiskANN index。

**方法**：
```bash
./bang_search sift1m_index ./sift1m_query.bin ./sift1m_groundtruth.bin 10000 10 float l2
# auto 模式：自动 sweep L=10, 22, 34, ..., MAX_L（step=12）
```

**预期**：
- L 增大：recall 提升，QPS 下降
- recall 在某个 L 值饱和
- 结果写入 `results/sift1m_sweep.csv`

**2*L ≤ 1024 约束**（`bang_search.cu:439`）：
- MAX_L = (1024/2) = 512（上界）
- 实际 MAX_L 定义需查看源码 header

**当前状态**：needs_gpu + needs DiskANN build。

---

## 测试 T-06：Exact Rerank 正确性

**目的**：验证 `compute_L2Dist + compute_NearestNeighbours` 输出的 recall 高于纯 PQ search。  
**对应 finding**：RERANK-01。  
**需要**：`needs_gpu`。

**方法**：
1. 修改 `bang_query()` 添加一个版本：rerank 前直接从 worklist 输出 top-k（不做 exact rerank）。
2. 比较两个版本在相同 L 下的 recall。

**预期**：exact rerank 版本 recall 明显高于纯 PQ 版本（论文 Table 5 ablation 显示 rerank 对 recall 贡献显著）。

**当前状态**：needs_gpu + needs code modification。

---

## 测试 T-07：Bloom Visited Filter Ablation

**目的**：验证 Bloom filter 对 recall 和 QPS 的影响（对应论文 ablation）。  
**对应 finding**：QUERY-02，QUERY-03。  
**需要**：`needs_gpu`。

**方法**：注释掉 `neighbor_filtering_new` kernel 调用，改为直接传递所有 neighbors，观察 recall 和 QPS 变化。

**预期**（来自论文）：
- 禁用 Bloom filter → recall 最多下降 10×（SIFT-1B 场景）
- 小规模（SIFT10K）影响可能较小

**当前状态**：needs_gpu + needs code modification。  
**TODO**：确认 `neighbor_filtering_new` 是否有独立宏控制路径。

---

## 测试 T-08：Eager Prefetch Ablation

**目的**：量化 `compute_parent2` 提前执行（eager prefetch）对 QPS 的贡献。  
**对应 finding**：QUERY-01。  
**需要**：`needs_gpu`。

**方法**：分别编译 `_NO_PRETECH` 注释（默认）和 `_NO_PRETECH` 开启两个版本，在相同 L、相同数据集上比较 QPS。

```bash
# 版本 1：默认（eager prefetch 开启）
# 版本 2：-D_NO_PRETECH（禁用 eager prefetch）
```

**预期**（来自论文 Figure 8 / Section 4.6）：eager prefetch 贡献约 8~12% QPS 提升。

**当前状态**：needs_gpu + needs recompile with macro。

---

## 测试 T-09：BANGSearch<uint8_t> 类型安全

**目的**：验证 `BANGSearch<uint8_t>` 是否因 `new BANGSearchInner<int>()` 导致错误。  
**对应 finding**：LOAD-01（高优先级）。  
**需要**：`needs_gpu`，bigann uint8 数据集。

**方法**：
1. 用 SIFT10K float 数据：正常运行，基准。
2. 用 bigann uint8 数据（若有）：检查 recall 是否合理、是否 segfault。

**预期**：uint8_t 路径由于 `sizeof(uint8_t)=1 ≠ sizeof(int)=4`，`entry_len` 计算错误，可能导致：
- segfault（访问越界 pIndex）
- recall=0（wrong neighbor ids）
- 可能碰巧工作（若 entry_len 是独立从 metadata 读取的）

**关键分析**：`bang_load()` 从 `_disk_metadata.bin` 读取 `ulluIndexEntryLen`，不依赖 sizeof(T) 计算，所以 entry 访问可能 ok。但 query 加载（`sizeof(T) * D * numQueries`）和 `compute_L2Dist<T>` 会用错误的 T。

**当前状态**：needs_gpu + needs uint8 dataset + needs code fix验证。

---

## 测试 T-10：Batch Size vs QPS

**目的**：验证 GPU 利用率随 batch size 增大而提升。  
**对应 finding**：KERNEL-01（性能瓶颈）。  
**需要**：`needs_gpu`。

**方法**：固定 L，改变 `numQueries`（1, 10, 100, 1000, 10000），测量 QPS。

**预期**：小 batch 下 GPU 利用率低，QPS 随 numQueries 增大而提升，直到 GPU 饱和（可通过 `nvidia-smi` 或 Nsight Systems 观察利用率）。

**当前状态**：needs_gpu。

---

## 执行记录模板

实际运行时，在此处补充结果：

```
| 测试 | 日期 | 硬件 | 结果 | CSV 文件 |
|---|---|---|---|---|
| T-01 | SKIP | - | 无 GPU | results/sift10k_smoke.log |
| T-02 | - | - | - | - |
...
```
