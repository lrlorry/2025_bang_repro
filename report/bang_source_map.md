# BANG Base 源码地图

> **审计基准**：`../BANG-Billion-Scale-ANN/BANG_Base/`，检查日期 2026-05-24。  
> **源文件**：`bang_search.cu`、`bang_search.cuh`、`test_driver.cpp`、`bang.h`、`ReadMe.pdf`。

---

## 1. 文件总览

| 文件 | 行数 | 主要职责 |
|---|---|---|
| `bang.h` | ~60 | Public API，`BANGSearch<T>` 模板类声明 |
| `bang_search.cuh` | ~373 | 内部数据结构定义 + 所有 kernel 声明 |
| `bang_search.cu` | ~1100+ | 全部实现：bang_load/alloc/init/query/free/unload + 所有 kernels |
| `test_driver.cpp` | ~599 | 独立 main()：CLI 解析、benchmark sweep、recall 计算 |
| `bang_preprocess.py` | ~100 | 将 DiskANN `_disk.index` 转换为 BANG 格式的 `_disk.bin` + `_disk_metadata.bin` |
| `CMakeLists.txt` | ~30 | cmake 构建配置 |

---

## 2. 核心数据结构（`bang_search.cuh`）

### 2.1 `GraphMedataData`（32 字节，packed）

```cpp
// bang_search.cuh:42
typedef struct __attribute__((__packed__)) _GraphMedatadata {
    unsigned long long ullMedoid;       // medoid node id
    unsigned long long ulluIndexEntryLen; // 每个 graph entry 的字节长度
    int uDatatype;
    unsigned uDim;
    unsigned uDegree;     // degree bound（必须 == MAX_R=64）
    unsigned uDatasetSize;
} GraphMedataData;
```

由 `bang_preprocess.py` 写入 `_disk_metadata.bin`，`bang_load()` 首先读取。

### 2.2 `IndexLoad`

```cpp
// bang_search.cuh:53
typedef struct _IndexLoad {
    unsigned long long MEDOID;
    unsigned uDataType;
    unsigned D;        // 维度
    unsigned R;        // degree bound
    unsigned N;        // 数据集大小
    unsigned int uChunks;  // PQ chunks 数量（= bytes per compressed vector）

    off_t size_indexfile;
    unsigned long long ullIndex_Entry_LEN;  // graph entry 步长（字节）

    // GPU 侧（常驻 HBM）
    uint8_t* d_compressedVectors;  // [N × uChunks] PQ codes
    float*   d_pqTable;            // [D × 256] 转置 PQ pivot table（coalescing 优化）
    unsigned* d_chunksOffset;      // [uChunks+1] chunk 边界
    float*   d_centroid;           // [D] dataset centroid

    // Host 侧（常驻 host RAM）
    uint8_t* pIndex;               // graph index（全部 entry 连续）
} IndexLoad;
```

**关键**：`pIndex` 是整个 graph 的 host pointer，不在 GPU。访问单个节点：

```cpp
// bang_search.cu:335
T* full_vector = (T*)(pIndex + ullIndex_Entry_LEN * node_id);
unsigned* num_neighbors = (unsigned*)(pIndex + ullIndex_Entry_LEN * node_id + D * sizeof(T));
unsigned* neighbor_ids  = num_neighbors + 1;
```

### 2.3 `GPUInstance`

```cpp
// bang_search.cuh:79
typedef struct _GPUInstance {
    // kernel block sizes
    unsigned numThreads_K1;  // 256  populate_pqDist_par
    unsigned numThreads_K2;  // 512  compute_neighborDist_par
    unsigned numThreads_K3;  // R+1  sort kernel
    unsigned numThreads_K3_merge; // 2*L  merge kernel（≤ 1024）
    unsigned numThreads_K4;  // 1    compute_parent
    unsigned numThreads_K5;  // 256  neighbor_filtering_new

    // GPU buffers（搜索期间）
    void*         d_queriesFP;          // [numQ × D]
    result_ann_t* d_nearestNeighbours;  // [recall × numQ] 最终输出
    float*        d_pqDistTables;       // [numQ × uChunks × 256]
    float*        d_BestLSetsDist;      // [numQ × L] worklist 距离
    unsigned*     d_BestLSets;          // [numQ × L] worklist ids
    bool*         d_BestLSets_visited;  // [numQ × L] visited 标记
    unsigned*     d_parents;            // [numQ × SIZEPARENTLIST(=2)] D2H parent
    float*        d_neighborsDist_query;// [numQ × (R+1)] filtered neighbor PQ dist
    unsigned*     d_neighbors;          // [numQ × (R+1)] filtered neighbor ids
    unsigned*     d_neighbors_temp;     // [numQ × (R+1)] 未过滤 H2D 缓冲
    unsigned*     d_numNeighbors_query; // [numQ] filtered neighbor count
    unsigned*     d_processed_bit_vec;  // [numQ × BF_MEMORY] Bloom visited set
    bool*         d_nextIter;           // 是否继续下一轮（1 bool）
    void*         d_FPSetCoordsList;    // [MAX_PARENTS × numQ × D × sizeof(T)] rerank 候选
    unsigned*     d_FPSetCoordsList_Counts; // [numQ]
    float*        d_L2distances;        // [MAX_PARENTS × numQ] exact distances
    unsigned*     d_L2ParentIds;        // [MAX_PARENTS × numQ] candidate ids

    // CUDA streams
    cudaStream_t streamFPTransfers; // candidate full vectors H2D
    cudaStream_t streamParent;      // parent ids D2H
    cudaStream_t streamChildren;    // neighbor ids H2D
    cudaStream_t streamKernels;     // GPU kernels
} GPUInstance;
```

### 2.4 `HostInstance`

```cpp
// bang_search.cuh:139
typedef struct _HostInstance {
    unsigned numCPUthreads;  // 硬编码 64（bang_search.cu:413）
    unsigned* parents;       // [numQ × 2] pinned，D2H parent 结果
    unsigned* neighbors;     // [numQ × (R+1)] pinned，CPU fetch 后的 neighbors
    unsigned* numNeighbors_query; // [numQ] pinned
    void*     FPSetCoordsList;   // [MAX_PARENTS × numQ × D × sizeof(T)] pinned
    // ...
} HostInstance;
```

---

## 3. 调用链

### 3.1 整体初始化

```text
main()                                      test_driver.cpp:564
  └─ run_anns<T>()                          test_driver.cpp:339
       ├─ BANGSearch<T> objBANG             对象构造
       │    └─ new BANGSearchInner<int>()   ⚠️ 始终 int，忽略 T（bang_search.cu:73）
       ├─ objBANG.bang_load(argv[1])        读取所有 index/PQ 文件
       ├─ objBANG.bang_set_searchparams()   设置 recall_k, L, dist_fn
       ├─ objBANG.bang_alloc(numQueries)    分配 GPU/host buffers + 创建 streams
       └─ loop: 多个 L 值
            ├─ objBANG.bang_init(numQ)      初始化 buffers，设置 medoid 为首个 parent
            ├─ objBANG.bang_query(...)      主搜索（见 3.2）
            └─ calculate_recall(...)        计算 recall@k
```

### 3.2 `bang_load()`（`bang_search.cu:139`）

```text
bang_load(prefix)
  ├─ 读 metadata: _disk_metadata.bin → GraphMedataData（含 medoid、D、R、N、entry_len）
  ├─ 读 PQ compressed: _pq_compressed.bin → host → cudaMemcpy → d_compressedVectors（GPU）
  ├─ 读 PQ pivots: _pq_pivots.bin
  │    ├─ pqTable[256 × D]（原始布局）
  │    ├─ transpose → pqTable_T[D × 256]（bang_search.cu:281-285，coalescing 优化）
  │    ├─ centroid[D]
  │    └─ chunksOffset[uChunks+1]
  │    → cudaMemcpy → d_pqTable / d_centroid / d_chunksOffset（GPU）
  └─ 读 graph: _disk.bin → host malloc → pIndex（host RAM，不上 GPU）
       → sanity check：打印 first/last neighbor
```

### 3.3 `bang_alloc()`（`bang_search.cu:367`）

```text
bang_alloc(numQueries)
  ├─ cudaMalloc: 所有 GPU buffers（d_queriesFP, d_pqDistTables, d_BestLSets*, ...）
  ├─ cudaStreamCreate × 4
  ├─ 设置 numCPUthreads = 64（硬编码）
  └─ cudaMallocHost: pinned host buffers（neighbors, parents, FPSetCoordsList）
```

### 3.4 `bang_init()`（`bang_search.cu:428`）

```text
bang_init(numQueries)
  ├─ 初始化 kernel block sizes（K1=256, K2=512, K3=R+1, K3_merge=2L, K5=256）
  ├─ cudaMemset: pqDistTables=0, Bloom=0, parents=0, BestLSets_count=0
  ├─ 从 pIndex 取 medoid 的邻居 → host neighbors buffer
  ├─ 所有 query 的 L2ParentIds = MEDOID（首个 candidate）
  └─ cudaMemcpy: neighbors_temp, L2ParentIds, d_FPSetCoordsList[0] → GPU
```

### 3.5 `bang_query()` 主循环（`bang_search.cu:570`）

```text
bang_query(queriesFP, numQ, ...)
  │
  ├─ cudaMemcpy d_queriesFP ← queriesFP（H2D，同步）
  │
  ├─ [K1] populate_pqDist_par<<<numQ, 256, D*(4+sizeof(T)), streamKernels>>>
  │        per-query PQ distance table（query - centroid distance per chunk/centroid）
  │
  ├─ [K5] neighbor_filtering_new<<<numQ, 256, 0, streamKernels>>>
  │        初始 Bloom filter（对 medoid 邻居）
  │
  ├─ [K2] compute_neighborDist_par<<<numQ, 512, 0, streamKernels>>>
  │        PQ asymmetric distance for filtered neighbors
  │
  ├─ [K4] compute_parent1<<<..., 1, 0, streamKernels>>>
  │        选初始 parent，写 d_parents
  │
  ├─ cudaStreamSynchronize(streamKernels)
  │
  └─ do {
       ├─ cudaMemcpyAsync d_parents → parents（D2H, streamParent）
       │
       ├─ [K3s] compute_BestLSets_par_sort_msort<<<numQ, R+1, 0, streamKernels>>>
       │         sort filtered neighbors by PQ dist（与 CPU fetch 并行）
       │
       ├─ [K3m] compute_BestLSets_par_merge<<<numQ, 2L, (R+1)*4, streamKernels>>>
       │         merge sorted neighbors into top-L worklist
       │
       ├─ cudaStreamSynchronize(streamParent)   ← 等 parent D2H 完成
       │
       ├─ [CPU OpenMP] 从 pIndex 取 parent 邻居
       │    ├─ memcpy full vector → FPSetCoordsList[iter * row]（rerank 缓冲）
       │    └─ memcpy neighbor ids → host neighbors
       │
       ├─ cudaMemcpyAsync neighbors → d_neighbors_temp（H2D, streamChildren）
       ├─ cudaMemcpyAsync FPSetCoordsList[iter] → d_FPSetCoordsList[iter]（H2D, streamFPTransfers）
       ├─ cudaStreamSynchronize(streamChildren)
       │
       ├─ [K5] neighbor_filtering_new<<<..., streamKernels>>>（新一轮 Bloom filter）
       ├─ [K2] compute_neighborDist_par<<<..., streamKernels>>>（新邻居 PQ dist）
       │
       ├─ [K4] compute_parent2<<<..., streamKernels>>>   ← eager prefetch
       │         在 sort/merge 之前选下一轮 parent（overlap CPU fetch）
       │
       ├─ cudaMemcpyAsync d_nextIter → nextIter（D2H, streamKernels）
       └─ cudaStreamSynchronize(streamKernels)
     } while(nextIter && iter < MAX_PARENTS_PERQUERY)
  │
  ├─ cudaStreamSynchronize(streamFPTransfers)  ← 等 full vectors 全到 GPU
  │
  ├─ [K_L2] compute_L2Dist<<<numQ, 256, D*sizeof(T)>>>
  │           exact L2 distance: query vs candidate full vectors
  │
  └─ [K_NN] compute_NearestNeighbours<<<numQ, MAX_PARENTS>>>
              sort candidates by exact dist, output top-k
```

---

## 4. Kernel 索引

| Kernel | 声明 | 功能 | block/grid | 关键参数 |
|---|---|---|---|---|
| `populate_pqDist_par<T>` | cuh:162 | per-query PQ dist table | numQ blocks, 256 threads | shared mem = D*(4+sizeof(T)) |
| `neighbor_filtering_new` | cuh:237 | Bloom visited filter | numQ blocks, 256 threads | `d_processed_bit_vec`（399887 bool/query）|
| `compute_neighborDist_par` | cuh:182 | PQ asymmetric distance | numQ blocks, 512 threads | `THREADS_PER_NEIGHBOR=8`，CUB WarpReduce |
| `compute_parent1` | cuh:190 | 初始 parent 选择 | ceil(numQ/1) blocks, 1 thread | 单线程/query |
| `compute_parent2` | cuh:201 | eager next parent | ceil(numQ/1) blocks, 1 thread | 单线程/query，在 sort/merge 前执行 |
| `compute_BestLSets_par_sort_msort` | cuh:213 | 对 filtered neighbors 排序 | numQ blocks, R+1 threads | parallel merge sort |
| `compute_BestLSets_par_merge` | cuh:222 | merge → top-L worklist | numQ blocks, 2L threads | shared mem = (R+1)*4；`2L ≤ 1024` |
| `compute_L2Dist<T>` | cuh:172 | exact L2 for rerank | numQ blocks, 256 threads | shared mem = D*sizeof(T) |
| `compute_NearestNeighbours` | cuh:246 | rerank → top-k 输出 | numQ blocks, MAX_PARENTS threads | 排序选 top-k |
| `compute_neighborDist_par_cachewarmup` | cuh:255 | **未确认**是否在主路径使用 | unknown | 可能是实验残留 |

---

## 5. 编译时宏与常量（`bang_search.cu`）

| 宏/常量 | 位置 | 值 | 说明 |
|---|---|---|---|
| `MAX_R` | cu:35 | 64 | graph degree bound；与 DiskANN `-R` 必须一致 |
| `BF_ENTRIES` | cu:48 | 399887 | 每 query Bloom filter 大小（bool array，非 bit-packed）|
| `BF_MEMORY` | cu:50 | ~399888 | 4-byte aligned Bloom 内存大小 |
| `NAX_EXTRA_ITERATION` | cu:53 | 50 | search iteration 额外上界 |
| `MAX_PARENTS_PERQUERY` | cu:54 | `MAX_L + 50` | rerank candidate 数量上界 |
| `SIZEPARENTLIST` | cu:58 | 2 | parent buffer 格式：[count, parent_id] |
| `_TIMERS` | cu:60 | 注释 | 启用 kernel 计时 |
| `_NO_PRETECH` | cu:62 | 注释 | 禁用 eager prefetch（`compute_parent2` 提前运行）|
| `_NO_ASYNC_FP` | cu:63 | 注释 | 禁用异步 full-vector H2D |
| `numCPUthreads` | cu:413 | 64（hardcoded）| TODO：应动态获取 |

---

## 6. 已知代码缺陷

### 6.1 `BANGSearch<T>` 构造函数类型安全问题

```cpp
// bang_search.cu:73
template <typename T>
BANGSearch<T>::BANGSearch() {
    m_pImpl = new BANGSearchInner<int>();  // 始终 int，忽略 T！
}
```

`BANGSearchInner<T>*` 指针被强制转换为 `BANGSearchInner<T>*` 使用（cu:86），这在 T=float 时碰巧正常（因为 int/float 都是 4 bytes），但 T=uint8_t 时会导致维度计算错误。**需要实验验证。**

### 6.2 `compute_neighborDist_par_cachewarmup` 未确认

```cpp
// bang_search.cuh:255
__global__ void compute_neighborDist_par_cachewarmup(unsigned* d_neighbors,
    uint8_t* d_compressedVectors, unsigned n_chunks, unsigned R);
```

声明存在，但 `bang_search.cu` 主路径中未找到调用。可能是实验残留或备用路径。TODO：grep 确认。

### 6.3 `d_mark` 初始化

```cpp
// bang_search.cu:446
gpuErrchk(cudaMemset(m_objGPUInst.d_mark, 1, sizeof(unsigned)*(numQueries)));
// 注释：ToDo, should be 0?
```

`cudaMemset` 按字节设置，`1` 实际会让每个 int 变成 `0x01010101 = 16843009`，不是 1。原始注释怀疑是否应为 0，但未确认。

---

## 7. 文件后缀映射（`bang_load()` 拼接逻辑）

```cpp
// bang_search.cu:39-45
#define PQ_PIVOTS_FILE_SUFFIX              "_pq_pivots.bin"
#define PQ_COMPRESSEDVECTORS_FILE_SUFFIX   "_pq_compressed.bin"
#define GRAPH_INDEX_FILE_SUFFIX            "_disk.bin"
#define GRAPH_INDEX_METADATA_FILE_SUFFIX   "_disk_metadata.bin"
```

`bang_load(prefix)` → 打开 `prefix + 每个后缀`。

`_disk.bin` 和 `_disk_metadata.bin` 由 `bang_preprocess.py` 从 DiskANN 的 `_disk.index` 生成。

---

## 8. Stream 用途映射

| Stream | 方向 | 用于 | 与哪个操作 overlap |
|---|---|---|---|
| `streamKernels` | — | 所有 GPU kernels | streamParent D2H（CPU fetch 期间）|
| `streamParent` | D2H | `d_parents → parents` | sort/merge kernels |
| `streamChildren` | H2D | `neighbors → d_neighbors_temp` | FP transfer + kernels |
| `streamFPTransfers` | H2D | `FPSetCoordsList → d_FPSetCoordsList` | 全部 search loop；rerank 前 sync |
