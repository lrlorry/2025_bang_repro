#include "plain/plain_search.cuh"
#include "plain/config.cuh"
#include "common/cuda_utils.cuh"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <math_constants.h>

namespace bang_repro::plain {

// ─────────────────────────────────────────────────────────────────────────────
// Kernel 1: 构造 per-query PQ distance table
//
// 对应 BANG bang_search.cu::populate_pqDist_par<T>
//
// 每个 query 分配一个 block，256 个线程对应 256 个 centroid。
// 对每个 PQ chunk c，计算 query[c] 到所有 256 个 centroid 的 L2²，
// 写入 d_pq_dist_table[q * M * 256 + c * 256 + centroid_id]。
// 后续 neighbor distance 计算变为 table lookup + sum，避免反复从 pqTable 算。
//
// BANG 优化：pqTable_T 是转置布局（D×256），改善 centroid access coalescing。
// 本 plain 版使用非转置布局（M × 256 × chunk_dim）以保持可读性。
// ─────────────────────────────────────────────────────────────────────────────
__global__ void populate_pq_dist_table_kernel(
    const float* __restrict__ d_queries,   // [numQ * dim]
    const float* __restrict__ d_pq_table,  // [M * 256 * chunk_dim]
    float*       d_pq_dist_table,          // [numQ * M * 256]
    int dim, int M, int chunk_dim)
{
  int qid  = blockIdx.x;           // query index
  int cent = threadIdx.x;          // centroid index (0..255)
  if (cent >= kPQCents) return;

  const float* q = d_queries + (long long)qid * dim;

  for (int c = 0; c < M; c++) {
    const float* centroid = d_pq_table + ((long long)c * kPQCents + cent) * chunk_dim;
    const float* q_chunk  = q + c * chunk_dim;
    float dist = 0.f;
    for (int d = 0; d < chunk_dim; d++) {
      float diff = q_chunk[d] - centroid[d];
      dist += diff * diff;
    }
    d_pq_dist_table[(long long)qid * M * kPQCents + c * kPQCents + cent] = dist;
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Kernel 2: Bloom filter 去重（标记 visited）
//
// 对应 BANG bang_search.cu::neighbor_filtering_new
//
// 对每个 neighbor id，用两个 FNV-1a style hash 查 bool 数组。
// 未访问 → 标记为已访问，写入 d_nb_out；
// 已访问 → 写 -1（filtered out）。
//
// BANG 细节：无 __syncthreads()（设计如此，接受轻微 race），
//            Bloom filter 是 global bool array（非 bit-packed，非 shared hash）。
// ─────────────────────────────────────────────────────────────────────────────
__device__ __forceinline__ int hash1(int x) {
  unsigned u = (unsigned)x;
  u = (u ^ 0xdeadbeef) + (u << 4);
  u ^= (u >> 10);
  return (int)(u % (unsigned)kBF);
}

__device__ __forceinline__ int hash2(int x) {
  unsigned u = (unsigned)x * 2654435761u;
  u ^= u >> 16;
  return (int)(u % (unsigned)kBF);
}

__global__ void bloom_filter_kernel(
    const int* __restrict__ d_nb_in,   // [numQ * R] neighbors to check
    bool*      d_bloom,                 // [numQ * kBF] per-query visited array
    int*       d_nb_out,                // [numQ * R] output: -1 if filtered
    int R)
{
  int qid = blockIdx.x;
  int tid = threadIdx.x;
  if (tid >= R) return;

  int nb = d_nb_in[qid * R + tid];
  d_nb_out[qid * R + tid] = -1;
  if (nb < 0) return;

  bool* bloom_q = d_bloom + (long long)qid * kBF;
  int h1 = hash1(nb);
  int h2 = hash2(nb);
  // Race 风险和 BANG 一致：不做 __syncthreads() 保护
  if (bloom_q[h1] || bloom_q[h2]) return;
  bloom_q[h1] = true;
  bloom_q[h2] = true;
  d_nb_out[qid * R + tid] = nb;
}

// ─────────────────────────────────────────────────────────────────────────────
// Kernel 3: PQ asymmetric distance（table lookup + sum）
//
// 对应 BANG bang_search.cu::compute_neighborDist_par
//
// 对每个 filtered neighbor，从 PQ dist table 按 code 查表求和得近似 distance。
// BANG 用 THREADS_PER_NEIGHBOR=8 + CUB WarpReduce；
// plain 版每线程算完整一个 neighbor（等价于 THREADS_PER_NEIGHBOR=1）。
// ─────────────────────────────────────────────────────────────────────────────
__global__ void pq_distance_kernel(
    const int*     __restrict__ d_nb,          // [numQ * R] filtered neighbors
    const uint8_t* __restrict__ d_codes,        // [N * M] PQ codes
    const float*   __restrict__ d_pq_dist_tbl,  // [numQ * M * 256]
    float*         d_dists,                     // [numQ * R] output distances
    int R, int M, int N)
{
  int qid = blockIdx.x;
  int tid = threadIdx.x;
  if (tid >= R) return;

  int nb = d_nb[qid * R + tid];
  d_dists[qid * R + tid] = CUDART_INF_F;
  if (nb < 0 || nb >= N) return;

  const uint8_t* codes = d_codes + (long long)nb * M;
  const float*   tbl   = d_pq_dist_tbl + (long long)qid * M * kPQCents;

  float dist = 0.f;
  for (int c = 0; c < M; c++)
    dist += tbl[c * kPQCents + codes[c]];
  d_dists[qid * R + tid] = dist;
}

// ─────────────────────────────────────────────────────────────────────────────
// Kernel 4: 更新 worklist（sorted top-L 插入）
//
// 对应 BANG bang_search.cu::compute_BestLSets_par_sort_msort +
//          compute_BestLSets_par_merge
//
// BANG 用 parallel merge sort + shared memory merge（复杂）。
// plain 版：每个 query 一个线程，serial sorted insert，O(R × L）。
// 逻辑完全等价，只是没有 GPU 并行优化。
// ─────────────────────────────────────────────────────────────────────────────
__global__ void update_worklist_kernel(
    const int*   __restrict__ d_nb,       // [numQ * R]
    const float* __restrict__ d_dists,    // [numQ * R]
    int*   d_wl_ids,    // [numQ * kL]
    float* d_wl_dist,   // [numQ * kL]
    bool*  d_wl_exp,    // [numQ * kL] expanded flags
    int R)
{
  int qid = blockIdx.x * blockDim.x + threadIdx.x;
  if (qid >= gridDim.x * blockDim.x) return;

  int*   ids  = d_wl_ids  + qid * kL;
  float* dist = d_wl_dist + qid * kL;
  bool*  exp  = d_wl_exp  + qid * kL;

  for (int i = 0; i < R; i++) {
    int   nb = d_nb   [qid * R + i];
    float d  = d_dists[qid * R + i];
    if (nb < 0 || d >= dist[kL - 1]) continue;

    // 线性去重
    bool dup = false;
    for (int j = 0; j < kL; j++) if (ids[j] == nb) { dup = true; break; }
    if (dup) continue;

    // 插入排序
    int pos = kL - 1;
    while (pos > 0 && d < dist[pos - 1]) {
      ids[pos] = ids[pos-1]; dist[pos] = dist[pos-1]; exp[pos] = exp[pos-1];
      pos--;
    }
    ids[pos] = nb; dist[pos] = d; exp[pos] = false;
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Kernel 5: 选择下一轮 parent（eager prefetch）
//
// 对应 BANG bang_search.cu::compute_parent2
//
// BANG 默认路径（eager prefetch）：在 sort/merge 之前选 parent，
// 让 CPU 尽早开始 fetch graph neighbors，与 GPU sort/merge overlap。
// plain 版顺序执行，无 overlap（所以放在 update 之后），但逻辑相同。
// ─────────────────────────────────────────────────────────────────────────────
__global__ void select_next_parent_kernel(
    const int*  __restrict__ d_wl_ids,   // [numQ * kL]
    bool*       d_wl_exp,                // [numQ * kL]
    int*        d_next_parent,           // [numQ] output: -1 if none
    bool*       d_any_active)            // [1] set true if any query has a parent
{
  int qid = blockIdx.x * blockDim.x + threadIdx.x;
  if (qid >= gridDim.x * blockDim.x) return;

  const int* ids = d_wl_ids + qid * kL;
  bool*      exp = d_wl_exp + qid * kL;

  d_next_parent[qid] = -1;
  for (int i = 0; i < kL; i++) {
    if (ids[i] >= 0 && !exp[i]) {
      d_next_parent[qid] = ids[i];
      exp[i] = true;
      *d_any_active = true;
      break;
    }
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Kernel 6: Exact L2 rerank
//
// 对应 BANG bang_search.cu::compute_L2Dist<T> + compute_NearestNeighbours
//
// BANG 在 search loop 中异步 H2D candidate full vectors，
// rerank 前 cudaStreamSynchronize(streamFPTransfers)。
// plain 版：直接 memcpy，无异步。
//
// 对 worklist 前 kL 个候选做 exact L2，覆盖 PQ 近似距离，输出 top-kTopK。
// ─────────────────────────────────────────────────────────────────────────────
__global__ void exact_rerank_kernel(
    const float* __restrict__ d_queries,      // [numQ * dim]
    const float* __restrict__ d_cand_vecs,    // [numQ * kL * dim] full vectors
    const int*   __restrict__ d_wl_ids,       // [numQ * kL]
    int*   d_out_ids,    // [numQ * kTopK]
    float* d_out_dists,  // [numQ * kTopK]
    int dim)
{
  int qid = blockIdx.x;    // query index
  int tid = threadIdx.x;   // candidate index (0..kL-1)
  if (tid >= kL) return;

  int cid = d_wl_ids[qid * kL + tid];
  // 计算 exact L2²
  float exact_d = CUDART_INF_F;
  if (cid >= 0) {
    const float* q = d_queries   + (long long)qid * dim;
    const float* v = d_cand_vecs + ((long long)qid * kL + tid) * dim;
    exact_d = 0.f;
    for (int d = 0; d < dim; d++) {
      float diff = q[d] - v[d];
      exact_d += diff * diff;
    }
  }

  // 每个线程写自己的 (id, dist)，再串行 top-kTopK（plain 版只让 thread 0 做）
  // 使用 shared memory 做 block-level top-k
  extern __shared__ float shm[];  // [kL * 2]: dist + id
  float* shm_dist = shm;
  int*   shm_id   = (int*)(shm + kL);

  shm_dist[tid] = exact_d;
  shm_id[tid]   = cid;
  __syncthreads();

  // thread 0 做简单 serial selection top-kTopK
  if (tid == 0) {
    bool used[kL] = {};
    for (int k = 0; k < kTopK; k++) {
      int   best_i = -1;
      float best_d = CUDART_INF_F;
      for (int j = 0; j < kL; j++) {
        if (!used[j] && shm_dist[j] < best_d) {
          best_d = shm_dist[j]; best_i = j;
        }
      }
      if (best_i < 0) { d_out_ids[qid * kTopK + k] = -1; d_out_dists[qid * kTopK + k] = CUDART_INF_F; }
      else {
        d_out_ids  [qid * kTopK + k] = shm_id  [best_i];
        d_out_dists[qid * kTopK + k] = shm_dist[best_i];
        used[best_i] = true;
      }
    }
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// 主搜索函数：实现 BANG CPU-GPU pipeline
//
// 对应 BANGSearchInner<T>::bang_query()
//
// plain 版与 BANG 原版的关键区别：
//   - 无 CUDA streams，所有操作顺序执行（BANG 用 4 streams 重叠 CPU/GPU 操作）
//   - 无 OpenMP CPU fetch（BANG 用 numCPUthreads=64 并行 fetch 邻居）
//   - 串行 worklist update（BANG 用 parallel merge sort + shared memory merge）
//   - 无异步 FP vector prefetch（BANG 在 loop 内异步 H2D candidate full vectors）
// ─────────────────────────────────────────────────────────────────────────────
void search_bang_plain(
    const HostGraph& graph,
    const DevicePQ&  pq,
    const float*     d_queries,
    int numQ, int dim,
    int*   d_out_ids,
    float* d_out_dists)
{
  const int N = graph.N;
  const int R = graph.R;
  const int M = pq.M;

  // ── GPU buffer 分配 ────────────────────────────────────────────────────────
  // PQ distance table: numQ × M × 256
  float* d_pq_tbl = nullptr;
  CUDA_CHECK(cudaMalloc(&d_pq_tbl, (long long)numQ * M * kPQCents * sizeof(float)));

  // Worklist: sorted top-L candidates per query
  int*   d_wl_ids  = nullptr;
  float* d_wl_dist = nullptr;
  bool*  d_wl_exp  = nullptr;
  CUDA_CHECK(cudaMalloc(&d_wl_ids,  (long long)numQ * kL * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_wl_dist, (long long)numQ * kL * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_wl_exp,  (long long)numQ * kL * sizeof(bool)));

  // Bloom filter: per-query bool array（BF_ENTRIES bools）
  bool* d_bloom = nullptr;
  CUDA_CHECK(cudaMalloc(&d_bloom, (long long)numQ * kBF * sizeof(bool)));

  // Neighbor transfer buffers（CPU → GPU per iteration）
  int* d_nb_h2d = nullptr;  // H2D 传来的邻居
  int* d_nb_flt = nullptr;  // Bloom filter 后的邻居
  CUDA_CHECK(cudaMalloc(&d_nb_h2d, numQ * R * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_nb_flt, numQ * R * sizeof(int)));

  // PQ distance of filtered neighbors
  float* d_nb_dists = nullptr;
  CUDA_CHECK(cudaMalloc(&d_nb_dists, numQ * R * sizeof(float)));

  // Next parent per query（D2H per iteration）
  int*  d_next_parent = nullptr;
  bool* d_any_active  = nullptr;
  CUDA_CHECK(cudaMalloc(&d_next_parent, numQ * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_any_active, sizeof(bool)));

  // Candidate full vectors for exact rerank（host 侧 paged，D2H before rerank）
  float* d_cand_vecs = nullptr;
  CUDA_CHECK(cudaMalloc(&d_cand_vecs, (long long)numQ * kL * dim * sizeof(float)));

  // ── Host 侧 pinned buffer ─────────────────────────────────────────────────
  std::vector<int>   h_next_parent(numQ);
  std::vector<int>   h_nb(numQ * R);
  std::vector<float> h_cand_vecs(numQ * kL * dim);

  // ── 初始化：worklist 填 INF，Bloom filter 清零 ──────────────────────────
  {
    std::vector<int>   init_ids(numQ * kL, -1);
    std::vector<float> init_dist(numQ * kL, 1e30f);
    CUDA_CHECK(cudaMemcpy(d_wl_ids,  init_ids.data(),  numQ * kL * sizeof(int),   cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_wl_dist, init_dist.data(), numQ * kL * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_wl_exp,  0, (long long)numQ * kL * sizeof(bool)));
    CUDA_CHECK(cudaMemset(d_bloom,   0, (long long)numQ * kBF * sizeof(bool)));
  }

  // ── Stage 1: PQ distance table（populate_pqDist_par 等价）─────────────────
  // grid = numQ, block = 256（每线程对应一个 centroid）
  populate_pq_dist_table_kernel<<<numQ, kPQCents>>>(
      d_queries, pq.d_table, d_pq_tbl, dim, M, pq.chunk_dim);
  CUDA_CHECK(cudaGetLastError());

  // ── Stage 2: 初始种子（medoid 附近 or query 自身，这里用 node 0 作为 medoid） ──
  // BANG: compute_parent1 选初始种子放入 worklist
  // plain: 直接用 node 0 为每个 query 的起始 parent
  {
    std::vector<int> seed_nb(numQ * R);
    for (int q = 0; q < numQ; q++)
      for (int r = 0; r < R; r++)
        seed_nb[q * R + r] = graph.adj[(long long)graph.medoid * R + r];
    CUDA_CHECK(cudaMemcpy(d_nb_h2d, seed_nb.data(), numQ * R * sizeof(int), cudaMemcpyHostToDevice));
  }

  // Bloom filter + PQ distance + update worklist（初始种子）
  bloom_filter_kernel<<<numQ, R>>>(d_nb_h2d, d_bloom, d_nb_flt, R);
  CUDA_CHECK(cudaGetLastError());
  pq_distance_kernel<<<numQ, R>>>(d_nb_flt, pq.d_codes, d_pq_tbl, d_nb_dists, R, M, N);
  CUDA_CHECK(cudaGetLastError());
  update_worklist_kernel<<<numQ, 1>>>(d_nb_flt, d_nb_dists, d_wl_ids, d_wl_dist, d_wl_exp, R);
  CUDA_CHECK(cudaGetLastError());

  // ── Stage 2: Greedy search loop（CPU-GPU pipeline）────────────────────────
  //
  // BANG 核心流程：
  //   GPU → CPU: parent ids  (streamParent D2H)
  //   CPU: fetch parent's neighbors from host graph (OpenMP)
  //   CPU → GPU: neighbors   (streamChildren H2D)
  //   GPU: Bloom filter + PQ distance + update worklist
  //   repeat
  //
  // plain 版：所有步骤顺序同步执行，无 stream overlap
  for (int iter = 0; iter < kMaxIter; iter++) {

    // GPU: 选下一轮 parent（eager prefetch 在 BANG 中发生在 sort/merge 前）
    bool h_any = false;
    CUDA_CHECK(cudaMemset(d_any_active, 0, sizeof(bool)));
    select_next_parent_kernel<<<numQ, 1>>>(d_wl_ids, d_wl_exp, d_next_parent, d_any_active);
    CUDA_CHECK(cudaGetLastError());

    // D2H: parent ids
    CUDA_CHECK(cudaMemcpy(h_next_parent.data(), d_next_parent, numQ * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&h_any, d_any_active, sizeof(bool), cudaMemcpyDeviceToHost));
    if (!h_any) break;  // 所有 query 的 worklist 已全部展开，收敛

    // CPU: 从 host graph 读取每个 query 的 parent 邻居
    // 对应 BANG 源码 bang_query() 的 OpenMP fetch region
    for (int q = 0; q < numQ; q++) {
      int parent = h_next_parent[q];
      if (parent < 0) {
        for (int r = 0; r < R; r++) h_nb[q * R + r] = -1;
        continue;
      }
      for (int r = 0; r < R; r++)
        h_nb[q * R + r] = graph.adj[(long long)parent * R + r];
    }

    // H2D: neighbors
    CUDA_CHECK(cudaMemcpy(d_nb_h2d, h_nb.data(), numQ * R * sizeof(int), cudaMemcpyHostToDevice));

    // GPU: Bloom filter + PQ distance + update worklist
    bloom_filter_kernel<<<numQ, R>>>(d_nb_h2d, d_bloom, d_nb_flt, R);
    CUDA_CHECK(cudaGetLastError());
    pq_distance_kernel<<<numQ, R>>>(d_nb_flt, pq.d_codes, d_pq_tbl, d_nb_dists, R, M, N);
    CUDA_CHECK(cudaGetLastError());
    update_worklist_kernel<<<numQ, 1>>>(d_nb_flt, d_nb_dists, d_wl_ids, d_wl_dist, d_wl_exp, R);
    CUDA_CHECK(cudaGetLastError());
  }

  // ── Stage 3: Exact rerank ──────────────────────────────────────────────────
  //
  // BANG: search loop 中已异步 H2D candidate full vectors（streamFPTransfers），
  //       rerank 前 cudaStreamSynchronize(streamFPTransfers)，再 compute_L2Dist。
  // plain: 搜索结束后同步拉取 full vectors，再 exact rerank。
  {
    std::vector<int> h_wl_ids(numQ * kL);
    CUDA_CHECK(cudaMemcpy(h_wl_ids.data(), d_wl_ids, numQ * kL * sizeof(int), cudaMemcpyDeviceToHost));

    for (int q = 0; q < numQ; q++)
      for (int i = 0; i < kL; i++) {
        int cid = h_wl_ids[q * kL + i];
        if (cid < 0 || cid >= N) {
          std::fill(h_cand_vecs.begin() + (long long)(q * kL + i) * dim,
                    h_cand_vecs.begin() + (long long)(q * kL + i) * dim + dim, 0.f);
        } else {
          std::copy(graph.vecs.begin() + (long long)cid * dim,
                    graph.vecs.begin() + (long long)cid * dim + dim,
                    h_cand_vecs.begin() + (long long)(q * kL + i) * dim);
        }
      }
    CUDA_CHECK(cudaMemcpy(d_cand_vecs, h_cand_vecs.data(),
                          (long long)numQ * kL * dim * sizeof(float), cudaMemcpyHostToDevice));
  }

  // exact_rerank_kernel：每 query 一个 block，kL 个线程
  size_t shm_bytes = kL * (sizeof(float) + sizeof(int));
  exact_rerank_kernel<<<numQ, kL, shm_bytes>>>(
      d_queries, d_cand_vecs, d_wl_ids, d_out_ids, d_out_dists, dim);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  // ── 释放 GPU buffers ───────────────────────────────────────────────────────
  cudaFree(d_pq_tbl); cudaFree(d_wl_ids); cudaFree(d_wl_dist); cudaFree(d_wl_exp);
  cudaFree(d_bloom); cudaFree(d_nb_h2d); cudaFree(d_nb_flt); cudaFree(d_nb_dists);
  cudaFree(d_next_parent); cudaFree(d_any_active); cudaFree(d_cand_vecs);
}

}  // namespace bang_repro::plain
