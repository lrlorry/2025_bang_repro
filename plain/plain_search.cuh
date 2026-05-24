#pragma once
#include <cstdint>
#include <vector>

namespace bang_repro::plain {

// ── Host graph（驻留 CPU RAM，不上传到 GPU）────────────────────────────────
//
// 对应 BANG 源码 bang_search.cu 中的 pIndex：
//   每个 entry = [full_vector (float) | neighbor_count (uint32) | neighbor_ids (uint32 * R)]
//
// 这里拆分为两个独立数组，比 BANG 原版更易读。
struct HostGraph {
  int N, R;
  std::vector<float> vecs;  // [N * dim]  full precision vectors
  std::vector<int>   adj;   // [N * R]    neighbor ids  (-1 = empty slot)
};

// ── GPU PQ 数据 ───────────────────────────────────────────────────────────────
//
// 对应 BANG 源码 bang_search.cu 中的 d_compressedVectors / d_pqTable_T：
//   d_codes:  N × M uint8，每个 vector 每个 chunk 的 centroid index
//   d_table:  M × 256 × chunk_dim float32，PQ codebook
struct DevicePQ {
  uint8_t* d_codes;  // [N * M]
  float*   d_table;  // [M * kPQCents * kChunkDim]
  int N, M, chunk_dim;
};

// ── 搜索函数（主入口）────────────────────────────────────────────────────────
//
// 对应 BANG 源码 BANGSearchInner<T>::bang_query()
void search_bang_plain(
  const HostGraph& graph,
  const DevicePQ&  pq,
  const float*     d_queries,   // [numQ * dim] on GPU
  int numQ, int dim,
  int*   d_out_ids,             // [numQ * kTopK] output on GPU
  float* d_out_dists            // [numQ * kTopK] output on GPU
);

}  // namespace bang_repro::plain
