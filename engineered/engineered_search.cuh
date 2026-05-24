#pragma once
#include "plain/plain_search.cuh"  // 复用 HostGraph, DevicePQ 定义

namespace bang_repro::engineered {

// ── engineered 搜索函数（主入口）──────────────────────────────────────────────
//
// 相比 plain::search_bang_plain，工程优化点：
//   1. Transposed PQ table（pqTable_T）改善 centroid access coalescing
//   2. 8-thread segmented PQ distance（kTPNbr=8）+ warp reduce（CUB WarpReduce 等价）
//   3. Shared-memory parallel worklist merge（BANG compute_BestLSets_par_merge 等价）
//   4. 4 CUDA streams（streamParent D2H / streamChildren H2D /
//                      streamFPTransfers H2D / streamKernels）CPU-GPU overlap
//   5. OpenMP CPU graph fetch（numCPUthreads，对应 BANG bang_search.cu:413）
void search_bang_engineered(
  const bang_repro::plain::HostGraph& graph,
  const bang_repro::plain::DevicePQ&  pq,
  const float*     d_queries,   // [numQ * dim]
  int numQ, int dim,
  int*   d_out_ids,             // [numQ * kTopK]
  float* d_out_dists
);

}  // namespace bang_repro::engineered
