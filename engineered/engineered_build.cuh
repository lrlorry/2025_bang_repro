#pragma once
#include "plain/plain_search.cuh"

namespace bang_repro::engineered {

// ── GPU 加速 Vamana 图构建 ───────────────────────────────────────────────────
//
// 对应 BANG 的预处理阶段（bang_preprocess.py），将 CPU-heavy 部分搬到 GPU：
//
//   1. GPU 批量 L2 距离矩阵（used for greedy_search seed + prune distance queries）
//      kernel: batch_l2_kernel<<<N, 128>>> — 为每个节点计算与所有邻居候选的距离
//
//   2. CPU RobustPrune（α-RNG pruning 逻辑不变，但距离已由 GPU 计算好）
//      与 plain_build.cu 共享同一 robust_prune() 逻辑
//
//   3. OpenMP 并行外层循环（Vamana 主循环按 batch 并行，每个节点独立 prune）
//      #pragma omp parallel for schedule(dynamic, 32)
//
// 注意：严格的 Vamana 串行化（反向边写回）与并行化存在冲突；
//       此处按 BANG 原版选择：先并行收集所有节点的新邻居，再统一处理反向边。
//       这是一个已知的近似，与 DiskANN 论文一致（pass-1 parallel, pass-2 serial）

bang_repro::plain::HostGraph build_vamana_engineered(
  const std::vector<float>& vecs,
  int N, int dim,
  int R,
  float alpha   = 1.2f,
  int L_build   = 64
);

}  // namespace bang_repro::engineered
