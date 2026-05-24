#pragma once
#include "plain/plain_search.cuh"

namespace bang_repro::plain {

// ── Vamana 图构建（CPU-only reference 实现）────────────────────────────────
//
// 对应 BANG 源码 bang_preprocess.py 中的 build_vamana()：
//   1. find_medoid：找距离所有点平均最近的节点作为出发点
//   2. 初始化：每个节点随机 R 个邻居
//   3. Vamana 主循环（random order）：
//      a. greedy_search(p, graph, medoid, L_build) → 候选集
//      b. robust_prune(p, candidates, alpha, R) → N(p)（α-RNG pruning）
//      c. 反向边添加：for j in N(p): N(j) ∪= {p}; if |N(j)| > R: prune(j)
//
// BANG 原版使用两轮：第一轮 alpha=1.0，第二轮 alpha=1.2
// 这里合并为可配置的单轮调用。

HostGraph build_vamana_plain(
  const std::vector<float>& vecs,
  int N, int dim,
  int R,                // 图度数（BANG MAX_R=64，demo 用 kR）
  float alpha = 1.2f,  // α-RNG 裁剪系数（BANG 第二轮 alpha=1.2）
  int L_build = 64     // build-time worklist 大小（≥ R）
);

}  // namespace bang_repro::plain
