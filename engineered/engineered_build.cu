#include "engineered/engineered_build.cuh"
#include "common/cuda_utils.cuh"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>
#include <vector>
#include <cstdio>
#include <omp.h>

namespace bang_repro::engineered {

// ────────────────────────────────────────────────────────────────────────────
// GPU kernel：批量计算一个节点与 K 个候选节点之间的 L2^2 距离
// grid: (num_queries,)   block: (min(K, 256),)
// 每个 block 处理一个 query 节点，每个线程计算一个候选距离
// ────────────────────────────────────────────────────────────────────────────
__global__ void batch_l2_kernel(
    const float* __restrict__ vecs,    // [N * dim]
    const int*   __restrict__ queries, // [numQ]  query node ids
    const int*   __restrict__ cands,   // [numQ * K]  candidate node ids
    float*       __restrict__ dists,   // [numQ * K]  output distances
    int dim, int K)
{
  int q  = blockIdx.x;
  int ki = threadIdx.x;
  if (ki >= K) return;

  int qid  = queries[q];
  int cid  = cands[q * K + ki];
  if (cid < 0) { dists[q * K + ki] = 1e30f; return; }

  const float* qv = vecs + (long long)qid * dim;
  const float* cv = vecs + (long long)cid * dim;
  float d = 0.f;
  for (int i = 0; i < dim; i++) { float t = qv[i] - cv[i]; d += t * t; }
  dists[q * K + ki] = d;
}

// ── 共享工具（与 plain_build.cu 相同逻辑，避免跨编译单元依赖） ──────────────

static inline float l2sq_host(const float* a, const float* b, int dim)
{
  float d = 0.f;
  for (int i = 0; i < dim; i++) { float t = a[i] - b[i]; d += t * t; }
  return d;
}

static int find_medoid_eng(const float* vecs, int N, int dim)
{
  std::vector<double> centroid(dim, 0.0);
  for (int i = 0; i < N; i++)
    for (int d = 0; d < dim; d++)
      centroid[d] += vecs[(long long)i * dim + d];
  for (double& c : centroid) c /= N;

  float best = 1e30f; int med = 0;
  for (int i = 0; i < N; i++) {
    float dist = 0.f;
    const float* v = vecs + (long long)i * dim;
    for (int d = 0; d < dim; d++) { float t = v[d] - (float)centroid[d]; dist += t*t; }
    if (dist < best) { best = dist; med = i; }
  }
  return med;
}

// α-RNG 裁剪：candidates 已按 d(p,c) 升序
static std::vector<int> robust_prune_eng(
    int p, std::vector<std::pair<float,int>>& cands,
    const float* vecs, int dim, int R, float alpha)
{
  std::vector<int> result;
  result.reserve(R);
  for (auto& [dp_c, c] : cands) {
    if (c == p) continue;
    if ((int)result.size() >= R) break;
    bool dominated = false;
    const float* cv = vecs + (long long)c * dim;
    for (int r : result) {
      float d_rc = l2sq_host(vecs + (long long)r * dim, cv, dim);
      if (d_rc <= alpha * dp_c) { dominated = true; break; }
    }
    if (!dominated) result.push_back(c);
  }
  return result;
}

// ── greedy_search（CPU，邻居距离来自 host 计算） ──────────────────────────────
// engineered 版本与 plain 逻辑相同；GPU 批量距离用于 robust_prune 阶段。
static std::vector<std::pair<float,int>> greedy_search_eng(
    int query_node,
    const float* vecs,
    const std::vector<int>& adj,
    int N, int dim, int R, int medoid, int L)
{
  const float* qv = vecs + (long long)query_node * dim;
  std::vector<bool> visited(N, false);

  // max-heap worklist，保留最近 L 个候选
  std::vector<std::pair<float,int>> wl;
  wl.reserve(L + R);

  auto push_cand = [&](int node) {
    if (visited[node] || node == query_node) return;
    visited[node] = true;
    float d = l2sq_host(qv, vecs + (long long)node * dim, dim);
    wl.push_back({d, node});
    std::push_heap(wl.begin(), wl.end());
    if ((int)wl.size() > L) { std::pop_heap(wl.begin(), wl.end()); wl.pop_back(); }
  };

  visited[medoid] = true;
  if (medoid != query_node) {
    float d = l2sq_host(qv, vecs + (long long)medoid * dim, dim);
    wl.push_back({d, medoid});
    std::push_heap(wl.begin(), wl.end());
  }

  std::vector<bool> expanded(N, false);
  bool changed = true;
  while (changed) {
    changed = false;
    std::vector<std::pair<float,int>> sv = wl;
    std::sort_heap(sv.begin(), sv.end()); // ascending
    std::make_heap(wl.begin(), wl.end()); // restore
    for (auto& [d, node] : sv) {
      if (expanded[node]) continue;
      expanded[node] = true; changed = true;
      for (int r = 0; r < R; r++) {
        int nb = adj[(long long)node * R + r];
        if (nb < 0) break;
        push_cand(nb);
      }
      break;
    }
  }
  std::sort(wl.begin(), wl.end());
  return wl;
}

// ── build_vamana_engineered ──────────────────────────────────────────────────
bang_repro::plain::HostGraph build_vamana_engineered(
    const std::vector<float>& vecs,
    int N, int dim, int R, float alpha, int L_build)
{
  std::printf("[build-eng] Vamana engineered: N=%d dim=%d R=%d alpha=%.1f L_build=%d\n",
              N, dim, R, alpha, L_build);
  std::printf("[build-eng] OpenMP threads: %d\n", omp_get_max_threads());

  bang_repro::plain::HostGraph g;
  g.N = N; g.R = R; g.vecs = vecs;
  g.adj.assign((long long)N * R, -1);

  // 1. 随机初始化
  std::mt19937 rng(1234);
  std::vector<int> pool(N); std::iota(pool.begin(), pool.end(), 0);
  for (int i = 0; i < N; i++) {
    std::shuffle(pool.begin(), pool.end(), rng);
    int k = 0;
    for (int j = 0; j < N && k < R; j++)
      if (pool[j] != i) g.adj[(long long)i * R + k++] = pool[j];
  }

  // 2. 找 medoid
  int medoid = find_medoid_eng(vecs.data(), N, dim);
  std::printf("[build-eng] medoid = %d\n", medoid);

  // 4. Vamana 主循环
  std::vector<int> order(N); std::iota(order.begin(), order.end(), 0);
  std::shuffle(order.begin(), order.end(), rng);

  // pass-1：并行为每个节点收集新邻居（不写反向边，避免数据竞争）
  // 存储为 new_adj_flat[i*R .. i*R+R-1]
  std::vector<int> new_adj_flat((long long)N * R, -1);

  // 每线程独立做 host L2（batch_l2_kernel 可替换此处，每线程创建独立 stream + buffer）

  #pragma omp parallel for schedule(dynamic, 32)
  for (int idx = 0; idx < N; idx++) {
    int p = order[idx];

    // greedy_search（读 g.adj，只读，无竞争）
    auto candidates = greedy_search_eng(p, vecs.data(), g.adj, N, dim, R, medoid, L_build);
    if (candidates.empty()) continue;

    // robust_prune（纯 CPU，每线程独立）
    auto new_nbrs = robust_prune_eng(p, candidates, vecs.data(), dim, R, alpha);

    int cnt = std::min((int)new_nbrs.size(), R);
    for (int k = 0; k < R; k++)
      new_adj_flat[(long long)p * R + k] = (k < cnt) ? new_nbrs[k] : -1;
  }

  // pass-2：更新 adj + 反向边（串行，避免写竞争）
  for (int p = 0; p < N; p++) {
    for (int k = 0; k < R; k++)
      g.adj[(long long)p * R + k] = new_adj_flat[(long long)p * R + k];
  }

  // 反向边：对每个节点 p，对其邻居 j：将 p 加入 N(j)，超限则 prune
  auto prune_to_R = [&](int node, std::vector<int>& nbr_set) -> std::vector<int> {
    if ((int)nbr_set.size() <= R) return nbr_set;
    const float* pv = vecs.data() + (long long)node * dim;
    std::vector<std::pair<float,int>> cands;
    cands.reserve(nbr_set.size());
    for (int nb : nbr_set) {
      float d = l2sq_host(pv, vecs.data() + (long long)nb * dim, dim);
      cands.push_back({d, nb});
    }
    std::sort(cands.begin(), cands.end());
    return robust_prune_eng(node, cands, vecs.data(), dim, R, alpha);
  };

  for (int p = 0; p < N; p++) {
    for (int k = 0; k < R; k++) {
      int j = g.adj[(long long)p * R + k];
      if (j < 0) break;

      // 收集 N(j)
      std::vector<int> j_nbrs;
      for (int r = 0; r < R; r++) {
        int nb = g.adj[(long long)j * R + r];
        if (nb < 0) break;
        j_nbrs.push_back(nb);
      }
      if (std::find(j_nbrs.begin(), j_nbrs.end(), p) == j_nbrs.end())
        j_nbrs.push_back(p);

      if ((int)j_nbrs.size() > R) j_nbrs = prune_to_R(j, j_nbrs);

      int cnt = std::min((int)j_nbrs.size(), R);
      for (int r = 0; r < R; r++)
        g.adj[(long long)j * R + r] = (r < cnt) ? j_nbrs[r] : -1;
    }
  }

  g.medoid = medoid;

  std::printf("[build-eng] Vamana engineered done.\n");
  return g;
}

}  // namespace bang_repro::engineered
