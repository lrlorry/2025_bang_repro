#include "plain/plain_build.cuh"
#include "plain/plain_search.cuh"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>
#include <vector>
#include <cstdio>

namespace bang_repro::plain {

// ────────────────────────────────────────────────────────────────────────────
// 工具函数
// ────────────────────────────────────────────────────────────────────────────

static inline float l2sq(const float* a, const float* b, int dim)
{
  float d = 0.f;
  for (int i = 0; i < dim; i++) { float t = a[i] - b[i]; d += t * t; }
  return d;
}

// ── find_medoid ──────────────────────────────────────────────────────────────
// 对应 BANG bang_preprocess.py: find_medoid()
// 找到距离数据集质心欧氏最近的点作为 medoid（Vamana 从此出发搜索）
static int find_medoid(const float* vecs, int N, int dim)
{
  // 计算质心
  std::vector<double> centroid(dim, 0.0);
  for (int i = 0; i < N; i++)
    for (int d = 0; d < dim; d++)
      centroid[d] += vecs[(long long)i * dim + d];
  for (double& c : centroid) c /= N;

  // 找距质心最近的点
  float best = 1e30f;
  int med = 0;
  for (int i = 0; i < N; i++) {
    float dist = 0.f;
    const float* v = vecs + (long long)i * dim;
    for (int d = 0; d < dim; d++) {
      float t = v[d] - (float)centroid[d];
      dist += t * t;
    }
    if (dist < best) { best = dist; med = i; }
  }
  return med;
}

// ── greedy_search ────────────────────────────────────────────────────────────
// 对应 BANG bang_preprocess.py: greedy_search()
// 从 medoid 出发，在当前图上贪心搜索距 query p 最近的 L 个候选
// 返回: 按距离升序排列的 {dist, node} 对，最多 L 个
static std::vector<std::pair<float,int>> greedy_search(
    int query_node,
    const std::vector<float>& vecs,
    const std::vector<int>&   adj,
    int N, int dim, int R, int medoid, int L)
{
  const float* qv = vecs.data() + (long long)query_node * dim;

  // visited 标记（小 N 下用 bool 数组即可）
  std::vector<bool> visited(N, false);

  // worklist: {dist, node}，按 dist 升序
  std::vector<std::pair<float,int>> wl;
  wl.reserve(L + R);

  auto push_candidate = [&](int node) {
    if (visited[node] || node == query_node) return;
    visited[node] = true;
    float d = l2sq(qv, vecs.data() + (long long)node * dim, dim);
    wl.push_back({d, node});
    std::push_heap(wl.begin(), wl.end()); // max-heap
    if ((int)wl.size() > L) {
      std::pop_heap(wl.begin(), wl.end());
      wl.pop_back();
    }
  };

  // 从 medoid 开始
  visited[medoid] = true;
  if (medoid != query_node) {
    float d = l2sq(qv, vecs.data() + (long long)medoid * dim, dim);
    wl.push_back({d, medoid});
    std::push_heap(wl.begin(), wl.end());
  }

  // 贪心扩展：每轮从 wl 中取最近未扩展节点
  std::vector<bool> expanded(N, false);
  bool changed = true;
  while (changed) {
    changed = false;
    // 从 wl 找第一个未 expanded 的节点（wl 是 max-heap，转换为 sorted view）
    std::vector<std::pair<float,int>> sorted_wl = wl;
    std::sort_heap(sorted_wl.begin(), sorted_wl.end()); // 变为升序（pair 按 first 比）
    // 注意 sort_heap 后变无效堆，wl 此后只用来收集结果

    for (auto& [d, node] : sorted_wl) {
      if (expanded[node]) continue;
      expanded[node] = true;
      changed = true;
      // 扩展 node 的邻居
      for (int r = 0; r < R; r++) {
        int nb = adj[(long long)node * R + r];
        if (nb < 0) break;
        push_candidate(nb);
      }
      break; // 每轮只扩展最近一个
    }
    // 重新将 wl 变为合法 max-heap
    std::make_heap(wl.begin(), wl.end());
  }

  // 返回升序结果
  std::sort(wl.begin(), wl.end());
  return wl;
}

// ── robust_prune ────────────────────────────────────────────────────────────
// 对应 BANG bang_preprocess.py: robust_prune()
// α-RNG 裁剪：按距离从小到大遍历候选，保留满足 α-RNG 条件的候选
// 条件: 对于候选 c，若已选集 result 中存在 r 使得 d(r,c) ≤ α·d(p,c) 则舍弃 c
static std::vector<int> robust_prune(
    int p,
    std::vector<std::pair<float,int>>& candidates, // {d(p,c), c}，已升序
    const std::vector<float>& vecs,
    int dim, int R, float alpha)
{
  std::vector<int> result;
  result.reserve(R);

  for (auto& [dp_c, c] : candidates) {
    if (c == p) continue;
    if ((int)result.size() >= R) break;

    bool dominated = false;
    const float* cv = vecs.data() + (long long)c * dim;
    for (int r : result) {
      const float* rv = vecs.data() + (long long)r * dim;
      float d_rc = l2sq(rv, cv, dim);
      if (d_rc <= alpha * dp_c) { dominated = true; break; }
    }
    if (!dominated) result.push_back(c);
  }
  return result;
}

// ── build_vamana_plain ───────────────────────────────────────────────────────
// 对应 BANG bang_preprocess.py: build_vamana()
// 完整 Vamana 构建：随机初始化 → 主循环（greedy_search + robust_prune）→ 反向边
HostGraph build_vamana_plain(
    const std::vector<float>& vecs,
    int N, int dim, int R, float alpha, int L_build)
{
  std::printf("[build] Vamana plain: N=%d dim=%d R=%d alpha=%.1f L_build=%d\n",
              N, dim, R, alpha, L_build);

  HostGraph g;
  g.N = N; g.R = R;
  g.vecs = vecs;
  g.adj.assign((long long)N * R, -1);

  // 1. 随机初始化邻居
  std::mt19937 rng(1234);
  std::vector<int> pool(N);
  std::iota(pool.begin(), pool.end(), 0);
  for (int i = 0; i < N; i++) {
    std::shuffle(pool.begin(), pool.end(), rng);
    int k = 0;
    for (int j = 0; j < N && k < R; j++)
      if (pool[j] != i) g.adj[(long long)i * R + k++] = pool[j];
  }

  // 2. 找 medoid
  int medoid = find_medoid(vecs.data(), N, dim);
  std::printf("[build] medoid = %d\n", medoid);

  // 3. 随机序 Vamana 主循环
  std::vector<int> order(N);
  std::iota(order.begin(), order.end(), 0);
  std::shuffle(order.begin(), order.end(), rng);

  // 辅助 lambda：将新邻居列表写回 adj，超出 R 则 robust_prune
  auto set_neighbors = [&](int node, std::vector<int>& nbrs) {
    // 写入 adj
    int cnt = std::min((int)nbrs.size(), R);
    for (int k = 0; k < R; k++)
      g.adj[(long long)node * R + k] = (k < cnt) ? nbrs[k] : -1;
  };

  auto prune_to_R = [&](int node, std::vector<int>& nbr_set) -> std::vector<int> {
    if ((int)nbr_set.size() <= R) return nbr_set;
    const float* pv = vecs.data() + (long long)node * dim;
    std::vector<std::pair<float,int>> cands;
    cands.reserve(nbr_set.size());
    for (int nb : nbr_set) {
      float d = l2sq(pv, vecs.data() + (long long)nb * dim, dim);
      cands.push_back({d, nb});
    }
    std::sort(cands.begin(), cands.end());
    return robust_prune(node, cands, vecs, dim, R, alpha);
  };

  for (int idx = 0; idx < N; idx++) {
    int p = order[idx];

    // a. 贪心搜索得到候选集
    auto candidates = greedy_search(p, vecs, g.adj, N, dim, R, medoid, L_build);
    if (candidates.empty()) continue;

    // b. α-RNG 裁剪 → N(p)
    auto new_nbrs = robust_prune(p, candidates, vecs, dim, R, alpha);

    // 写回 N(p)
    set_neighbors(p, new_nbrs);

    // c. 反向边添加：for j in N(p): N(j) ∪= {p}; if overflow: prune
    for (int j : new_nbrs) {
      // 收集 N(j) 的当前邻居
      std::vector<int> j_nbrs;
      for (int k = 0; k < R; k++) {
        int nb = g.adj[(long long)j * R + k];
        if (nb < 0) break;
        j_nbrs.push_back(nb);
      }
      // 加入 p
      if (std::find(j_nbrs.begin(), j_nbrs.end(), p) == j_nbrs.end())
        j_nbrs.push_back(p);

      if ((int)j_nbrs.size() > R) {
        j_nbrs = prune_to_R(j, j_nbrs);
      }
      set_neighbors(j, j_nbrs);
    }

    if (idx % 1000 == 0)
      std::printf("[build] progress %d / %d\n", idx, N);
  }

  g.medoid = medoid;
  std::printf("[build] Vamana plain done.\n");
  return g;
}

}  // namespace bang_repro::plain
