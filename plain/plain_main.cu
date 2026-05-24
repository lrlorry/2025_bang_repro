#include "common/cuda_utils.cuh"
#include "plain/config.cuh"
#include "plain/plain_build.cuh"
#include "plain/plain_search.cuh"

#include <cstdio>
#include <random>
#include <vector>
#include <algorithm>
#include <numeric>
#include <cmath>

// 生成 random PQ（随机 centroid，用于测试搜索机制，不保证 PQ 质量）
static void make_random_pq(
    const std::vector<float>& vecs, int N, int dim, int M, int chunk_dim,
    std::vector<uint8_t>& h_codes, std::vector<float>& h_table,
    std::mt19937& rng)
{
  std::normal_distribution<float> normal(0.f, 1.f);
  // random centroids: M × 256 × chunk_dim
  h_table.resize((long long)M * 256 * chunk_dim);
  for (float& x : h_table) x = normal(rng);

  // 对每个 vector，找最近 centroid（per chunk）
  h_codes.resize((long long)N * M);
  for (int i = 0; i < N; i++) {
    const float* v = vecs.data() + (long long)i * dim;
    for (int c = 0; c < M; c++) {
      float best_d = 1e30f; int best_k = 0;
      const float* vc = v + c * chunk_dim;
      for (int k = 0; k < 256; k++) {
        const float* cent = h_table.data() + ((long long)c * 256 + k) * chunk_dim;
        float d = 0.f;
        for (int d_ = 0; d_ < chunk_dim; d_++) {
          float diff = vc[d_] - cent[d_]; d += diff * diff;
        }
        if (d < best_d) { best_d = d; best_k = k; }
      }
      h_codes[i * M + c] = (uint8_t)best_k;
    }
  }
}

// 暴力 brute-force 精确 top-k（用于计算 recall）
static std::vector<int> brute_force_topk(
    const std::vector<float>& dataset, const std::vector<float>& queries,
    int N, int numQ, int dim, int topK)
{
  std::vector<int> result(numQ * topK, -1);
  for (int q = 0; q < numQ; q++) {
    const float* qv = queries.data() + (long long)q * dim;
    std::vector<std::pair<float, int>> dists(N);
    for (int i = 0; i < N; i++) {
      const float* v = dataset.data() + (long long)i * dim;
      float d = 0.f;
      for (int d_ = 0; d_ < dim; d_++) { float diff = qv[d_] - v[d_]; d += diff * diff; }
      dists[i] = {d, i};
    }
    std::partial_sort(dists.begin(), dists.begin() + topK, dists.end());
    for (int k = 0; k < topK; k++) result[q * topK + k] = dists[k].second;
  }
  return result;
}

int main()
{
  using namespace bang_repro::plain;
  std::mt19937 rng(42);
  std::normal_distribution<float> normal(0.f, 1.f);

  const int N    = kN;
  const int dim  = kDim;
  const int numQ = kNumQ;
  const int R    = kR;
  const int M    = kM;
  const int chunk_dim = kChunkDim;
  const int topK = kTopK;

  std::printf("BANG plain reproduction (Vamana build + GPU PQ search)\n");
  std::printf("N=%d, dim=%d, numQ=%d, R=%d, M=%d, L=%d, topK=%d\n\n",
              N, dim, numQ, R, M, kL, topK);

  // 生成随机 float 数据集和 query
  std::vector<float> h_dataset(N * dim), h_queries(numQ * dim);
  for (float& x : h_dataset) x = normal(rng);
  for (float& x : h_queries) x = normal(rng);

  // 构建 Vamana 图（CPU Vamana，对应 BANG bang_preprocess.py）
  std::printf("Building Vamana graph (host RAM, plain CPU)...\n");
  HostGraph graph = build_vamana_plain(h_dataset, N, dim, R, /*alpha=*/1.2f, /*L_build=*/64);

  // 构建 PQ（GPU HBM）
  std::printf("Building PQ (GPU HBM)...\n");
  std::vector<uint8_t> h_codes;
  std::vector<float>   h_table;
  make_random_pq(h_dataset, N, dim, M, chunk_dim, h_codes, h_table, rng);

  // 上传到 GPU
  DevicePQ pq;
  pq.N = N; pq.M = M; pq.chunk_dim = chunk_dim;
  CUDA_CHECK(cudaMalloc(&pq.d_codes, (long long)N * M * sizeof(uint8_t)));
  CUDA_CHECK(cudaMalloc(&pq.d_table, (long long)M * 256 * chunk_dim * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(pq.d_codes, h_codes.data(), (long long)N * M * sizeof(uint8_t), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(pq.d_table, h_table.data(), (long long)M * 256 * chunk_dim * sizeof(float), cudaMemcpyHostToDevice));

  float* d_queries  = nullptr;
  int*   d_out_ids  = nullptr;
  float* d_out_dists = nullptr;
  CUDA_CHECK(cudaMalloc(&d_queries,   (long long)numQ * dim  * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_out_ids,   (long long)numQ * topK * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_out_dists, (long long)numQ * topK * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_queries, h_queries.data(), numQ * dim * sizeof(float), cudaMemcpyHostToDevice));

  // 运行 BANG plain 搜索
  std::printf("Running BANG plain search...\n");
  search_bang_plain(graph, pq, d_queries, numQ, dim, d_out_ids, d_out_dists);

  // 取回结果
  std::vector<int>   h_out_ids(numQ * topK);
  std::vector<float> h_out_dists(numQ * topK);
  CUDA_CHECK(cudaMemcpy(h_out_ids.data(),   d_out_ids,   numQ * topK * sizeof(int),   cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(h_out_dists.data(), d_out_dists, numQ * topK * sizeof(float), cudaMemcpyDeviceToHost));

  // 计算 recall@topK 对比暴力搜索
  std::printf("Computing brute-force ground truth...\n");
  auto gt = brute_force_topk(h_dataset, h_queries, N, numQ, dim, topK);

  int hit = 0, total = numQ * topK;
  for (int q = 0; q < numQ; q++) {
    for (int k = 0; k < topK; k++) {
      int gt_id = gt[q * topK + k];
      for (int k2 = 0; k2 < topK; k2++)
        if (h_out_ids[q * topK + k2] == gt_id) { hit++; break; }
    }
  }

  std::printf("\nResults:\n");
  for (int q = 0; q < std::min(numQ, 4); q++) {
    std::printf("  query %2d top-%d: ", q, topK);
    for (int k = 0; k < topK; k++)
      std::printf("%d(%.2f) ", h_out_ids[q * topK + k], h_out_dists[q * topK + k]);
    std::printf("\n");
  }
  std::printf("\nRecall@%d: %.3f  (%d / %d)\n", topK,
              (float)hit / total, hit, total);
  std::printf("\nNote: Vamana graph + random PQ centroids.\n");
  std::printf("      Recall improves significantly with trained PQ codebooks.\n");

  // 清理
  cudaFree(pq.d_codes); cudaFree(pq.d_table);
  cudaFree(d_queries); cudaFree(d_out_ids); cudaFree(d_out_dists);
  return 0;
}
