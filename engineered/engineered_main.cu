#include "common/cuda_utils.cuh"
#include "engineered/config.cuh"
#include "engineered/engineered_build.cuh"
#include "engineered/engineered_search.cuh"
#include "plain/plain_search.cuh"

#include <cstdio>
#include <random>
#include <vector>
#include <algorithm>
#include <numeric>

static void make_random_pq(
    const std::vector<float>& vecs, int N, int dim, int M, int chunk_dim,
    std::vector<uint8_t>& h_codes, std::vector<float>& h_table, std::mt19937& rng)
{
  std::normal_distribution<float> normal(0.f, 1.f);
  h_table.resize((long long)M * 256 * chunk_dim);
  for (float& x : h_table) x = normal(rng);
  h_codes.resize((long long)N * M);
  for (int i = 0; i < N; i++) {
    const float* v = vecs.data() + (long long)i * dim;
    for (int c = 0; c < M; c++) {
      float best_d = 1e30f; int best_k = 0;
      const float* vc = v + c * chunk_dim;
      for (int k = 0; k < 256; k++) {
        const float* cent = h_table.data() + ((long long)c * 256 + k) * chunk_dim;
        float d = 0.f;
        for (int d_ = 0; d_ < chunk_dim; d_++) { float diff = vc[d_] - cent[d_]; d += diff*diff; }
        if (d < best_d) { best_d = d; best_k = k; }
      }
      h_codes[i * M + c] = (uint8_t)best_k;
    }
  }
}

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
      for (int d_ = 0; d_ < dim; d_++) { float diff = qv[d_] - v[d_]; d += diff*diff; }
      dists[i] = {d, i};
    }
    std::partial_sort(dists.begin(), dists.begin() + topK, dists.end());
    for (int k = 0; k < topK; k++) result[q * topK + k] = dists[k].second;
  }
  return result;
}

int main()
{
  using namespace bang_repro::engineered;
  std::mt19937 rng(42);
  std::normal_distribution<float> normal(0.f, 1.f);

  const int N = kN, dim = kDim, numQ = kNumQ, R = kR, M = kM;

  std::printf("BANG engineered reproduction (Vamana build + GPU PQ search)\n");
  std::printf("N=%d, dim=%d, numQ=%d, R=%d, M=%d, L=%d, topK=%d\n", N, dim, numQ, R, M, kL, kTopK);
  std::printf("Build: OpenMP parallel greedy_search + robust_prune\n");
  std::printf("Search: transposed PQ table, 8-thread PQ dist, shared-mem merge, 4 streams, OpenMP\n\n");

  std::vector<float> h_dataset(N * dim), h_queries(numQ * dim);
  for (float& x : h_dataset) x = normal(rng);
  for (float& x : h_queries) x = normal(rng);

  std::printf("Building Vamana graph (GPU batch L2 + OpenMP prune)...\n");
  auto graph = bang_repro::engineered::build_vamana_engineered(
      h_dataset, N, dim, R, /*alpha=*/1.2f, /*L_build=*/64);

  std::printf("Building PQ (GPU HBM)...\n");
  std::vector<uint8_t> h_codes;
  std::vector<float>   h_table;
  make_random_pq(h_dataset, N, dim, M, kChunkDim, h_codes, h_table, rng);

  bang_repro::plain::DevicePQ pq;
  pq.N = N; pq.M = M; pq.chunk_dim = kChunkDim;
  CUDA_CHECK(cudaMalloc(&pq.d_codes, (long long)N * M * sizeof(uint8_t)));
  CUDA_CHECK(cudaMalloc(&pq.d_table, (long long)M * 256 * kChunkDim * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(pq.d_codes, h_codes.data(), (long long)N * M, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(pq.d_table, h_table.data(), (long long)M * 256 * kChunkDim * sizeof(float), cudaMemcpyHostToDevice));

  float* d_queries   = nullptr;
  int*   d_out_ids   = nullptr;
  float* d_out_dists = nullptr;
  CUDA_CHECK(cudaMalloc(&d_queries,   (long long)numQ * dim  * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_out_ids,   (long long)numQ * kTopK * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_out_dists, (long long)numQ * kTopK * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_queries, h_queries.data(), numQ * dim * sizeof(float), cudaMemcpyHostToDevice));

  std::printf("Running BANG engineered search...\n");
  search_bang_engineered(graph, pq, d_queries, numQ, dim, d_out_ids, d_out_dists);

  std::vector<int>   h_out_ids(numQ * kTopK);
  std::vector<float> h_out_dists(numQ * kTopK);
  CUDA_CHECK(cudaMemcpy(h_out_ids.data(),   d_out_ids,   numQ * kTopK * sizeof(int),   cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(h_out_dists.data(), d_out_dists, numQ * kTopK * sizeof(float), cudaMemcpyDeviceToHost));

  std::printf("Computing ground truth...\n");
  auto gt = brute_force_topk(h_dataset, h_queries, N, numQ, dim, kTopK);

  int hit = 0, total = numQ * kTopK;
  for (int q = 0; q < numQ; q++)
    for (int k = 0; k < kTopK; k++) {
      int gt_id = gt[q * kTopK + k];
      for (int k2 = 0; k2 < kTopK; k2++)
        if (h_out_ids[q * kTopK + k2] == gt_id) { hit++; break; }
    }

  std::printf("\nResults (first 4 queries):\n");
  for (int q = 0; q < std::min(numQ, 4); q++) {
    std::printf("  query %2d: ", q);
    for (int k = 0; k < kTopK; k++)
      std::printf("%d(%.2f) ", h_out_ids[q * kTopK + k], h_out_dists[q * kTopK + k]);
    std::printf("\n");
  }
  std::printf("\nRecall@%d: %.3f  (%d / %d)\n", kTopK, (float)hit/total, hit, total);

  cudaFree(pq.d_codes); cudaFree(pq.d_table);
  cudaFree(d_queries); cudaFree(d_out_ids); cudaFree(d_out_dists);
  return 0;
}
