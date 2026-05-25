// bench/bang_bench.cu
// BANG SIFT1M benchmark — builds Vamana graph (OpenMP), runs GPU PQ search,
// outputs ONE CSV line to stdout: L,recall,qps,search_ms
//
// Build with one of:
//   bang_bench_L16   (-DkL_OVERRIDE=16)
//   bang_bench_L32   (-DkL_OVERRIDE=32)
//   bang_bench_L48   (-DkL_OVERRIDE=48)
//   bang_bench_L64   (-DkL_OVERRIDE=64)
//
// Usage:
//   bang_bench_L32 --base sift1m_data/sift_base.fvecs \
//                  --query sift1m_data/sift_query.fvecs \
//                  --gt    sift1m_data/sift_groundtruth.ivecs \
//                  --N     100000

#include "common/cuda_utils.cuh"
#include "common/fvecs_io.cuh"
#include "engineered/config.cuh"
#include "engineered/engineered_build.cuh"
#include "engineered/engineered_search.cuh"
#include "plain/plain_search.cuh"

#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <vector>
#include <random>
#include <algorithm>
#include <chrono>

using namespace bang_repro::engineered;
using bang_repro::plain::HostGraph;
using bang_repro::plain::DevicePQ;

// ── PQ codebook: random samples from dataset as centroids ────────────────────
// Better recall than pure Gaussian random, no expensive k-means
static void build_pq_sampled(
    const std::vector<float>& vecs, int N, int dim,
    int M, int chunk_dim,
    std::vector<uint8_t>& h_codes,
    std::vector<float>&   h_table,
    std::mt19937& rng)
{
  std::uniform_int_distribution<int> ud(0, N - 1);
  h_table.resize((long long)M * 256 * chunk_dim);

  for (int c = 0; c < M; c++) {
    for (int k = 0; k < 256; k++) {
      int idx = ud(rng);
      const float* src = vecs.data() + (long long)idx * dim + c * chunk_dim;
      float*       dst = h_table.data() + ((long long)c * 256 + k) * chunk_dim;
      for (int d = 0; d < chunk_dim; d++) dst[d] = src[d];
    }
  }

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

// ── Recall@topK using ANNS ground truth ────────────────────────────────────
static float compute_recall(
    const std::vector<int>& gt,   // [numQ * gt_k]
    const std::vector<int>& pred, // [numQ * topK]
    int numQ, int gt_k, int topK)
{
  int hit = 0, total = numQ * topK;
  for (int q = 0; q < numQ; q++) {
    for (int k = 0; k < topK; k++) {
      int id = pred[q * topK + k];
      int lim = std::min(topK, gt_k);
      for (int g = 0; g < lim; g++) {
        if (gt[q * gt_k + g] == id) { hit++; break; }
      }
    }
  }
  return (float)hit / total;
}

int main(int argc, char** argv)
{
  const char* base_path  = "sift1m_data/sift_base.fvecs";
  const char* query_path = "sift1m_data/sift_query.fvecs";
  const char* gt_path    = "sift1m_data/sift_groundtruth.ivecs";
  int max_N = 100000;

  for (int i = 1; i < argc; i++) {
    if      (!strcmp(argv[i], "--base")  && i+1 < argc) base_path  = argv[++i];
    else if (!strcmp(argv[i], "--query") && i+1 < argc) query_path = argv[++i];
    else if (!strcmp(argv[i], "--gt")    && i+1 < argc) gt_path    = argv[++i];
    else if (!strcmp(argv[i], "--N")     && i+1 < argc) max_N      = std::atoi(argv[++i]);
  }

  // ── Load data ───────────────────────────────────────────────────────────────
  int N = 0, dim = 0, numQ = 0, qdim = 0, gt_N = 0, gt_k = 0;

  std::fprintf(stderr, "[bang_bench] Loading base   %s (max %d)\n", base_path,  max_N);
  auto h_base    = load_fvecs(base_path,  N,    dim,  max_N);
  std::fprintf(stderr, "[bang_bench] Loading query  %s\n", query_path);
  auto h_queries = load_fvecs(query_path, numQ, qdim);
  std::fprintf(stderr, "[bang_bench] Loading GT     %s\n", gt_path);
  auto h_gt      = load_ivecs(gt_path,    gt_N, gt_k);

  numQ = std::min(numQ, gt_N);   // align to available GT
  std::fprintf(stderr, "[bang_bench] N=%d dim=%d numQ=%d gt_k=%d kL=%d\n",
               N, dim, numQ, gt_k, kL);

  const int M = kM, chunk_dim = kChunkDim, topK = kTopK, R = kR;

  // ── Build Vamana graph ──────────────────────────────────────────────────────
  std::fprintf(stderr, "[bang_bench] Building Vamana (R=%d alpha=1.2 L_build=64) ...\n", R);
  auto t_build0 = std::chrono::high_resolution_clock::now();
  auto graph = build_vamana_engineered(h_base, N, dim, R, 1.2f, 64);
  auto t_build1 = std::chrono::high_resolution_clock::now();
  double build_s = std::chrono::duration<double>(t_build1 - t_build0).count();
  std::fprintf(stderr, "[bang_bench] Vamana done in %.2fs, medoid=%d\n", build_s, graph.medoid);

  // ── Build PQ ────────────────────────────────────────────────────────────────
  std::fprintf(stderr, "[bang_bench] Building PQ (M=%d chunk_dim=%d) ...\n", M, chunk_dim);
  std::mt19937 rng(54321);
  std::vector<uint8_t> h_codes;
  std::vector<float>   h_table;
  build_pq_sampled(h_base, N, dim, M, chunk_dim, h_codes, h_table, rng);
  std::fprintf(stderr, "[bang_bench] PQ done.\n");

  // ── Upload PQ to GPU ────────────────────────────────────────────────────────
  DevicePQ pq;
  pq.N = N; pq.M = M; pq.chunk_dim = chunk_dim;
  CUDA_CHECK(cudaMalloc(&pq.d_codes, (long long)N * M * sizeof(uint8_t)));
  CUDA_CHECK(cudaMalloc(&pq.d_table, (long long)M * 256 * chunk_dim * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(pq.d_codes, h_codes.data(), (long long)N * M,
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(pq.d_table, h_table.data(),
                        (long long)M * 256 * chunk_dim * sizeof(float),
                        cudaMemcpyHostToDevice));

  float* d_queries   = nullptr;
  int*   d_out_ids   = nullptr;
  float* d_out_dists = nullptr;
  CUDA_CHECK(cudaMalloc(&d_queries,   (long long)numQ * dim  * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_out_ids,   (long long)numQ * topK * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_out_dists, (long long)numQ * topK * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_queries, h_queries.data(),
                        (long long)numQ * dim * sizeof(float), cudaMemcpyHostToDevice));

  // ── Warmup ──────────────────────────────────────────────────────────────────
  std::fprintf(stderr, "[bang_bench] Warmup ...\n");
  search_bang_engineered(graph, pq, d_queries, numQ, dim, d_out_ids, d_out_dists);
  CUDA_CHECK(cudaDeviceSynchronize());

  // ── Timed search (5 repeats) ────────────────────────────────────────────────
  const int NREP = 5;
  auto t0 = std::chrono::high_resolution_clock::now();
  for (int rep = 0; rep < NREP; rep++)
    search_bang_engineered(graph, pq, d_queries, numQ, dim, d_out_ids, d_out_dists);
  CUDA_CHECK(cudaDeviceSynchronize());
  auto t1 = std::chrono::high_resolution_clock::now();
  double search_ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / NREP;
  double qps = numQ / (search_ms / 1000.0);

  // ── Recall ──────────────────────────────────────────────────────────────────
  std::vector<int> h_out(numQ * topK);
  CUDA_CHECK(cudaMemcpy(h_out.data(), d_out_ids,
                        (long long)numQ * topK * sizeof(int), cudaMemcpyDeviceToHost));
  float recall = compute_recall(h_gt, h_out, numQ, gt_k, topK);

  std::fprintf(stderr, "[bang_bench] L=%d  Recall@%d=%.4f  QPS=%.0f  search_ms=%.3f\n",
               kL, topK, recall, qps, search_ms);

  // ── CSV output (stdout, no header — bench.sh adds header) ───────────────────
  std::printf("%d,%.6f,%.2f,%.3f\n", kL, recall, qps, search_ms);

  cudaFree(pq.d_codes); cudaFree(pq.d_table);
  cudaFree(d_queries); cudaFree(d_out_ids); cudaFree(d_out_dists);
  return 0;
}
