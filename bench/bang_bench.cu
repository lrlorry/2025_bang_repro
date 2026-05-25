// bench/bang_bench.cu
// BANG SIFT1M benchmark — GPU PQ search, outputs ONE CSV line: L,recall,qps,search_ms
//
// Pass --graph PATH to skip Vamana build and load pre-built graph from bang_build.
//
// Usage:
//   bang_bench_L32 --base  sift1m_data/sift_base.fvecs \
//                  --query sift1m_data/sift_query.fvecs \
//                  --gt    sift1m_data/sift_groundtruth.ivecs \
//                  --graph results/sift1m_graph.bin \
//                  --N 1000000

#include "common/cuda_utils.cuh"
#include "common/fvecs_io.cuh"
#include "common/graph_io.cuh"
#include "engineered/config.cuh"
#include "engineered/engineered_build.cuh"
#include "engineered/engineered_search.cuh"
#include "plain/plain_search.cuh"

#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <vector>
#include <random>
#include <numeric>
#include <algorithm>
#include <chrono>
#include <omp.h>

using namespace bang_repro::engineered;
using bang_repro::plain::HostGraph;
using bang_repro::plain::DevicePQ;

// ── PQ codebook: k-means training (n_iter iterations, OpenMP assignment) ─────
// Random sampling gives recall~1% on SIFT1M; k-means gives recall~50-80%.
static void build_pq_kmeans(
    const std::vector<float>& vecs, int N, int dim,
    int M, int chunk_dim,
    std::vector<uint8_t>& h_codes,
    std::vector<float>&   h_table,
    std::mt19937& rng,
    int n_iter = 10)
{
  h_table.resize((long long)M * 256 * chunk_dim);
  h_codes.resize((long long)N * M);

  // per-chunk k-means: centroids fit in cache (256×16 floats = 16KB)
  std::vector<int>   assign(N);
  std::vector<float> new_cents(256 * chunk_dim);
  std::vector<int>   cnt(256);

  for (int c = 0; c < M; c++) {
    float* cents = h_table.data() + (long long)c * 256 * chunk_dim;

    // Init centroids: shuffle and pick first 256
    std::vector<int> idx(N);
    std::iota(idx.begin(), idx.end(), 0);
    std::shuffle(idx.begin(), idx.end(), rng);
    for (int k = 0; k < 256; k++) {
      const float* src = vecs.data() + (long long)idx[k] * dim + c * chunk_dim;
      std::copy(src, src + chunk_dim, cents + k * chunk_dim);
    }

    for (int iter = 0; iter < n_iter; iter++) {
      // Assignment (parallel)
      #pragma omp parallel for schedule(dynamic, 2000)
      for (int i = 0; i < N; i++) {
        const float* v = vecs.data() + (long long)i * dim + c * chunk_dim;
        float best_d = 1e30f; int best_k = 0;
        for (int k = 0; k < 256; k++) {
          const float* ct = cents + k * chunk_dim;
          float d = 0.f;
          for (int d_ = 0; d_ < chunk_dim; d_++) { float diff = v[d_]-ct[d_]; d += diff*diff; }
          if (d < best_d) { best_d = d; best_k = k; }
        }
        assign[i] = best_k;
      }
      // Update centroids (sequential, fast)
      std::fill(new_cents.begin(), new_cents.end(), 0.f);
      std::fill(cnt.begin(), cnt.end(), 0);
      for (int i = 0; i < N; i++) {
        int k = assign[i]; cnt[k]++;
        const float* v = vecs.data() + (long long)i * dim + c * chunk_dim;
        for (int d_ = 0; d_ < chunk_dim; d_++) new_cents[k * chunk_dim + d_] += v[d_];
      }
      for (int k = 0; k < 256; k++)
        if (cnt[k] > 0)
          for (int d_ = 0; d_ < chunk_dim; d_++)
            cents[k * chunk_dim + d_] = new_cents[k * chunk_dim + d_] / cnt[k];
    }
    // Final codes from last assignment
    for (int i = 0; i < N; i++) h_codes[(long long)i * M + c] = (uint8_t)assign[i];
    std::fprintf(stderr, "[pq_kmeans] chunk %d/%d done\n", c+1, M);
  }
}

// (kept for reference, replaced by build_pq_kmeans)
static void build_pq_sampled_unused(
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

// ── Brute-force GT for truncated N (avoids gt-ID mismatch) ──────────────────
static std::vector<int> brute_force_gt(
    const std::vector<float>& base, const std::vector<float>& queries,
    int N, int numQ, int dim, int topK)
{
  std::vector<int> gt(numQ * topK);
  for (int q = 0; q < numQ; q++) {
    const float* qv = queries.data() + (long long)q * dim;
    std::vector<std::pair<float,int>> dists(N);
    for (int i = 0; i < N; i++) {
      const float* v = base.data() + (long long)i * dim;
      float d = 0.f;
      for (int d_ = 0; d_ < dim; d_++) { float diff = qv[d_]-v[d_]; d += diff*diff; }
      dists[i] = {d, i};
    }
    std::partial_sort(dists.begin(), dists.begin()+topK, dists.end());
    for (int k = 0; k < topK; k++) gt[q*topK+k] = dists[k].second;
  }
  return gt;
}

int main(int argc, char** argv)
{
  const char* base_path  = "sift1m_data/sift_base.fvecs";
  const char* query_path = "sift1m_data/sift_query.fvecs";
  const char* gt_path    = "sift1m_data/sift_groundtruth.ivecs";
  const char* graph_path = nullptr;   // --graph: load pre-built graph (skip build)
  int  max_N     = 1000000;
  bool local_gt  = false;
  int  max_numQ  = 1000;

  for (int i = 1; i < argc; i++) {
    if      (!strcmp(argv[i], "--base")     && i+1 < argc) base_path  = argv[++i];
    else if (!strcmp(argv[i], "--query")    && i+1 < argc) query_path = argv[++i];
    else if (!strcmp(argv[i], "--gt")       && i+1 < argc) gt_path    = argv[++i];
    else if (!strcmp(argv[i], "--graph")    && i+1 < argc) graph_path = argv[++i];
    else if (!strcmp(argv[i], "--N")        && i+1 < argc) max_N      = std::atoi(argv[++i]);
    else if (!strcmp(argv[i], "--local-gt"))                local_gt   = true;
    else if (!strcmp(argv[i], "--numQ")     && i+1 < argc) max_numQ   = std::atoi(argv[++i]);
  }

  // ── Load data ───────────────────────────────────────────────────────────────
  int N = 0, dim = 0, numQ = 0, qdim = 0;

  std::fprintf(stderr, "[bang_bench] Loading base   %s (max %d)\n", base_path, max_N);
  auto h_base    = load_fvecs(base_path,  N,   dim,  max_N);
  std::fprintf(stderr, "[bang_bench] Loading query  %s\n", query_path);
  auto h_queries = load_fvecs(query_path, numQ, qdim, local_gt ? max_numQ : -1);

  // ── Ground truth ─────────────────────────────────────────────────────────────
  std::vector<int> h_gt;
  int gt_k = kTopK;
  if (local_gt) {
    std::fprintf(stderr, "[bang_bench] Computing brute-force GT (N=%d numQ=%d) ...\n", N, numQ);
    h_gt = brute_force_gt(h_base, h_queries, N, numQ, dim, kTopK);
    std::fprintf(stderr, "[bang_bench] GT done.\n");
  } else {
    int gt_N = 0;
    std::fprintf(stderr, "[bang_bench] Loading GT %s\n", gt_path);
    h_gt = load_ivecs(gt_path, gt_N, gt_k);
    numQ = std::min(numQ, gt_N);
  }

  std::fprintf(stderr, "[bang_bench] N=%d dim=%d numQ=%d gt_k=%d kL=%d\n",
               N, dim, numQ, gt_k, kL);

  const int M = kM, chunk_dim = kChunkDim, topK = kTopK, R = kR;

  // ── Vamana graph: load from file or build ───────────────────────────────────
  bang_repro::plain::HostGraph graph;
  if (graph_path) {
    std::fprintf(stderr, "[bang_bench] Loading graph from %s ...\n", graph_path);
    graph = load_graph(graph_path);
    graph.vecs = h_base;   // reattach full-precision vectors for exact rerank
  } else {
    std::fprintf(stderr, "[bang_bench] Building Vamana (R=%d alpha=1.2 L_build=64) ...\n", R);
    auto t0 = std::chrono::high_resolution_clock::now();
    graph = build_vamana_engineered(h_base, N, dim, R, 1.2f, 64);
    double build_s = std::chrono::duration<double>(
        std::chrono::high_resolution_clock::now() - t0).count();
    std::fprintf(stderr, "[bang_bench] Vamana done in %.1fs, medoid=%d\n", build_s, graph.medoid);
  }

  // ── Build PQ ────────────────────────────────────────────────────────────────
  std::fprintf(stderr, "[bang_bench] Building PQ (M=%d chunk_dim=%d) ...\n", M, chunk_dim);
  std::mt19937 rng(54321);
  std::vector<uint8_t> h_codes;
  std::vector<float>   h_table;
  build_pq_kmeans(h_base, N, dim, M, chunk_dim, h_codes, h_table, rng, /*n_iter=*/10);
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
