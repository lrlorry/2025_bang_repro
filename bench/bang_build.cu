// bench/bang_build.cu
// Build Vamana graph from fvecs file once and save to disk.
// bang_bench_LXX can then load the pre-built graph instead of rebuilding.
//
// Usage:
//   bang_build --base sift1m_data/sift_base.fvecs \
//              --N 1000000 \
//              --out results/sift1m_graph.bin

#include "common/fvecs_io.cuh"
#include "common/graph_io.cuh"
#include "engineered/config.cuh"
#include "engineered/engineered_build.cuh"

#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <chrono>

int main(int argc, char** argv)
{
  const char* base_path = "sift1m_data/sift_base.fvecs";
  const char* out_path  = "results/sift1m_graph.bin";
  int max_N = 1000000;

  for (int i = 1; i < argc; i++) {
    if      (!strcmp(argv[i], "--base") && i+1 < argc) base_path = argv[++i];
    else if (!strcmp(argv[i], "--out")  && i+1 < argc) out_path  = argv[++i];
    else if (!strcmp(argv[i], "--N")    && i+1 < argc) max_N     = std::atoi(argv[++i]);
  }

  int N = 0, dim = 0;
  std::fprintf(stderr, "[bang_build] Loading %s (max %d) ...\n", base_path, max_N);
  auto h_base = load_fvecs(base_path, N, dim, max_N);
  std::fprintf(stderr, "[bang_build] N=%d dim=%d\n", N, dim);

  const int R = bang_repro::engineered::kR;
  std::fprintf(stderr, "[bang_build] Building Vamana (R=%d alpha=1.2 L_build=64, OpenMP) ...\n", R);
  auto t0 = std::chrono::high_resolution_clock::now();
  auto graph = bang_repro::engineered::build_vamana_engineered(h_base, N, dim, R, 1.2f, 64);
  auto t1 = std::chrono::high_resolution_clock::now();
  double elapsed = std::chrono::duration<double>(t1 - t0).count();
  std::fprintf(stderr, "[bang_build] Done in %.1fs  medoid=%d\n", elapsed, graph.medoid);

  save_graph(graph, dim, out_path);
  return 0;
}
