// tools/diskann_to_bang.cpp
// Convert DiskANN in-memory index graph file → bang_repro graph.bin
//
// DiskANN graph file format (.graph):
//   [index_size: uint64]      — total bytes of adjacency data
//   [max_degree: uint32]      — R (max out-degree)
//   for each node i (0 to N-1):
//     [degree_i: uint32]
//     [neighbor_0 ... neighbor_{degree_i-1}: uint32]
//
// Our graph.bin format (common/graph_io.cuh):
//   [N: int32][dim: int32][R: int32][medoid: int32]
//   [adj: N * R int32]   (-1 = empty slot)
//
// Usage:
//   diskann_to_bang --graph  sift_index.graph \
//                   --base   sift_base.fvecs   \
//                   --N      1000000            \
//                   --out    results/sift1m_graph.bin

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <numeric>
#include <algorithm>
#include <cmath>

// ── fvecs loader (inline, no external deps) ──────────────────────────────────
static std::vector<float> load_fvecs_simple(const char* path, int& N, int& dim, int max_N = -1)
{
  FILE* f = std::fopen(path, "rb");
  if (!f) { std::fprintf(stderr, "Cannot open %s\n", path); std::exit(1); }
  int d = 0;
  std::fread(&d, sizeof(int), 1, f);
  dim = d;
  long long rec = sizeof(int) + (long long)d * sizeof(float);
  std::fseek(f, 0, SEEK_END);
  long long fsz = std::ftell(f);
  int total_N = (int)(fsz / rec);
  N = (max_N > 0 && max_N < total_N) ? max_N : total_N;
  std::vector<float> data((long long)N * d);
  std::fseek(f, 0, SEEK_SET);
  for (int i = 0; i < N; i++) {
    int dc; std::fread(&dc, sizeof(int), 1, f);
    std::fread(data.data() + (long long)i * d, sizeof(float), d, f);
  }
  std::fclose(f);
  return data;
}

// ── compute medoid (centroid → nearest neighbor) ─────────────────────────────
static int compute_medoid(const std::vector<float>& vecs, int N, int dim)
{
  // centroid
  std::vector<double> centroid(dim, 0.0);
  for (int i = 0; i < N; i++)
    for (int d = 0; d < dim; d++)
      centroid[d] += vecs[(long long)i * dim + d];
  for (int d = 0; d < dim; d++) centroid[d] /= N;

  // nearest to centroid
  float best = 1e30f; int best_i = 0;
  for (int i = 0; i < N; i++) {
    float dist = 0.f;
    for (int d = 0; d < dim; d++) {
      float diff = vecs[(long long)i * dim + d] - (float)centroid[d];
      dist += diff * diff;
    }
    if (dist < best) { best = dist; best_i = i; }
  }
  return best_i;
}

int main(int argc, char** argv)
{
  const char* graph_path = nullptr;
  const char* base_path  = nullptr;
  const char* out_path   = "results/sift1m_graph.bin";
  int max_N = -1;

  for (int i = 1; i < argc; i++) {
    if      (!strcmp(argv[i], "--graph") && i+1 < argc) graph_path = argv[++i];
    else if (!strcmp(argv[i], "--base")  && i+1 < argc) base_path  = argv[++i];
    else if (!strcmp(argv[i], "--out")   && i+1 < argc) out_path   = argv[++i];
    else if (!strcmp(argv[i], "--N")     && i+1 < argc) max_N      = std::atoi(argv[++i]);
  }

  if (!graph_path || !base_path) {
    std::fprintf(stderr, "Usage: %s --graph <file> --base <fvecs> [--N <max>] [--out <file>]\n", argv[0]);
    return 1;
  }

  // ── Read DiskANN graph file ─────────────────────────────────────────────────
  FILE* fg = std::fopen(graph_path, "rb");
  if (!fg) { std::perror(graph_path); return 1; }

  uint64_t index_size = 0;
  uint32_t max_degree = 0;
  std::fread(&index_size, sizeof(uint64_t), 1, fg);
  std::fread(&max_degree, sizeof(uint32_t), 1, fg);
  std::fprintf(stderr, "[convert] DiskANN graph: index_size=%lu max_degree=%u\n",
               (unsigned long)index_size, max_degree);

  // Read all adjacency lists
  std::vector<std::vector<uint32_t>> adj_lists;
  adj_lists.reserve(2000000);
  while (!std::feof(fg)) {
    uint32_t degree = 0;
    if (std::fread(&degree, sizeof(uint32_t), 1, fg) != 1) break;
    std::vector<uint32_t> nbrs(degree);
    if (degree > 0)
      std::fread(nbrs.data(), sizeof(uint32_t), degree, fg);
    adj_lists.push_back(std::move(nbrs));
  }
  std::fclose(fg);

  int N_graph = (int)adj_lists.size();
  std::fprintf(stderr, "[convert] Read %d nodes from DiskANN graph (max_degree=%u)\n",
               N_graph, max_degree);

  // ── Load base vectors for medoid computation ─────────────────────────────────
  int N = 0, dim = 0;
  std::fprintf(stderr, "[convert] Loading base vectors %s (max %d) ...\n", base_path, max_N);
  auto vecs = load_fvecs_simple(base_path, N, dim, max_N);
  if (max_N > 0) N = std::min(N, max_N);
  // Clip graph to N nodes (DiskANN may have a frozen entry-point node at index N)
  int N_use = std::min(N, N_graph);
  std::fprintf(stderr, "[convert] N=%d dim=%d N_use=%d\n", N, dim, N_use);

  // ── Compute medoid ──────────────────────────────────────────────────────────
  std::fprintf(stderr, "[convert] Computing medoid ...\n");
  int medoid = compute_medoid(vecs, N_use, dim);
  std::fprintf(stderr, "[convert] medoid=%d\n", medoid);

  // ── Write bang graph.bin ────────────────────────────────────────────────────
  int R = (int)max_degree;
  FILE* fout = std::fopen(out_path, "wb");
  if (!fout) { std::perror(out_path); return 1; }

  int hdr[4] = { N_use, dim, R, medoid };
  std::fwrite(hdr, sizeof(int), 4, fout);

  std::vector<int> row(R, -1);
  for (int i = 0; i < N_use; i++) {
    std::fill(row.begin(), row.end(), -1);
    const auto& nbrs = adj_lists[i];
    int cnt = std::min((int)nbrs.size(), R);
    for (int j = 0; j < cnt; j++) row[j] = (int)nbrs[j];
    std::fwrite(row.data(), sizeof(int), R, fout);
  }
  std::fclose(fout);
  std::fprintf(stderr, "[convert] Saved %s  (N=%d R=%d medoid=%d)\n",
               out_path, N_use, R, medoid);
  return 0;
}
