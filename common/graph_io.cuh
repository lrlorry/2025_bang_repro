#pragma once
#include "plain/plain_search.cuh"
#include <cstdio>
#include <cstdlib>
#include <vector>

// Binary graph file format:
//   [N (int32)][dim (int32)][R (int32)][medoid (int32)]
//   [adj: N*R int32]
// Vectors are NOT stored — caller reloads from original fvecs file.

static void save_graph(const bang_repro::plain::HostGraph& g, int dim, const char* path)
{
  FILE* f = std::fopen(path, "wb");
  if (!f) { std::fprintf(stderr, "Cannot write %s\n", path); std::exit(1); }
  int hdr[4] = { g.N, dim, g.R, g.medoid };
  std::fwrite(hdr, sizeof(int), 4, f);
  std::fwrite(g.adj.data(), sizeof(int), (long long)g.N * g.R, f);
  std::fclose(f);
  std::fprintf(stderr, "[graph_io] Saved  %s  (N=%d R=%d medoid=%d)\n",
               path, g.N, g.R, g.medoid);
}

// Loads adj + metadata; caller must populate g.vecs separately.
static bang_repro::plain::HostGraph load_graph(const char* path)
{
  FILE* f = std::fopen(path, "rb");
  if (!f) { std::fprintf(stderr, "Cannot open graph %s\n", path); std::exit(1); }
  int hdr[4];
  if (std::fread(hdr, sizeof(int), 4, f) != 4) {
    std::fprintf(stderr, "Corrupt graph header in %s\n", path); std::exit(1);
  }
  int N = hdr[0], R = hdr[2], medoid = hdr[3];
  bang_repro::plain::HostGraph g;
  g.N = N; g.R = R; g.medoid = medoid;
  g.adj.resize((long long)N * R);
  if (std::fread(g.adj.data(), sizeof(int), (long long)N * R, f) != (size_t)((long long)N * R)) {
    std::fprintf(stderr, "Corrupt graph data in %s\n", path); std::exit(1);
  }
  std::fclose(f);
  std::fprintf(stderr, "[graph_io] Loaded %s  (N=%d R=%d medoid=%d)\n",
               path, N, R, medoid);
  return g;
}
