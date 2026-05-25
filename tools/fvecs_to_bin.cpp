// tools/fvecs_to_bin.cpp
// Convert fvecs/bvecs/ivecs to DiskANN binary format:
//   [N: int32][dim: int32][data: N * dim * sizeof(T)]
//
// Usage:
//   fvecs_to_bin sift_base.fvecs sift_base.bin float
//   fvecs_to_bin sift_query.fvecs sift_query.bin float

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

int main(int argc, char** argv)
{
  if (argc < 4) {
    std::fprintf(stderr, "Usage: %s input.fvecs output.bin [float|uint8|int]\n", argv[0]);
    return 1;
  }
  const char* in_path  = argv[1];
  const char* out_path = argv[2];
  // dtype arg[3] is informational only — we read raw bytes from the file

  FILE* fin = std::fopen(in_path, "rb");
  if (!fin) { std::perror(in_path); return 1; }

  // Read dimension from first record header
  int dim = 0;
  if (std::fread(&dim, sizeof(int), 1, fin) != 1) {
    std::fprintf(stderr, "Failed to read dim\n"); return 1;
  }

  // Determine element size from dtype string
  size_t elem_sz = 4;
  if (!strcmp(argv[3], "uint8")) elem_sz = 1;

  long long rec = sizeof(int) + (long long)dim * elem_sz;
  std::fseek(fin, 0, SEEK_END);
  long long fsz = std::ftell(fin);
  int N = (int)(fsz / rec);
  std::fseek(fin, 0, SEEK_SET);

  std::fprintf(stderr, "[fvecs_to_bin] N=%d dim=%d elem_sz=%zu\n", N, dim, elem_sz);

  FILE* fout = std::fopen(out_path, "wb");
  if (!fout) { std::perror(out_path); return 1; }

  // DiskANN bin header
  std::fwrite(&N,   sizeof(int), 1, fout);
  std::fwrite(&dim, sizeof(int), 1, fout);

  std::vector<char> buf(dim * elem_sz);
  for (int i = 0; i < N; i++) {
    int dim_check;
    std::fread(&dim_check, sizeof(int), 1, fin);
    std::fread(buf.data(), elem_sz, dim, fin);
    std::fwrite(buf.data(), elem_sz, dim, fout);
  }

  std::fclose(fin);
  std::fclose(fout);
  std::fprintf(stderr, "[fvecs_to_bin] Done → %s\n", out_path);
  return 0;
}
