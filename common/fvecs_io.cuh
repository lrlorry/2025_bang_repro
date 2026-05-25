#pragma once
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <string>

// ── fvecs / bvecs / ivecs 读取器 ──────────────────────────────────────────
// SIFT1M 标准格式：每条向量前 4 字节为 int32 维度，后接 dim 个元素

template<typename T>
static std::vector<T> load_vecs(const char* path, int& N, int& dim, int max_N = -1)
{
  FILE* f = std::fopen(path, "rb");
  if (!f) {
    std::fprintf(stderr, "Cannot open %s\n", path);
    std::exit(1);
  }

  // 读第一个向量的维度
  int d = 0;
  if (std::fread(&d, sizeof(int), 1, f) != 1) {
    std::fprintf(stderr, "Failed to read dim from %s\n", path);
    std::exit(1);
  }
  dim = d;

  // 每条记录大小（字节）
  long long rec = sizeof(int) + (long long)d * sizeof(T);

  // 统计总向量数
  std::fseek(f, 0, SEEK_END);
  long long fsz = std::ftell(f);
  int total_N = (int)(fsz / rec);
  N = (max_N > 0 && max_N < total_N) ? max_N : total_N;

  std::vector<T> data((long long)N * d);
  std::fseek(f, 0, SEEK_SET);

  for (int i = 0; i < N; i++) {
    int dim_check;
    if (std::fread(&dim_check, sizeof(int), 1, f) != 1) break;
    if (std::fread(data.data() + (long long)i * d, sizeof(T), d, f) != (size_t)d) break;
  }
  std::fclose(f);
  return data;
}

// 便捷函数
static std::vector<float> load_fvecs(const char* p, int& N, int& d, int max_N=-1)
{ return load_vecs<float>(p, N, d, max_N); }

static std::vector<uint8_t> load_bvecs(const char* p, int& N, int& d, int max_N=-1)
{ return load_vecs<uint8_t>(p, N, d, max_N); }

static std::vector<int> load_ivecs(const char* p, int& N, int& d, int max_N=-1)
{ return load_vecs<int>(p, N, d, max_N); }
