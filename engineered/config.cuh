#pragma once

namespace bang_repro::engineered {

constexpr int kN       = 8192;
constexpr int kDim     = 128;
constexpr int kNumQ    = 16;
constexpr int kR       = 16;     // graph degree
constexpr int kM       = 8;      // PQ chunks
constexpr int kPQCents = 256;
constexpr int kChunkDim = kDim / kM;
constexpr int kL       = 32;     // worklist length
constexpr int kTopK    = 10;
constexpr int kBF      = 399887; // Bloom filter entries/query (BANG: BF_ENTRIES)
constexpr int kMaxIter = 300;

// ── engineered 专有参数 ───────────────────────────────────────────────────────
// 对应 BANG bang_search.cu::THREADS_PER_NEIGHBOR=8
// 一个 neighbor 用 kTPNbr 个线程并行做 PQ chunk sum（warp reduce）
constexpr int kTPNbr   = 8;

// 每个 query 的 worklist merge block size：2*kL 个线程（BANG: 2*L <= 1024 约束）
// plain 版 serial insert 无此限制；engineered 版需满足 2*kL <= 1024
static_assert(2 * kL <= 1024, "2*kL must fit in one CUDA block");

}  // namespace bang_repro::engineered
