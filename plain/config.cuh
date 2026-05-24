#pragma once

namespace bang_repro::plain {

// ── 数据集参数 ────────────────────────────────────────────────────────────────
constexpr int kN       = 8192;   // 数据集大小（self-contained demo 用小值）
constexpr int kDim     = 128;    // 向量维度
constexpr int kNumQ    = 16;     // query batch size

// ── 图参数（对应 BANG MAX_R=64，demo 用较小值） ─────────────────────────────
constexpr int kR       = 16;     // graph degree

// ── PQ 参数 ──────────────────────────────────────────────────────────────────
constexpr int kM       = 8;      // PQ chunks（uChunks）
constexpr int kPQCents = 256;    // centroids per chunk（BANG 固定 256）
constexpr int kChunkDim = kDim / kM;  // 每个 chunk 的维度

// ── 搜索参数 ─────────────────────────────────────────────────────────────────
constexpr int kL       = 32;     // worklist length（BANG: worklist_length）
constexpr int kTopK    = 10;     // 最终输出 top-k
constexpr int kBF      = 399887; // Bloom filter entries/query（BANG: BF_ENTRIES）
constexpr int kMaxIter = 300;    // 最大迭代轮数

}  // namespace bang_repro::plain
