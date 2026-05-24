#!/usr/bin/env python3
"""
estimate_resources.py
估算 BANG Base 在不同数据集规模下的内存、磁盘、GPU HBM 需求。

用法：
    python3 scripts/estimate_resources.py
    python3 scripts/estimate_resources.py --N 10000000 --dim 128 --dtype float32 --R 64 --B_gb 1.0

参数含义（对应 BANG/DiskANN 参数）：
    N        向量数量（base dataset size）
    dim      向量维度
    dtype    base vector 数据类型（float32=4B, uint8=1B, int8=1B）
    R        graph degree bound（对应 DiskANN -R，BANG MAX_R=64）
    B_gb     DiskANN -B 参数（GiB），控制 PQ 压缩比
             uChunks ≈ B_gb * 1024^3 / N（bytes per vector）
             注意：B_gb 是 PQ 压缩向量总占用的上界，不是单向量大小

输出：
    各数据集规模的资源估算表（打印到 stdout）
"""

import argparse
import math

DTYPE_BYTES = {"float32": 4, "float": 4, "uint8": 1, "int8": 1}

PRESETS = [
    {"name": "SIFT10K",   "N": 10_000,       "dim": 128, "dtype": "float32", "R": 64, "B_gb": 0.0012},
    {"name": "SIFT1M",    "N": 1_000_000,    "dim": 128, "dtype": "float32", "R": 64, "B_gb": 0.1},
    {"name": "SIFT10M",   "N": 10_000_000,   "dim": 128, "dtype": "uint8",   "R": 64, "B_gb": 1.0},
    {"name": "SIFT100M",  "N": 100_000_000,  "dim": 128, "dtype": "uint8",   "R": 64, "B_gb": 10.0},
    {"name": "SIFT1B",    "N": 1_000_000_000,"dim": 128, "dtype": "uint8",   "R": 64, "B_gb": 100.0},
    {"name": "DEEP100M",  "N": 100_000_000,  "dim": 96,  "dtype": "float32", "R": 64, "B_gb": 10.0},
    {"name": "DEEP1B",    "N": 1_000_000_000,"dim": 96,  "dtype": "float32", "R": 64, "B_gb": 64.0},
    {"name": "GIST1M",    "N": 1_000_000,    "dim": 960, "dtype": "float32", "R": 64, "B_gb": 1.0},
]


def fmt_bytes(n_bytes: float) -> str:
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if n_bytes < 1024:
            return f"{n_bytes:.1f} {unit}"
        n_bytes /= 1024
    return f"{n_bytes:.1f} PB"


def estimate(N: int, dim: int, dtype: str, R: int, B_gb: float,
             n_queries: int = 10000, worklist_L: int = 100) -> dict:
    vbytes = DTYPE_BYTES.get(dtype, 4)

    # --- base vectors（host RAM）---
    base_bytes = N * dim * vbytes

    # --- DiskANN graph index（host RAM）---
    # 每个 entry = [full vector | neighbor_count | neighbor_ids]
    # = dim * vbytes + 4 + R * 4
    entry_bytes = dim * vbytes + 4 + R * 4
    # bang_preprocess.py 处理后格式为连续 entry，无 hole
    graph_bytes = N * entry_bytes

    # --- PQ compressed vectors（GPU HBM）---
    # uChunks = B_gb * GiB / N（bytes per vector）
    # 每个 PQ code = uint8，所以 PQ 总大小 = N * uChunks bytes
    B_bytes = B_gb * (1024 ** 3)
    uChunks = max(1, int(B_bytes / N))
    pq_compressed_bytes = N * uChunks  # uint8

    # --- PQ pivots（GPU HBM）---
    # pqTable: 256 * dim * float32
    pq_table_bytes = 256 * dim * 4
    centroid_bytes = dim * 4
    chunk_offsets_bytes = (uChunks + 1) * 4
    pq_gpu_meta_bytes = pq_table_bytes + centroid_bytes + chunk_offsets_bytes

    # --- per-query buffers（GPU HBM）---
    # d_pqDistTables: n_queries * uChunks * 256 * float32
    pq_dist_tables_bytes = n_queries * uChunks * 256 * 4
    # d_BestLSets + dist + visited: n_queries * worklist_L * (4+4+1)
    worklist_bytes = n_queries * worklist_L * (4 + 4 + 1)
    # d_processed_bit_vec: n_queries * BF_ENTRIES * bool（399887 bool ≈ 400KB/query）
    BF_ENTRIES = 399887
    bloom_bytes = n_queries * BF_ENTRIES * 1
    # d_compressedVectors 已含在 pq_compressed_bytes
    # d_neighbors + temp + aux: n_queries * (R+1) * 3 * 4
    neighbor_buf_bytes = n_queries * (R + 1) * 3 * 4
    # d_FPSetCoordsList（候选 full vectors for rerank）: MAX_PARENTS * n_queries * dim * vbytes
    MAX_PARENTS = worklist_L + 50
    fp_set_bytes = MAX_PARENTS * n_queries * dim * vbytes
    # d_L2distances + d_L2ParentIds * 2 each
    l2_buf_bytes = MAX_PARENTS * n_queries * (4 + 4) * 2

    per_query_gpu_bytes = (pq_dist_tables_bytes + worklist_bytes + bloom_bytes +
                           neighbor_buf_bytes + fp_set_bytes + l2_buf_bytes)

    # --- 总 GPU HBM ---
    gpu_total_bytes = pq_compressed_bytes + pq_gpu_meta_bytes + per_query_gpu_bytes

    # --- 总 host RAM（graph + base vectors）---
    # BANG Base 不把 base vectors 单独存，而是嵌在 graph entry 里
    # 所以 host RAM ≈ graph_bytes（已包含 full vectors）
    # 额外：pinned host buffers（较小，忽略）
    host_ram_bytes = graph_bytes

    # --- 磁盘（下载 + index 文件）---
    # 需要存放：base vectors + graph index + pq files
    disk_bytes = base_bytes + graph_bytes + pq_compressed_bytes + pq_gpu_meta_bytes * 2

    return {
        "name": f"N={N:,}",
        "N": N, "dim": dim, "dtype": dtype, "R": R, "B_gb": B_gb,
        "uChunks": uChunks,
        "base_bytes": base_bytes,
        "graph_bytes": graph_bytes,
        "pq_compressed_bytes": pq_compressed_bytes,
        "per_query_gpu_bytes": per_query_gpu_bytes,
        "gpu_total_bytes": gpu_total_bytes,
        "host_ram_bytes": host_ram_bytes,
        "disk_bytes": disk_bytes,
    }


def print_table(results: list[dict]) -> None:
    headers = ["数据集", "N", "dim", "dtype", "uChunks",
               "base vectors", "graph(host)", "PQ compressed(GPU)",
               "per-query GPU buf", "GPU HBM 总计", "host RAM", "磁盘估算"]
    rows = []
    for r in results:
        rows.append([
            r.get("preset_name", r["name"]),
            f"{r['N']:,}",
            str(r["dim"]),
            r["dtype"],
            str(r["uChunks"]),
            fmt_bytes(r["base_bytes"]),
            fmt_bytes(r["graph_bytes"]),
            fmt_bytes(r["pq_compressed_bytes"]),
            fmt_bytes(r["per_query_gpu_bytes"]),
            fmt_bytes(r["gpu_total_bytes"]),
            fmt_bytes(r["host_ram_bytes"]),
            fmt_bytes(r["disk_bytes"]),
        ])

    col_widths = [max(len(h), max(len(row[i]) for row in rows))
                  for i, h in enumerate(headers)]
    fmt = "  ".join(f"{{:<{w}}}" for w in col_widths)
    sep = "  ".join("-" * w for w in col_widths)

    print(fmt.format(*headers))
    print(sep)
    for row in rows:
        print(fmt.format(*row))


def main():
    parser = argparse.ArgumentParser(description="BANG 资源估算")
    parser.add_argument("--N", type=int, help="向量数量（覆盖预设）")
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--dtype", default="float32", choices=list(DTYPE_BYTES))
    parser.add_argument("--R", type=int, default=64)
    parser.add_argument("--B_gb", type=float, default=1.0, help="DiskANN -B 参数（GiB）")
    parser.add_argument("--n_queries", type=int, default=10000)
    parser.add_argument("--worklist_L", type=int, default=100)
    args = parser.parse_args()

    print("====== BANG Base 资源估算 ======")
    print(f"假设：n_queries={args.n_queries}, worklist_L={args.worklist_L}")
    print("注意：")
    print("  - host RAM ≈ graph index 大小（full vectors 嵌入 graph entry）")
    print("  - GPU HBM 包含 PQ compressed vectors + per-query buffers")
    print("  - 磁盘包含原始 base + graph + PQ 文件，实际可能更大")
    print("  - uChunks = B_gb*GiB/N，影响 PQ 精度和 GPU 占用")
    print("  - per-query GPU buf 包含 d_FPSetCoordsList（full vector rerank 缓冲）")
    print("")

    if args.N is not None:
        results = [estimate(args.N, args.dim, args.dtype, args.R, args.B_gb,
                            args.n_queries, args.worklist_L)]
        results[0]["preset_name"] = f"custom N={args.N:,}"
        print_table(results)
    else:
        results = []
        for p in PRESETS:
            r = estimate(p["N"], p["dim"], p["dtype"], p["R"], p["B_gb"],
                         args.n_queries, args.worklist_L)
            r["preset_name"] = p["name"]
            results.append(r)
        print_table(results)

    print("")
    print("关键限制（来自源码）：")
    print("  MAX_R=64（compile-time，修改需重新编译 bang_search.cu:35）")
    print("  BF_ENTRIES=399887（Bloom filter，per query bool array，非 bit-packed）")
    print("  2*L <= 1024（merge kernel 线程数上限，bang_search.cu:439）")
    print("  numCPUthreads=64（硬编码，bang_search.cu:413）")
    print("")
    print("SIFT1B 需要约 640 GB host RAM（论文原文）+ A100 80GB GPU。")
    print("DEEP1B 需要约 640 GB host RAM（论文 README）。")


if __name__ == "__main__":
    main()
