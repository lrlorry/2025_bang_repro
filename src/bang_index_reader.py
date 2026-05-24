#!/usr/bin/env python3
"""
bang_index_reader.py
CPU-only 工具：读取 BANG binary 文件，验证结构、打印统计信息。
不需要 GPU，用于在无 GPU 环境下验证 index 完整性。

用法：
    python3 src/bang_index_reader.py inspect  <prefix> <dim> <dtype> [--sample N]
    python3 src/bang_index_reader.py pq-stats <prefix> <dim>
    python3 src/bang_index_reader.py pq-dist  <prefix> <dim> <dtype> <query_vec_as_floats...>

命令：
    inspect    读取 _disk.bin + _disk_metadata.bin，打印图统计（度数分布、medoid 等）
    pq-stats   读取 _pq_pivots.bin，打印 PQ 参数（uChunks, dim, centroid 范数分布）
    pq-dist    对一个 query 向量计算 PQ asymmetric distance（验证 PQ 正确性）

示例：
    python3 src/bang_index_reader.py inspect sift10kfiles/sift10k_index 128 float32
    python3 src/bang_index_reader.py pq-stats sift10kfiles/sift10k_index 128
"""

import argparse
import struct
import sys
import os
import numpy as np

DTYPE_MAP = {"float32": ("f", 4), "float": ("f", 4),
             "uint8": ("B", 1), "int8": ("b", 1)}


# ---------------------------------------------------------------------------
# 读取 _disk_metadata.bin
# ---------------------------------------------------------------------------

def read_metadata(prefix: str) -> dict:
    path = prefix + "_disk_metadata.bin"
    if not os.path.exists(path):
        sys.exit(f"[ERROR] 文件不存在: {path}")
    with open(path, "rb") as f:
        medoid       = struct.unpack('<Q', f.read(8))[0]
        entry_len    = struct.unpack('<Q', f.read(8))[0]  # max_node_len in bytes
        dtype_code   = struct.unpack('<I', f.read(4))[0]
        dim          = struct.unpack('<I', f.read(4))[0]
        degree       = struct.unpack('<I', f.read(4))[0]
        n_nodes      = struct.unpack('<I', f.read(4))[0]
    dtype_name = {0: "int8", 1: "uint8", 2: "float32"}.get(dtype_code, f"?({dtype_code})")
    return {"medoid": medoid, "entry_len": entry_len,
            "dtype": dtype_name, "dim": dim, "degree": degree, "N": n_nodes}


# ---------------------------------------------------------------------------
# inspect: 读取 _disk.bin，验证 entry 结构，打印度数统计
# ---------------------------------------------------------------------------

def cmd_inspect(prefix: str, dim: int, dtype: str, sample: int):
    if dtype not in DTYPE_MAP:
        sys.exit(f"[ERROR] dtype 必须是 float32/uint8/int8，收到 {dtype}")
    fmt_char, vbytes = DTYPE_MAP[dtype]

    meta = read_metadata(prefix)
    print("=== _disk_metadata.bin ===")
    for k, v in meta.items():
        print(f"  {k:12s}: {v}")
    print()

    # 验证 entry_len 和参数是否一致
    expected_entry = dim * vbytes + 4 + meta["degree"] * 4
    if meta["entry_len"] != expected_entry:
        print(f"[WARN] entry_len={meta['entry_len']} != dim*vbytes+4+R*4={expected_entry}")
        print("       （若 dtype/dim 与构图时不一致会出现此警告）")
    else:
        print(f"[OK]  entry_len={meta['entry_len']} 与 dim={dim}, vbytes={vbytes}, R={meta['degree']} 一致")

    disk_path = prefix + "_disk.bin"
    if not os.path.exists(disk_path):
        sys.exit(f"[ERROR] 文件不存在: {disk_path}")

    filesize = os.path.getsize(disk_path)
    expected_size = meta["N"] * meta["entry_len"]
    if filesize != expected_size:
        print(f"[WARN] disk.bin 大小={filesize:,} != N*entry_len={expected_size:,}")
    else:
        print(f"[OK]  disk.bin 大小={filesize:,} = {meta['N']:,} × {meta['entry_len']}")

    # 采样节点，统计度数
    N = meta["N"]
    entry_len = meta["entry_len"]
    R = meta["degree"]
    n_sample = min(sample, N)
    indices = np.random.choice(N, n_sample, replace=False)

    degrees = []
    with open(disk_path, "rb") as f:
        for idx in indices:
            f.seek(idx * entry_len + dim * vbytes)
            deg = struct.unpack('<I', f.read(4))[0]
            degrees.append(deg)

    degrees = np.array(degrees)
    print(f"\n=== 度数统计（随机采样 {n_sample:,} 节点）===")
    print(f"  min={degrees.min()}, max={degrees.max()}, mean={degrees.mean():.2f}, "
          f"p50={np.percentile(degrees,50):.0f}, p5={np.percentile(degrees,5):.0f}")
    print(f"  度数=R({R}) 的比例: {(degrees == R).mean()*100:.1f}%")

    # 打印 medoid 节点信息
    print(f"\n=== Medoid 节点 (id={meta['medoid']}) ===")
    with open(disk_path, "rb") as f:
        f.seek(meta['medoid'] * entry_len)
        vec_raw = f.read(dim * vbytes)
        deg = struct.unpack('<I', f.read(4))[0]
        nbs = [struct.unpack('<I', f.read(4))[0] for _ in range(deg)]
    if dtype == "float32" or dtype == "float":
        vec = np.frombuffer(vec_raw, dtype=np.float32)
        print(f"  向量 L2 norm : {np.linalg.norm(vec):.4f}")
        print(f"  向量前5维    : {vec[:5]}")
    else:
        vec = np.frombuffer(vec_raw, dtype=np.uint8 if dtype == "uint8" else np.int8)
        print(f"  向量前5维    : {vec[:5]}")
    print(f"  度数         : {deg}")
    print(f"  邻居 ids     : {nbs[:min(10, deg)]}")


# ---------------------------------------------------------------------------
# pq-stats: 读取 _pq_pivots.bin，打印 PQ 参数
# ---------------------------------------------------------------------------

def cmd_pq_stats(prefix: str, dim: int):
    """
    _pq_pivots.bin 格式（DiskANN 输出）：
      section 1: pq table = [256, D] float32
      section 2: centroid = [D] float32
      section 3: chunk offsets = [uChunks+1] uint32（或 float32 存 uint，DiskANN 用 float 存）
    """
    path = prefix + "_pq_pivots.bin"
    if not os.path.exists(path):
        sys.exit(f"[ERROR] 文件不存在: {path}")

    filesize = os.path.getsize(path)
    # 推断 uChunks：filesize = 256*D*4 + D*4 + (uChunks+1)*4
    # => uChunks = (filesize - D*4*(256+1)) / 4 - 1
    pivot_and_centroid = 256 * dim * 4 + dim * 4
    if filesize <= pivot_and_centroid:
        sys.exit(f"[ERROR] pq_pivots.bin 太小（{filesize}B），无法推断 uChunks")
    n_chunk_offsets = (filesize - pivot_and_centroid) // 4
    uChunks = n_chunk_offsets - 1

    print(f"=== {path} ===")
    print(f"  filesize    : {filesize:,} B")
    print(f"  dim         : {dim}")
    print(f"  uChunks     : {uChunks}")
    print(f"  PQ table    : 256 × {dim} float32 = {256*dim*4:,} B")
    print(f"  centroid    : {dim} float32 = {dim*4} B")
    print(f"  chunk offsets: {uChunks+1} entries")

    data = np.fromfile(path, dtype=np.float32)
    pq_table  = data[:256 * dim].reshape(256, dim)
    centroid  = data[256 * dim: 256 * dim + dim]
    # chunk offsets 存为 float32（DiskANN 惯例）
    chunk_off = data[256 * dim + dim:].astype(np.int32)

    print(f"\n  centroid L2 norm  : {np.linalg.norm(centroid):.4f}")
    print(f"  pq_table row norms: min={np.linalg.norm(pq_table, axis=1).min():.4f}, "
          f"max={np.linalg.norm(pq_table, axis=1).max():.4f}")
    print(f"  chunk_offsets[0:5]: {chunk_off[:5].tolist()}")
    print(f"  chunk_offsets[-3:]: {chunk_off[-3:].tolist()}")


# ---------------------------------------------------------------------------
# pq-dist: 对给定 query 向量计算 PQ asymmetric distance
# ---------------------------------------------------------------------------

def cmd_pq_dist(prefix: str, dim: int, dtype: str, query_vals: list):
    """
    复现 BANG GPU kernel populate_pqDist_par 的 CPU 等价逻辑，
    验证 PQ distance table 计算是否正确。
    """
    if dtype not in DTYPE_MAP:
        sys.exit(f"[ERROR] dtype 必须是 float32/uint8/int8，收到 {dtype}")

    # 读取 query 向量
    if len(query_vals) != dim:
        sys.exit(f"[ERROR] 需要提供 {dim} 个 float 值，收到 {len(query_vals)}")
    query = np.array(query_vals, dtype=np.float32)

    # 读取 PQ pivots
    pq_path = prefix + "_pq_pivots.bin"
    if not os.path.exists(pq_path):
        sys.exit(f"[ERROR] 文件不存在: {pq_path}")
    data = np.fromfile(pq_path, dtype=np.float32)
    filesize = os.path.getsize(pq_path)
    pivot_bytes = 256 * dim * 4 + dim * 4
    n_chunk_offsets = (filesize - pivot_bytes) // 4
    uChunks = n_chunk_offsets - 1

    pq_table  = data[:256 * dim].reshape(256, dim)
    centroid  = data[256 * dim: 256 * dim + dim]
    chunk_off = data[256 * dim + dim:].astype(np.int32)

    # 减去 centroid（bang_search.cu::populate_pqDist_par 在距离计算前减 centroid）
    q_centered = query - centroid

    # 构建 per-query PQ distance table：shape [uChunks, 256]
    pq_dist_table = np.zeros((uChunks, 256), dtype=np.float32)
    for chunk in range(uChunks):
        c_start = chunk_off[chunk]
        c_end   = chunk_off[chunk + 1]
        q_chunk = q_centered[c_start:c_end]
        for cent_id in range(256):
            p_chunk = pq_table[cent_id, c_start:c_end] - centroid[c_start:c_end]
            diff = q_chunk - p_chunk
            pq_dist_table[chunk, cent_id] = float(np.dot(diff, diff))

    print(f"=== PQ distance table for query (dim={dim}, uChunks={uChunks}) ===")
    print(f"  table shape: {pq_dist_table.shape}")
    print(f"  chunk 0 distances (to 256 centroids):")
    print(f"    min={pq_dist_table[0].min():.4f}, max={pq_dist_table[0].max():.4f}, "
          f"argmin={pq_dist_table[0].argmin()}")

    # 读取 PQ compressed vectors 并计算 distance to first 5 vectors
    comp_path = prefix + "_pq_compressed.bin"
    if not os.path.exists(comp_path):
        print(f"[SKIP] {comp_path} 不存在，跳过 top-5 distance 验证")
        return

    comp = np.fromfile(comp_path, dtype=np.uint8)
    N = len(comp) // uChunks
    comp = comp.reshape(N, uChunks)

    print(f"\n  PQ asymmetric distance to first 5 vectors:")
    for i in range(min(5, N)):
        dist = sum(pq_dist_table[c, comp[i, c]] for c in range(uChunks))
        print(f"    vector {i}: PQ dist = {dist:.4f}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="BANG binary index CPU analyzer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    sub = parser.add_subparsers(dest="cmd")

    p_inspect = sub.add_parser("inspect", help="检查 _disk.bin 结构")
    p_inspect.add_argument("prefix", help="index prefix（如 sift10kfiles/sift10k_index）")
    p_inspect.add_argument("dim", type=int)
    p_inspect.add_argument("dtype", choices=["float32", "float", "uint8", "int8"])
    p_inspect.add_argument("--sample", type=int, default=1000)

    p_pq = sub.add_parser("pq-stats", help="打印 PQ pivots 统计")
    p_pq.add_argument("prefix")
    p_pq.add_argument("dim", type=int)

    p_dist = sub.add_parser("pq-dist", help="计算单个 query 的 PQ distance table")
    p_dist.add_argument("prefix")
    p_dist.add_argument("dim", type=int)
    p_dist.add_argument("dtype", choices=["float32", "float", "uint8", "int8"])
    p_dist.add_argument("query", nargs="+", type=float,
                        metavar="v", help="query vector values (dim floats)")

    args = parser.parse_args()
    if args.cmd is None:
        parser.print_help()
        sys.exit(0)

    if args.cmd == "inspect":
        cmd_inspect(args.prefix, args.dim, args.dtype, args.sample)
    elif args.cmd == "pq-stats":
        cmd_pq_stats(args.prefix, args.dim)
    elif args.cmd == "pq-dist":
        cmd_pq_dist(args.prefix, args.dim, args.dtype, args.query)


if __name__ == "__main__":
    main()
