#!/usr/bin/env python3
"""
bang_preprocess.py
将 DiskANN 构图产物 (_disk.index) 转换为 BANG 所需格式。

原版：BANG-Billion-Scale-ANN/BANG_Base/bang_preprocess.py
改进：更完整的 usage、进度显示、错误检查、参数说明。

用法：
    python3 scripts/bang_preprocess.py <index.index> <out.bin> <dim> <dtype> <R>

参数：
    index.index   DiskANN build_disk_index 产生的 _disk.index 文件
    out.bin       输出 _disk.bin 路径（同时生成 _disk_metadata.bin）
    dim           向量维度（例如 128）
    dtype         0=int8, 1=uint8, 2=float32
    R             DiskANN 构图时的 -R 参数（graph degree bound，BANG 要求 R=64）

输出：
    <out.bin>             graph index 二进制（full vectors + neighbor_count + neighbor_ids）
    <out_metadata.bin>    图元数据（medoid, entry_len, dtype, dim, degree, N）

注意：
    - SECTORLEN 默认 4096（pq_flash_index.h 默认值）；若 DiskANN 构图时用了非默认扇区大小，
      需手动修改本脚本的 SECTORLEN 变量。
    - 构图时必须使用 -R 64（对应 BANG 源码 MAX_R=64，bang_search.cu:190 有 assert）。
"""

import struct
import sys
import numpy as np

SECTORLEN = 4096  # pq_flash_index.h 默认扇区大小

DTYPE_MAP = {0: ("int8",  1),
             1: ("uint8", 1),
             2: ("float32", 4)}


def main():
    if len(sys.argv) != 6:
        print(__doc__)
        sys.exit(1)

    index_path  = sys.argv[1]
    out_path    = sys.argv[2]
    dim         = int(sys.argv[3])
    dtype_code  = int(sys.argv[4])
    degree      = int(sys.argv[5])

    if dtype_code not in DTYPE_MAP:
        print(f"错误：dtype 必须是 0(int8), 1(uint8), 2(float32)，收到 {dtype_code}")
        sys.exit(1)

    dtype_name, dtype_bytes = DTYPE_MAP[dtype_code]
    meta_path = out_path[:-4] + "_metadata" + out_path[-4:]

    print(f"输入  index : {index_path}")
    print(f"输出  bin   : {out_path}")
    print(f"输出  meta  : {meta_path}")
    print(f"dim={dim}, dtype={dtype_name}({dtype_bytes}B), R={degree}")
    print(f"SECTORLEN   : {SECTORLEN}")
    print()

    with open(index_path, "rb") as f, \
         open(out_path, "wb") as w, \
         open(meta_path, "wb") as w1:

        # --- 读取 metadata sector（DiskANN pq_flash_index 格式）---
        f.read(4)           # reserved
        f.read(4)           # reserved

        total_nodes = struct.unpack('<Q', f.read(8))[0]
        num_dim     = struct.unpack('<Q', f.read(8))[0]
        medoid      = struct.unpack('<Q', f.read(8))[0]
        w1.write(struct.pack('<Q', medoid))

        max_node_len = struct.unpack('<Q', f.read(8))[0]
        w1.write(struct.pack('<Q', max_node_len))

        w1.write(struct.pack('<I', dtype_code))
        w1.write(struct.pack('<I', dim))
        w1.write(struct.pack('<I', degree))

        nodes_per_sec = struct.unpack('<Q', f.read(8))[0]
        f.read(8)  # skip
        f.read(8)  # skip
        f.read(8)  # skip

        filesize = struct.unpack('<Q', f.read(8))[0]
        n_sectors = int(filesize / SECTORLEN) - 1

        print(f"total_nodes={total_nodes:,}, dim={num_dim}, medoid={medoid}")
        print(f"max_node_len={max_node_len}B, nodes_per_sector={nodes_per_sec}")
        print(f"filesize={filesize:,}B, sectors={n_sectors:,}")

        if num_dim != dim:
            print(f"警告：index 中 dim={num_dim} 与参数 dim={dim} 不一致！")

        # --- 转换每个节点 ---
        nodes_read = 0
        for i in range(n_sectors):
            f.seek((i + 1) * SECTORLEN)
            for _ in range(nodes_per_sec):
                if nodes_read == total_nodes:
                    break
                # full vector coordinates
                for _ in range(dim):
                    w.write(f.read(dtype_bytes))
                # neighbor count (uint32)
                raw_deg = f.read(4)
                w.write(raw_deg)
                d = struct.unpack('<I', raw_deg)[0]
                if d == 0 or d > degree:
                    print(f"错误：节点 {nodes_read} degree={d} 超出范围 [1,{degree}]")
                    sys.exit(1)
                # neighbor ids，读取后排序再写入
                arr = np.zeros(d, dtype='<u4')
                for k in range(d):
                    arr[k] = struct.unpack('<I', f.read(4))[0]
                arr_sorted = np.sort(arr)
                for nb in arr_sorted:
                    w.write(struct.pack('<I', nb))
                # pad 到 degree（读掉剩余 slot）
                for _ in range(d, degree):
                    w.write(f.read(4))
                nodes_read += 1

            if (i + 1) % max(1, n_sectors // 20) == 0:
                pct = (i + 1) / n_sectors * 100
                print(f"  [{pct:5.1f}%] {nodes_read:,}/{total_nodes:,} nodes", flush=True)

        w1.write(struct.pack('<I', nodes_read))

    print(f"\n完成：共转换 {nodes_read:,} 个节点")
    print(f"  graph bin  : {out_path}")
    print(f"  metadata   : {meta_path}")


if __name__ == "__main__":
    main()
