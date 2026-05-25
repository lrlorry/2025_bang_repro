#!/usr/bin/env python3
"""
Verify DiskANN index quality: search with diskannpy, compare to GT.
If diskannpy's own search gives good recall but bang gives bad recall,
the conversion (diskann_to_bang) is broken.
If diskannpy also gives bad recall, the index was built incorrectly.

Usage:
  python3 tools/verify_diskann.py \
    --index  results/diskann_tmp/sift1m \
    --query  sift1m_data/sift_query.fvecs \
    --gt     sift1m_data/sift_groundtruth.ivecs
"""
import argparse, struct, numpy as np, sys

def load_fvecs(path, max_n=None):
    vecs = []
    with open(path, "rb") as f:
        while True:
            d_buf = f.read(4)
            if not d_buf: break
            d = struct.unpack("i", d_buf)[0]
            v = np.frombuffer(f.read(4 * d), dtype=np.float32).copy()
            vecs.append(v)
            if max_n and len(vecs) >= max_n:
                break
    return np.array(vecs)

def load_ivecs(path, max_n=None):
    vecs = []
    with open(path, "rb") as f:
        while True:
            d_buf = f.read(4)
            if not d_buf: break
            d = struct.unpack("i", d_buf)[0]
            v = np.frombuffer(f.read(4 * d), dtype=np.int32).copy()
            vecs.append(v)
            if max_n and len(vecs) >= max_n:
                break
    return np.array(vecs)

def compute_recall(pred, gt, k=10):
    hits = 0
    for q in range(len(pred)):
        for p in pred[q][:k]:
            if p in gt[q][:k]:
                hits += 1
    return hits / (len(pred) * k)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--index",  default="results/diskann_tmp/sift1m")
    ap.add_argument("--query",  default="sift1m_data/sift_query.fvecs")
    ap.add_argument("--gt",     default="sift1m_data/sift_groundtruth.ivecs")
    ap.add_argument("--numq",   type=int, default=1000)
    ap.add_argument("--k",      type=int, default=10)
    ap.add_argument("--L",      type=int, default=64)
    args = ap.parse_args()

    print(f"Loading {args.numq} queries ...", flush=True)
    queries = load_fvecs(args.query, args.numq)
    gt      = load_ivecs(args.gt,    args.numq)
    print(f"Queries: {queries.shape}, GT: {gt.shape}", flush=True)

    try:
        import diskannpy
    except ImportError:
        print("diskannpy not installed: pip install diskannpy", file=sys.stderr)
        sys.exit(1)

    # Split index_directory and prefix
    import os
    index_dir    = os.path.dirname(args.index)
    index_prefix = os.path.basename(args.index)

    print(f"Loading DiskANN index {args.index} ...", flush=True)
    import inspect
    sig = inspect.signature(diskannpy.StaticMemoryIndex.__init__)
    print(f"StaticMemoryIndex signature: {sig}", flush=True)

    # Try different API versions
    try:
        idx = diskannpy.StaticMemoryIndex(
            index_directory=index_dir,
            index_prefix=index_prefix,
            num_threads=1,
            initial_search_complexity=args.L,
        )
    except TypeError:
        try:
            idx = diskannpy.StaticMemoryIndex(
                data_type="float",
                distance_metric="l2",
                index_directory=index_dir,
                index_prefix=index_prefix,
                num_threads=1,
                initial_search_complexity=args.L,
            )
        except TypeError:
            idx = diskannpy.StaticMemoryIndex(
                vector_dtype=np.float32,
                distance_metric="l2",
                index_directory=index_dir,
                index_prefix=index_prefix,
                num_threads=1,
                initial_search_complexity=args.L,
            )

    print(f"Searching L={args.L} ...", flush=True)
    # Try different search API versions
    try:
        ids, _ = idx.batch_search(queries, k_neighbors=args.k, complexity=args.L, num_threads=1)
    except (TypeError, AttributeError):
        try:
            ids, _ = idx.search(queries, k_neighbors=args.k, complexity=args.L)
        except TypeError:
            ids, _ = idx.search(queries, args.k, args.L)

    recall = compute_recall(ids, gt, args.k)
    print(f"\nDiskANN own search: Recall@{args.k} = {recall:.4f}  (L={args.L}, numQ={args.numq})")
    print("If recall is high (>0.5): DiskANN index is good, conversion is the bug.")
    print("If recall is low (~0.01): DiskANN index was built incorrectly.")
