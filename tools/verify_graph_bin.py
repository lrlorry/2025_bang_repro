#!/usr/bin/env python3
"""
Verify graph.bin by running a pure-Python greedy beam search.
If this gives good recall: graph is correct, bug is in CUDA search code.
If this gives bad recall:  graph.bin conversion is broken.

Also dumps raw DiskANN .graph header / first adjacency list to diagnose format issues.

Usage:
  python3 tools/verify_graph_bin.py \
    --graph_bin results/sift1m_graph.bin \
    --base      sift1m_data/sift_base.fvecs \
    --query     sift1m_data/sift_query.fvecs \
    --gt        sift1m_data/sift_groundtruth.ivecs \
    --diskann_graph results/diskann_tmp/sift1m.graph
"""
import argparse, struct, numpy as np, heapq, os, sys

# ── vec loaders ───────────────────────────────────────────────────────────────
def load_fvecs(path, max_n=None):
    vecs = []
    with open(path, "rb") as f:
        while True:
            b = f.read(4)
            if not b: break
            d = struct.unpack("i", b)[0]
            v = np.frombuffer(f.read(4*d), dtype=np.float32).copy()
            vecs.append(v)
            if max_n and len(vecs) >= max_n: break
    return np.array(vecs)

def load_ivecs(path, max_n=None):
    vecs = []
    with open(path, "rb") as f:
        while True:
            b = f.read(4)
            if not b: break
            d = struct.unpack("i", b)[0]
            v = np.frombuffer(f.read(4*d), dtype=np.int32).copy()
            vecs.append(v)
            if max_n and len(vecs) >= max_n: break
    return np.array(vecs)

# ── graph.bin loader ──────────────────────────────────────────────────────────
def load_graph_bin(path):
    with open(path, "rb") as f:
        hdr = struct.unpack("4i", f.read(16))
        N, dim, R, medoid = hdr
        print(f"[graph.bin] N={N} dim={dim} R={R} medoid={medoid}", flush=True)
        adj = np.frombuffer(f.read(N * R * 4), dtype=np.int32).reshape(N, R)
    # sanity checks
    valid = (adj >= 0) & (adj < N)
    fill  = (adj == -1)
    bad   = ~valid & ~fill
    print(f"[graph.bin] adj: valid={valid.sum()} fill={fill.sum()} bad={bad.sum()}", flush=True)
    degrees = (adj >= 0).sum(axis=1)
    print(f"[graph.bin] degree: min={degrees.min()} max={degrees.max()} mean={degrees.mean():.1f}", flush=True)
    print(f"[graph.bin] first node neighbors: {adj[0][:10]}", flush=True)
    return N, dim, R, medoid, adj

# ── DiskANN raw graph dump ────────────────────────────────────────────────────
def dump_diskann_graph(path, n_nodes=5):
    if not path or not os.path.exists(path):
        return
    sz = os.path.getsize(path)
    print(f"\n[diskann .graph] file size: {sz} bytes ({sz/1e6:.1f} MB)", flush=True)
    with open(path, "rb") as f:
        raw16 = f.read(16)
        print(f"[diskann .graph] first 16 bytes hex: {raw16.hex()}", flush=True)
        index_size, = struct.unpack("Q", raw16[:8])
        max_degree,  = struct.unpack("I", raw16[8:12])
        entry_point, = struct.unpack("I", raw16[12:16])
        print(f"[diskann .graph] index_size={index_size} max_degree={max_degree} entry_point={entry_point}", flush=True)
        for i in range(n_nodes):
            d_buf = f.read(4)
            if not d_buf: break
            deg = struct.unpack("I", d_buf)[0]
            nbrs = list(struct.unpack(f"{deg}I", f.read(4*deg))) if deg else []
            print(f"[diskann .graph] node {i}: degree={deg} neighbors={nbrs[:8]}", flush=True)

# ── beam search ──────────────────────────────────────────────────────────────
def beam_search(vecs, adj, query, medoid, L, k):
    """Greedy beam search, beam width L, return top-k node ids."""
    N = len(vecs)
    visited = set()
    # (dist, node)  — min-heap
    cands = []

    def push(node):
        if node < 0 or node >= N or node in visited:
            return
        visited.add(node)
        d = float(np.sum((vecs[node] - query)**2))
        heapq.heappush(cands, (d, node))

    push(medoid)

    # beam: keep best L candidates, expand unchecked ones
    checked = set()
    results = []

    while True:
        # find best unchecked candidate
        unchecked = [(d, n) for d, n in cands if n not in checked]
        if not unchecked:
            break
        unchecked.sort()
        best_d, best_n = unchecked[0]
        checked.add(best_n)
        results.append((best_d, best_n))
        if len(results) >= L:
            break
        for nbr in adj[best_n]:
            push(int(nbr))

    results.sort()
    return [n for _, n in results[:k]]

def compute_recall(pred, gt, k):
    hits = sum(p in gt[q][:k] for q in range(len(pred)) for p in pred[q][:k])
    return hits / (len(pred) * k)

# ── main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph_bin",      default="results/sift1m_graph.bin")
    ap.add_argument("--base",           default="sift1m_data/sift_base.fvecs")
    ap.add_argument("--query",          default="sift1m_data/sift_query.fvecs")
    ap.add_argument("--gt",             default="sift1m_data/sift_groundtruth.ivecs")
    ap.add_argument("--diskann_graph",  default="")
    ap.add_argument("--numq",  type=int, default=100)
    ap.add_argument("--k",     type=int, default=10)
    ap.add_argument("--L",     type=int, default=64)
    args = ap.parse_args()

    # dump raw diskann graph header if provided
    dump_diskann_graph(args.diskann_graph)

    # load graph.bin
    N, dim, R, medoid, adj = load_graph_bin(args.graph_bin)

    print(f"\nLoading {args.numq} queries and base vectors ...", flush=True)
    queries = load_fvecs(args.query, args.numq)
    gt      = load_ivecs(args.gt,    args.numq)
    print(f"Loading base vectors (this may take ~30s for SIFT1M) ...", flush=True)
    vecs    = load_fvecs(args.base)
    print(f"Base: {vecs.shape}", flush=True)

    print(f"\nRunning Python beam search (L={args.L}, {args.numq} queries) ...", flush=True)
    preds = []
    for qi, q in enumerate(queries):
        ids = beam_search(vecs, adj, q, medoid, args.L, args.k)
        preds.append(ids)
        if (qi+1) % 10 == 0:
            print(f"  {qi+1}/{args.numq}", flush=True)

    recall = compute_recall(preds, gt, args.k)
    print(f"\nPython beam search: Recall@{args.k} = {recall:.4f}  (L={args.L}, numQ={args.numq})")
    if recall > 0.5:
        print("GRAPH IS CORRECT — bug is in CUDA search code.")
    elif recall > 0.1:
        print("GRAPH IS PARTIALLY CORRECT — conversion may have minor issues.")
    else:
        print("GRAPH IS BROKEN — diskann_to_bang conversion has a bug.")
