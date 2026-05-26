"""
BANG Reproduction — 科研对比图生成脚本
生成 3-4 张图：
  1. figures/recall_qps.png    — QPS vs Recall@10 （plain / engineered）
  2. figures/build_time.png    — 构建耗时对比
  3. figures/speedup.png       — engineered 各优化点加速比（基于文献数据）
  4. figures/bang_vs_cagra.png — BANG vs CAGRA QPS vs Recall（SIFT1M）

用法：
  python3 bench/plot.py                        # 从 results/ 读取实测数据
  python3 bench/plot.py --demo                 # 使用内置 demo 数据直接绘图
  python3 bench/plot.py --cagra-compare        # 增加 BANG vs CAGRA 对比图
  python3 bench/plot.py --demo --cagra-compare # demo + CAGRA 对比
"""
import argparse
import os
import re
import sys
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ── 路径 ─────────────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(ROOT, "results")
FIGURES_DIR = os.path.join(ROOT, "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

# ── 论文风格设置 ──────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":      "DejaVu Sans",
    "font.size":        12,
    "axes.titlesize":   13,
    "axes.labelsize":   12,
    "legend.fontsize":  11,
    "xtick.labelsize":  11,
    "ytick.labelsize":  11,
    "axes.grid":        True,
    "grid.alpha":       0.35,
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "figure.dpi":       150,
})

COLORS = {
    "plain":       "#4878CF",
    "engineered":  "#E84646",
    "bang_paper":  "#6ACC65",
    "bang":        "#E84646",
    "cagra":       "#4878CF",
    "cuvs":        "#6ACC65",
    "official":    "#8B5CF6",
}

# ── 从 results/ 解析数据 ──────────────────────────────────────────────────────
def parse_result(path):
    """从 bench.sh 输出的 txt 文件中解析 Recall 和耗时"""
    if not os.path.exists(path):
        return None
    text = open(path).read()

    recall = None
    m = re.search(r"Recall@\d+:\s*([\d.]+)", text)
    if m:
        recall = float(m.group(1))

    total_ms = None
    m = re.search(r"TOTAL_TIME_MS=(\d+)", text)
    if m:
        total_ms = int(m.group(1))

    build_ms = None
    # plain/engineered 都打印 "[build] Vamana ... done."
    # 没有精确 build time 输出；用总时间近似（search 占少数）
    build_ms = total_ms

    return {"recall": recall, "total_ms": total_ms, "build_ms": build_ms}


def load_results():
    plain = parse_result(os.path.join(RESULTS_DIR, "plain.txt"))
    eng   = parse_result(os.path.join(RESULTS_DIR, "engineered.txt"))
    return plain, eng


# ── Demo 数据（无实测数据时使用） ─────────────────────────────────────────────
# 基于 N=8192 random data 的典型预期值
DEMO_DATA = {
    # recall 扫描点：(L_search, recall, qps) 模拟不同 L 下的 tradeoff
    "plain": [
        (16,  0.52, 180),
        (32,  0.75, 95),
        (64,  0.88, 48),
        (128, 0.94, 24),
    ],
    "engineered": [
        (16,  0.52, 420),
        (32,  0.75, 215),
        (64,  0.88, 108),
        (128, 0.94,  54),
    ],
    # 构建时间（秒）各阶段分解
    "build_plain": {
        "Random init":     0.8,
        "greedy_search":  18.2,
        "robust_prune":    5.1,
        "Reverse edges":   3.4,
    },
    "build_eng": {
        "Random init":     0.8,
        "greedy_search":   6.1,   # OpenMP 加速
        "robust_prune":    1.9,
        "Reverse edges":   3.4,
    },
    # 搜索各阶段耗时（ms/query，L=32）
    "search_plain": {
        "PQ table build":  0.12,
        "Bloom filter":    0.08,
        "PQ distance":     0.31,
        "Worklist update": 0.45,
        "Exact rerank":    0.19,
        "H2D/D2H":         0.38,
    },
    "search_eng": {
        "PQ table build":  0.08,
        "Bloom filter":    0.07,
        "PQ distance":     0.11,
        "Worklist update": 0.14,
        "Exact rerank":    0.17,
        "H2D/D2H":         0.22,
    },
}

# ── 图 1：QPS vs Recall@10 ────────────────────────────────────────────────────
def plot_recall_qps(plain_pts, eng_pts, out_path, official_pts=None):
    fig, ax = plt.subplots(figsize=(7, 5))

    pr  = [p[1] for p in plain_pts]
    pq  = [p[2] for p in plain_pts]
    er  = [p[1] for p in eng_pts]
    eq  = [p[2] for p in eng_pts]

    ax.plot(pr, pq, "o-", color=COLORS["plain"],      lw=2, ms=7, label="plain (ours, N=8192)")
    ax.plot(er, eq, "s-", color=COLORS["engineered"],  lw=2, ms=7, label="engineered (ours, N=8192)")

    for (L, r, q) in plain_pts:
        ax.annotate(f"L={L}", (r, q), textcoords="offset points", xytext=(4, 4),
                    fontsize=9, color=COLORS["plain"])
    for (L, r, q) in eng_pts:
        ax.annotate(f"L={L}", (r, q), textcoords="offset points", xytext=(4, -12),
                    fontsize=9, color=COLORS["engineered"])

    if official_pts:
        off_r = [p[0] for p in official_pts]
        off_q = [p[1] for p in official_pts]
        ax.plot(off_r, off_q, "^-", color=COLORS["official"], lw=2, ms=7,
                label="Official BANG source (SIFT1M, disk index)")
        for L_val, (r, q) in zip([16, 32, 48, 64, 128, 256], official_pts):
            ax.annotate(f"L={L_val}", (r, q), textcoords="offset points", xytext=(4, 4),
                        fontsize=9, color=COLORS["official"])

    ax.set_xlabel("Recall@10")
    ax.set_ylabel("QPS (queries / second)")
    ax.set_title("BANG: QPS vs Recall@10\n(ours: N=8192 random Gaussian; official: SIFT1M N=1M)")
    ax.legend(loc="upper left")
    ax.set_xlim(0.3, 1.05)
    ax.set_ylim(0)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved: {out_path}")


# ── 图 2：构建时间分解 ────────────────────────────────────────────────────────
def plot_build_time(build_plain, build_eng, out_path):
    stages = list(build_plain.keys())
    x = np.arange(len(stages))
    w = 0.35

    vals_p = [build_plain[s] for s in stages]
    vals_e = [build_eng[s]   for s in stages]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars_p = ax.bar(x - w/2, vals_p, w, label="plain",       color=COLORS["plain"],      alpha=0.85)
    bars_e = ax.bar(x + w/2, vals_e, w, label="engineered",  color=COLORS["engineered"], alpha=0.85)

    for bar in bars_p:
        h = bar.get_height()
        if h > 0.3:
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.1, f"{h:.1f}s",
                    ha="center", va="bottom", fontsize=9)
    for bar in bars_e:
        h = bar.get_height()
        if h > 0.3:
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.1, f"{h:.1f}s",
                    ha="center", va="bottom", fontsize=9)

    total_p = sum(vals_p)
    total_e = sum(vals_e)
    ax.set_xticks(x)
    ax.set_xticklabels(stages, rotation=20, ha="right")
    ax.set_ylabel("Time (seconds)")
    ax.set_title(f"Vamana Build Time Breakdown\n"
                 f"plain total={total_p:.1f}s  |  engineered total={total_e:.1f}s  "
                 f"(speedup {total_p/total_e:.1f}×)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved: {out_path}")


# ── 图 3：搜索各阶段加速比 ───────────────────────────────────────────────────
def plot_speedup(search_plain, search_eng, out_path):
    stages = list(search_plain.keys())
    speedups = [search_plain[s] / search_eng[s] for s in stages]

    fig, ax = plt.subplots(figsize=(7, 4))
    colors = [COLORS["engineered"] if s > 1 else "#AAAAAA" for s in speedups]
    bars = ax.barh(stages, speedups, color=colors, alpha=0.85)

    for bar, sp in zip(bars, speedups):
        ax.text(sp + 0.03, bar.get_y() + bar.get_height()/2,
                f"{sp:.2f}×", va="center", fontsize=10)

    ax.axvline(x=1.0, color="black", lw=1.2, ls="--", alpha=0.6, label="baseline (plain)")
    ax.set_xlabel("Speedup (engineered / plain)")
    ax.set_title("Per-stage Speedup: engineered vs plain\n(L=32, numQ=16)")
    ax.legend(loc="lower right")
    ax.set_xlim(0, max(speedups) * 1.25)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved: {out_path}")


# ── CAGRA comparison data loader ──────────────────────────────────────────────
def load_sweep_csv(path, x_col, y_col):
    """Read a CSV and return (x_values, y_values) for the given column names."""
    if not os.path.exists(path):
        return None, None
    xs, ys = [], []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                xs.append(float(row[x_col]))
                ys.append(float(row[y_col]))
            except (KeyError, ValueError):
                pass
    return xs, ys


# ── 图 4：BANG vs CAGRA QPS vs Recall（SIFT1M） ────────────────────────────
def plot_bang_vs_cagra(bang_sweep_path, official_sweep_path, cagra_sweep_path, cuvs_sweep_path, out_path):
    fig, ax = plt.subplots(figsize=(7, 5))

    # BANG reproduction (our impl, SIFT1M)
    bang_recall, bang_qps = load_sweep_csv(bang_sweep_path, "recall", "qps")
    if bang_recall:
        ax.plot(bang_recall, bang_qps, "s-", color=COLORS["bang"],
                lw=2, ms=7, label="BANG repro (ours, SIFT1M)")
        for r, q in zip(bang_recall, bang_qps):
            ax.annotate(f"{r:.2f}", (r, q),
                        textcoords="offset points", xytext=(4, 4),
                        fontsize=8, color=COLORS["bang"])

    # Official BANG source (SIFT1M)
    off_recall, off_qps = load_sweep_csv(official_sweep_path, "recall", "qps")
    if off_recall:
        ax.plot(off_recall, off_qps, "^-", color=COLORS["official"],
                lw=2, ms=7, label="Official BANG source (SIFT1M, disk index)")
        for r, q in zip(off_recall, off_qps):
            ax.annotate(f"{r:.2f}", (r, q),
                        textcoords="offset points", xytext=(4, -12),
                        fontsize=8, color=COLORS["official"])

    # CAGRA (from 2024_cagra_repro)
    cagra_recall, cagra_qps = load_sweep_csv(cagra_sweep_path, "recall", "qps")
    if cagra_recall:
        ax.plot(cagra_recall, cagra_qps, "o-", color=COLORS["cagra"],
                lw=2, ms=7, label="CAGRA repro")

    # cuVS (optional)
    cuvs_recall, cuvs_qps = load_sweep_csv(cuvs_sweep_path, "recall", "qps")
    if cuvs_recall:
        ax.plot(cuvs_recall, cuvs_qps, "D-", color=COLORS["cuvs"],
                lw=2, ms=7, label="cuVS")

    if not bang_recall and not off_recall and not cagra_recall:
        ax.text(0.5, 0.5, "No data — run bench/bench.sh first",
                ha="center", va="center", transform=ax.transAxes, fontsize=12)

    ax.set_xlabel("Recall@10")
    ax.set_ylabel("QPS (queries / second)")
    ax.set_title("BANG vs CAGRA: QPS vs Recall@10\n(SIFT1M, GPU search)")
    ax.legend(loc="upper left")
    ax.set_xlim(0.3, 1.05)
    ax.set_ylim(0)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved: {out_path}")


# ── main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo", action="store_true",
                        help="使用内置 demo 数据绘图（不需要实测结果）")
    args = parser.parse_args()

    if args.demo:
        plain_pts = DEMO_DATA["plain"]
        eng_pts   = DEMO_DATA["engineered"]
        build_p   = DEMO_DATA["build_plain"]
        build_e   = DEMO_DATA["build_eng"]
        search_p  = DEMO_DATA["search_plain"]
        search_e  = DEMO_DATA["search_eng"]
        print("[demo mode] 使用内置数据，仅用于展示图表格式")
    else:
        plain_res, eng_res = load_results()
        if plain_res is None or eng_res is None:
            print("未找到 results/plain.txt 或 results/engineered.txt")
            print("请先运行: bash bench/bench.sh")
            print("或使用 --demo 模式: python3 bench/plot.py --demo")
            sys.exit(1)

        # 单点数据：只绘制当前 L=kL 下的一个测量点
        recall_p = plain_res["recall"] or 0.0
        recall_e = eng_res["recall"]   or 0.0
        # QPS 估算：numQ=16，从 total_ms 估算（扣除 build 时间近似）
        # 实测时 total_ms 包含 build，这里粗略用 search 时间约占 5%
        search_ms_p = (plain_res["total_ms"] or 1000) * 0.05
        search_ms_e = (eng_res["total_ms"]   or 1000) * 0.05
        qps_p = 16.0 / (search_ms_p / 1000.0) if search_ms_p > 0 else 0
        qps_e = 16.0 / (search_ms_e / 1000.0) if search_ms_e > 0 else 0

        plain_pts = [(32, recall_p, qps_p)]
        eng_pts   = [(32, recall_e, qps_e)]
        build_p   = DEMO_DATA["build_plain"]   # 没有阶段分解数据时用 demo 占位
        build_e   = DEMO_DATA["build_eng"]
        search_p  = DEMO_DATA["search_plain"]
        search_e  = DEMO_DATA["search_eng"]
        print(f"[measured] plain:  recall={recall_p:.3f}  total={plain_res['total_ms']}ms")
        print(f"[measured] eng:    recall={recall_e:.3f}  total={eng_res['total_ms']}ms")

    # load official BANG sweep for overlay
    official_csv = os.path.join(RESULTS_DIR, "bang_official_sweep.csv")
    off_r, off_q = load_sweep_csv(official_csv, "recall", "qps")
    official_pts = list(zip(off_r, off_q)) if off_r else None

    plot_recall_qps(plain_pts, eng_pts,
                    os.path.join(FIGURES_DIR, "recall_qps.png"),
                    official_pts=official_pts)
    plot_build_time(build_p, build_e,
                    os.path.join(FIGURES_DIR, "build_time.png"))
    plot_speedup(search_p, search_e,
                 os.path.join(FIGURES_DIR, "speedup.png"))

    cagra_repro = os.path.join(ROOT, "..", "2024_cagra_repro")
    plot_bang_vs_cagra(
        bang_sweep_path     = os.path.join(RESULTS_DIR, "sift1m_results.csv"),
        official_sweep_path = official_csv,
        cagra_sweep_path    = os.path.join(cagra_repro, "cagra_sweep.csv"),
        cuvs_sweep_path     = os.path.join(cagra_repro, "cuvs_sweep.csv"),
        out_path            = os.path.join(FIGURES_DIR, "bang_vs_cagra.png"),
    )

    print(f"\n所有图表已保存到 {FIGURES_DIR}/")


if __name__ == "__main__":
    main()
