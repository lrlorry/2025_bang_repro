#!/usr/bin/env python3
"""
plot_results.py
读取 results/*.csv，生成 recall vs QPS 曲线图，保存到 figures/。

CSV 格式（由 parse_bang_output.py 生成）：
    dataset, dtype, dim, n_queries, recall_k, L, time_ms, qps, recall

用法：
    python3 scripts/plot_results.py
    python3 scripts/plot_results.py --results_dir results --figures_dir figures
    python3 scripts/plot_results.py --example   # 用内置示例数据生成示例图
"""

import argparse
import csv
import sys
from pathlib import Path
from collections import defaultdict

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("[WARN] matplotlib 未安装，无法生成图表。pip install matplotlib")


EXAMPLE_DATA = [
    # dataset, L, time_ms, qps, recall
    ("sift10k_example", 10,  45.2,  2212.4,  42.1),
    ("sift10k_example", 22,  78.3,  1276.5,  61.8),
    ("sift10k_example", 34, 112.1,   892.1,  74.3),
    ("sift10k_example", 46, 148.9,   671.6,  82.9),
    ("sift10k_example", 58, 187.2,   534.2,  88.4),
    ("sift10k_example", 70, 225.4,   443.6,  91.2),
    ("sift10k_example", 82, 261.3,   382.8,  93.5),
    ("sift10k_example", 94, 297.8,   335.8,  95.1),
]


def load_csvs(results_dir: Path) -> dict:
    """返回 {dataset_name: [(L, qps, recall), ...]}"""
    data = defaultdict(list)
    csv_files = list(results_dir.glob("*.csv"))
    if not csv_files:
        return data
    for fp in sorted(csv_files):
        with open(fp, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    dataset = row.get("dataset", fp.stem)
                    L = int(row["L"])
                    qps = float(row["qps"])
                    recall = float(row["recall"])
                    if qps > 0 and recall > 0:
                        data[dataset].append((L, qps, recall))
                except (KeyError, ValueError):
                    continue
    return data


def plot_recall_qps(datasets: dict, figures_dir: Path, title_suffix: str = "") -> None:
    if not HAS_MPL:
        print("[SKIP] 跳过绘图（matplotlib 未安装）")
        return

    fig, ax = plt.subplots(figsize=(8, 5))

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for i, (name, points) in enumerate(sorted(datasets.items())):
        points_sorted = sorted(points, key=lambda x: x[2])
        recalls = [p[2] for p in points_sorted]
        qps = [p[1] for p in points_sorted]
        color = colors[i % len(colors)]
        ax.plot(recalls, qps, "o-", label=name, color=color, markersize=5)
        # 标注 L 值
        for L, q, r in points_sorted[::max(1, len(points_sorted) // 4)]:
            ax.annotate(f"L={L}", (r, q), fontsize=7,
                        xytext=(3, 3), textcoords="offset points", color=color)

    ax.set_xlabel("Recall@k (%)", fontsize=12)
    ax.set_ylabel("QPS", fontsize=12)
    ax.set_title(f"BANG: Recall vs QPS{title_suffix}", fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))

    out_png = figures_dir / "recall_vs_qps.png"
    out_pdf = figures_dir / "recall_vs_qps.pdf"
    figures_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] 图表已保存：{out_png}")
    print(f"[OK] 图表已保存：{out_pdf}")


def plot_recall_latency(datasets: dict, figures_dir: Path) -> None:
    if not HAS_MPL:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for i, (name, points) in enumerate(sorted(datasets.items())):
        points_sorted = sorted(points, key=lambda x: x[2])
        recalls = [p[2] for p in points_sorted]
        # latency per query (ms): 从 qps 反推
        latencies = [1000.0 / p[1] if p[1] > 0 else float("inf") for p in points_sorted]
        color = colors[i % len(colors)]
        ax.plot(recalls, latencies, "s--", label=name, color=color, markersize=5)

    ax.set_xlabel("Recall@k (%)", fontsize=12)
    ax.set_ylabel("Latency per query (ms)", fontsize=12)
    ax.set_title("BANG: Recall vs Latency", fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    out_png = figures_dir / "recall_vs_latency.png"
    figures_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] 图表已保存：{out_png}")


def main():
    parser = argparse.ArgumentParser(description="BANG 结果可视化")
    parser.add_argument("--results_dir", default="results", help="CSV 文件目录")
    parser.add_argument("--figures_dir", default="figures", help="图表输出目录")
    parser.add_argument("--example", action="store_true", help="用内置示例数据生成示例图")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    figures_dir = Path(args.figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    if args.example:
        print("[INFO] 使用内置示例数据生成示例图...")
        datasets = defaultdict(list)
        for name, L, time_ms, qps, recall in EXAMPLE_DATA:
            datasets[name].append((L, qps, recall))
        plot_recall_qps(dict(datasets), figures_dir, title_suffix=" (示例数据)")
        plot_recall_latency(dict(datasets), figures_dir)
        return

    datasets = load_csvs(results_dir)
    if not datasets:
        print(f"[WARN] {results_dir} 中没有有效的 CSV 数据。")
        print("  运行 bash scripts/run_sift10k_smoke.sh 生成真实数据，")
        print("  或运行 python3 scripts/plot_results.py --example 生成示例图。")
        sys.exit(0)

    print(f"[INFO] 找到 {len(datasets)} 个数据集：{list(datasets.keys())}")
    plot_recall_qps(datasets, figures_dir)
    plot_recall_latency(datasets, figures_dir)


if __name__ == "__main__":
    main()
