#!/usr/bin/env python3
"""
parse_bang_output.py
解析 BANG 的 stdout 输出，提取 QPS、recall、worklist length、time。

BANG 的输出格式（来自 test_driver.cpp:402-403,526）：
    L    Time     QPS       {k}-r@{k}
    --   ----     ---       ------
    10   123.4    8103.7    45.23
    22   234.5    4264.3    67.89
    ...

说明：
    - L = worklist length
    - Time = 总耗时（毫秒，所有 5 次运行中的最后一次）
    - QPS = queries per second
    - {k}-r@{k} = Recall@k（百分比）

用法：
    python3 scripts/parse_bang_output.py \
        --input_file results/sift10k_smoke.log \
        --output_csv results/sift10k_smoke.csv \
        --dataset sift10k --dtype float --dim 128 --n_queries 100 --recall_k 10

    或通过 input_text 参数直接传字符串（供 run_sift10k_smoke.sh 调用）。
"""

import argparse
import csv
import re
import sys
from pathlib import Path


# BANG 输出每行格式：L<tab>Time<tab>QPS<tab><tab>Recall
# test_driver.cpp:526:
#   cout << nWLLen << "\t" << totalTime_wallclock << "\t" << throughput << "\t" << recall << endl;
# 注意每行可能有额外 tab，用稳健的 regex
ROW_PATTERN = re.compile(
    r"^(?P<L>\d+)\s+"
    r"(?P<time_ms>[\d.]+)\s+"
    r"(?P<qps>[\d.]+)\s+"
    r"(?P<recall>[\d.]+)\s*$"
)

# 识别 header 行（跳过）
HEADER_PATTERN = re.compile(r"^L\s+Time\s+QPS", re.IGNORECASE)


def parse_bang_text(text: str) -> list[dict]:
    rows = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if HEADER_PATTERN.match(line):
            continue
        m = ROW_PATTERN.match(line)
        if m:
            rows.append({
                "L": int(m.group("L")),
                "time_ms": float(m.group("time_ms")),
                "qps": float(m.group("qps")),
                "recall": float(m.group("recall")),
            })
    return rows


def write_csv(rows: list[dict], output_path: str, meta: dict) -> None:
    fieldnames = ["dataset", "dtype", "dim", "n_queries", "recall_k",
                  "L", "time_ms", "qps", "recall"]
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({**meta, **row})
    print(f"[OK] 写入 {len(rows)} 行到：{output_path}")


def main():
    parser = argparse.ArgumentParser(description="解析 BANG stdout 输出")
    parser.add_argument("--input_file", help="BANG 日志文件路径")
    parser.add_argument("--input_text", help="直接传入 BANG 输出文本（供脚本调用）")
    parser.add_argument("--output_csv", required=True, help="输出 CSV 路径")
    parser.add_argument("--dataset", default="unknown", help="数据集名称标签")
    parser.add_argument("--dtype", default="float")
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--n_queries", type=int, default=100)
    parser.add_argument("--recall_k", type=int, default=10)
    args = parser.parse_args()

    # 读取输入
    if args.input_text:
        text = args.input_text
    elif args.input_file:
        text = Path(args.input_file).read_text()
    else:
        print("[INFO] 未指定输入，从 stdin 读取...", file=sys.stderr)
        text = sys.stdin.read()

    rows = parse_bang_text(text)

    if not rows:
        print("[WARN] 未解析到任何数据行。")
        print("  预期格式（test_driver.cpp:526）：")
        print("    L<tab>time_ms<tab>QPS<tab><tab>Recall")
        print("  TODO: 如果 BANG 输出格式不同，请更新 ROW_PATTERN。")
        # 即使没有数据也写空 CSV
        write_csv([], args.output_csv, {
            "dataset": args.dataset, "dtype": args.dtype, "dim": args.dim,
            "n_queries": args.n_queries, "recall_k": args.recall_k,
        })
        sys.exit(0)

    meta = {
        "dataset": args.dataset,
        "dtype": args.dtype,
        "dim": args.dim,
        "n_queries": args.n_queries,
        "recall_k": args.recall_k,
    }
    write_csv(rows, args.output_csv, meta)

    # 打印摘要
    best_recall = max(r["recall"] for r in rows)
    best_qps = max(r["qps"] for r in rows)
    print(f"  解析到 {len(rows)} 个 worklist-length 点")
    print(f"  最高 recall：{best_recall:.2f}%（Recall@{args.recall_k}）")
    print(f"  最高 QPS：{best_qps:.1f}")


if __name__ == "__main__":
    main()
