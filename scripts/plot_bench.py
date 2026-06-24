#!/usr/bin/env python3
"""벤치마크 CSV → 처리시간 분포/시계열 그래프.
사용법: python plot_bench.py ../bench/cpu_baseline.csv
"""
import sys
import csv
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load(path):
    idx, ms, labels = [], [], []
    with open(path) as f:
        r = csv.DictReader(f)
        for row in r:
            idx.append(int(row["frame_index"]))
            ms.append(float(row["proc_ms"]))
            labels.append(row["label"])
    return idx, ms, labels


def main():
    if len(sys.argv) < 2:
        print("usage: python plot_bench.py <csv> [out_png]")
        sys.exit(1)
    path = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else "bench_plot.png"

    idx, ms, labels = load(path)
    mean = statistics.mean(ms)
    fps = 1000.0 / mean if mean else 0

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # 시계열
    ax1.plot(idx, ms, lw=0.8)
    ax1.axhline(mean, color="r", ls="--", label=f"mean {mean:.2f} ms")
    ax1.set_xlabel("frame index")
    ax1.set_ylabel("proc time (ms)")
    ax1.set_title(f"per-frame latency  ({fps:.1f} FPS)")
    ax1.legend()

    # 히스토그램
    ax2.hist(ms, bins=40, color="steelblue", edgecolor="k", alpha=0.7)
    ax2.axvline(mean, color="r", ls="--")
    ax2.set_xlabel("proc time (ms)")
    ax2.set_ylabel("count")
    ax2.set_title("latency distribution")

    fig.tight_layout()
    fig.savefig(out, dpi=120)
    print(f"saved {out}  (mean={mean:.3f} ms, FPS={fps:.1f}, N={len(ms)})")


if __name__ == "__main__":
    main()
