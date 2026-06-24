#!/usr/bin/env python3
"""CPU baseline vs CUDA 처리시간 비교 그래프.
사용법: python plot_compare.py <cpu.csv> <cuda.csv> [out_png]
"""
import sys
import csv
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_ms(path):
    ms = []
    with open(path) as f:
        for row in csv.DictReader(f):
            ms.append(float(row["proc_ms"]))
    return ms


def pct(xs, p):
    xs = sorted(xs)
    k = max(0, min(len(xs) - 1, int(round((p / 100.0) * (len(xs) - 1)))))
    return xs[k]


def main():
    if len(sys.argv) < 3:
        print("usage: python plot_compare.py <cpu.csv> <cuda.csv> [out_png]")
        sys.exit(1)
    cpu, cuda = load_ms(sys.argv[1]), load_ms(sys.argv[2])
    out = sys.argv[3] if len(sys.argv) > 3 else "compare.png"

    cpu_mean, cuda_mean = statistics.mean(cpu), statistics.mean(cuda)
    speedup = cpu_mean / cuda_mean

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.2))

    # 좌: 지연 분포 오버레이
    ax1.hist(cpu, bins=60, alpha=0.55, label=f"CPU  ({1000/cpu_mean:.0f} FPS)",
             color="indianred")
    ax1.hist(cuda, bins=60, alpha=0.55, label=f"CUDA ({1000/cuda_mean:.0f} FPS)",
             color="steelblue")
    ax1.set_xlabel("proc time (ms)")
    ax1.set_ylabel("count")
    ax1.set_title("latency distribution")
    ax1.legend()

    # 우: mean/p95/p99 막대 비교
    metrics = ["mean", "p95", "p99"]
    cpu_v = [cpu_mean, pct(cpu, 95), pct(cpu, 99)]
    cuda_v = [cuda_mean, pct(cuda, 95), pct(cuda, 99)]
    x = range(len(metrics))
    w = 0.38
    ax2.bar([i - w / 2 for i in x], cpu_v, w, label="CPU", color="indianred")
    ax2.bar([i + w / 2 for i in x], cuda_v, w, label="CUDA", color="steelblue")
    ax2.set_xticks(list(x))
    ax2.set_xticklabels(metrics)
    ax2.set_ylabel("ms / frame")
    ax2.set_title(f"CPU vs CUDA  (mean {speedup:.1f}x faster)")
    for i, (a, b) in enumerate(zip(cpu_v, cuda_v)):
        ax2.text(i - w / 2, a, f"{a:.2f}", ha="center", va="bottom", fontsize=8)
        ax2.text(i + w / 2, b, f"{b:.2f}", ha="center", va="bottom", fontsize=8)
    ax2.legend()

    fig.tight_layout()
    fig.savefig(out, dpi=120)
    print(f"saved {out}  speedup={speedup:.2f}x  "
          f"CPU mean={cpu_mean:.3f} CUDA mean={cuda_mean:.3f}")


if __name__ == "__main__":
    main()
