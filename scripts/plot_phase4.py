#!/usr/bin/env python3
"""Phase 4: 처리량(FPS) 비교 — CPU / CUDA 직렬 / CUDA 파이프라인.
사용법: python plot_phase4.py <out_png> <cpu_fps> <serial_fps> <pipe_fps>
"""
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    if len(sys.argv) != 5:
        print(__doc__)
        sys.exit(1)
    out = sys.argv[1]
    cpu, serial, pipe = map(float, sys.argv[2:5])

    labels = ["CPU\n(baseline)", "CUDA\n(serial)", "CUDA pipeline\n(GPU || CPU, lock-free)"]
    vals = [cpu, serial, pipe]
    colors = ["indianred", "steelblue", "seagreen"]

    fig, ax = plt.subplots(figsize=(7, 4.6))
    ax.bar(range(len(vals)), vals, 0.6, color=colors)
    ax.set_xticks(range(len(vals)))
    ax.set_xticklabels(labels)
    ax.set_ylabel("throughput (FPS)")
    ax.set_title("Phase 4 - lock-free pipeline throughput (equal thread budget)")

    for i, v in enumerate(vals):
        ax.text(i, v, f"{v:.0f} FPS", ha="center", va="bottom", fontweight="bold")
    ax.text(1, serial * 0.5, f"{serial/cpu:.1f}x vs CPU",
            ha="center", color="white", fontweight="bold")
    ax.text(2, pipe * 0.5,
            f"{pipe/cpu:.1f}x vs CPU\n{pipe/serial:.1f}x vs serial",
            ha="center", color="white", fontweight="bold")

    ax.set_ylim(0, pipe * 1.18)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    print(f"saved {out}  (pipe {pipe/serial:.2f}x vs serial, {pipe/cpu:.2f}x vs CPU)")


if __name__ == "__main__":
    main()
