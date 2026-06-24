#!/usr/bin/env python3
"""Phase 3: pageable vs pinned 전송시간 분해(스택 막대).
사용법:
  python plot_phase3.py <out_png> \
      <pg_h2d> <pg_kernel> <pg_d2h> <pn_h2d> <pn_kernel> <pn_d2h>
값 단위: ms/frame (CUDA event 측정 평균).
"""
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    if len(sys.argv) != 8:
        print(__doc__)
        sys.exit(1)
    out = sys.argv[1]
    pg = list(map(float, sys.argv[2:5]))   # H2D, kernel, D2H
    pn = list(map(float, sys.argv[5:8]))

    labels = ["pageable\n(Phase 2)", "pinned\n(Phase 3)"]
    h2d = [pg[0], pn[0]]
    ker = [pg[1], pn[1]]
    d2h = [pg[2], pn[2]]

    fig, ax = plt.subplots(figsize=(6, 4.5))
    x = range(len(labels))
    b1 = ax.bar(x, h2d, 0.5, label="H2D", color="#e0883b")
    b2 = ax.bar(x, ker, 0.5, bottom=h2d, label="kernels", color="#4878a8")
    bottom2 = [a + b for a, b in zip(h2d, ker)]
    b3 = ax.bar(x, d2h, 0.5, bottom=bottom2, label="D2H", color="#c0504d")

    for i in x:
        total = h2d[i] + ker[i] + d2h[i]
        ax.text(i, total, f"{total:.3f} ms", ha="center", va="bottom",
                fontsize=9, fontweight="bold")
        ax.text(i, bottom2[i] + d2h[i] / 2, f"D2H\n{d2h[i]:.3f}",
                ha="center", va="center", fontsize=8, color="white")

    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylabel("GPU-side time (ms / frame)")
    d2h_drop = 100 * (pg[2] - pn[2]) / pg[2]
    ax.set_title(f"GPU transfer breakdown — pinned D2H -{d2h_drop:.0f}%")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    print(f"saved {out}  (D2H {pg[2]:.3f}->{pn[2]:.3f} ms, -{d2h_drop:.0f}%)")


if __name__ == "__main__":
    main()
