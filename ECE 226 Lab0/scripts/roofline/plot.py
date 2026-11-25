import matplotlib.pyplot as plt
import numpy as np
import math

from device import *
from measure import *


def plot_roofline(peak_tflops, bw_tbs, gemm_points, add_points, softmax_points, title, fname=None):
    fig, ax = plt.subplots(figsize=(8.4, 5.0))
    ai = np.logspace(-1.2, 4.2, 500)
    ax.plot(ai, np.minimum(bw_tbs * ai, peak_tflops), color="#2F5496", lw=2.4,
            label=f"roofline: {peak_tflops:.3g} TFLOP/s, {bw_tbs:.3g} TB/s")
    ridge = peak_tflops / bw_tbs
    ax.plot([ridge], [peak_tflops], "o", color="#2F5496", ms=7)
    ax.annotate(
        f"ridge = {ridge:.0f} FLOP/B",
        (ridge, peak_tflops),
        xytext=(-92, -24),
        textcoords="offset points",
        color="#2F5496",
    )

    label_indices = set(range(min(3, len(gemm_points))))
    for i, p in enumerate(gemm_points):
        ax.plot(
            [p["ai"]], [p["tflops"]], "s", ms=7, color="#B03A22",
            label="GEMM" if i == 0 else "_nolegend_",
        )
        if i in label_indices:
            ax.annotate(
                f"N={p['n']}",
                (p["ai"], p["tflops"]),
                xytext=(7, 7),
                textcoords="offset points",
                fontsize=8,
            )

    def plot_sweep(records, marker, color, label, show_sizes=False):
        label_offsets = {}
        if show_sizes:
            # Several elementwise points can have nearly identical throughput.
            # Spread labels for each close group so every measured size stays readable.
            sorted_indices = sorted(range(len(records)), key=lambda i: records[i]["tflops"])
            groups = []
            for i in sorted_indices:
                if not groups or records[i]["tflops"] > records[groups[-1][-1]]["tflops"] * 1.18:
                    groups.append([i])
                else:
                    groups[-1].append(i)
            for group in groups:
                offsets = [12 * (j - (len(group) - 1) / 2) for j in range(len(group))]
                for i, offset in zip(group, offsets):
                    label_offsets[i] = offset

        for i, p in enumerate(records):
            ax.plot(
                [p["ai"]], [p["tflops"]], marker, ms=6,
                color=color, alpha=0.82,
                label=label if i == 0 else "_nolegend_",
            )
            if show_sizes:
                ax.annotate(
                    f"N={p['n']:,}",
                    (p["ai"], p["tflops"]),
                    xytext=(8, label_offsets.get(i, 0)),
                    textcoords="offset points",
                    fontsize=7,
                    color=color,
                    ha="left",
                    va="center",
                )

    plot_sweep(add_points, "o", "#25855A", "element-wise add", show_sizes=True)
    plot_sweep(softmax_points, "^", "#7A4EAB", "softmax")
    fig.subplots_adjust(bottom=0.20)

    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("arithmetic intensity (FLOP/byte)")
    ax.set_ylabel("achieved TFLOP/s")
    ax.set_title(title)
    ax.grid(True, which="major", color="#DDDDDD", lw=0.7)
    ax.legend(loc="lower right", frameon=False, fontsize=9)
    if fname:
        fig.savefig(device_out_dir / "roofline" / fname, dpi=160, bbox_inches="tight")
    if plt.get_backend().lower() != "agg":
        plt.show()
    return ridge


# Plot the precision with the HIGHEST measured ceiling -- that is the roof this machine actually has.
# On a GPU with matrix cores that is fp16 or bf16;
# on a CPU it is usually fp32, because low precision there is emulated rather than accelerated.
PLOT_DT = BEST_DT or next((d for d in ("fp16", "bf16", "fp32", "fp64") if d in ROOF_TFLOPS), None)

if __name__ == "__main__":
    pts = [r for r in GEMM if r["dtype"] == PLOT_DT]
    add_pts = [r for r in ELEMENTWISE_ADD if r["dtype"] == PLOT_DT]
    softmax_pts = [r for r in SOFTMAX if r["dtype"] == PLOT_DT]
    if PLOT_DT and pts and add_pts and softmax_pts and math.isfinite(ACHIEVED_BW_TBS) and ACHIEVED_BW_TBS > 0:
        ROOFLINE_PNG = f"roofline_{PLOT_DT}.png"
        RIDGE = plot_roofline(
            ROOF_TFLOPS[PLOT_DT],
            ACHIEVED_BW_TBS,
            pts,
            add_pts,
            softmax_pts,
            f"{DEVICE_NAME} - measured roofline ({PLOT_DT})",
            ROOFLINE_PNG,
        )
        print(f"wrote {ROOFLINE_PNG}")
        print(f"compute roof   : {ROOF_TFLOPS[PLOT_DT]:.3g} TFLOP/s ({PLOT_DT})")
        print(f"bandwidth      : {ACHIEVED_BW_TBS*1e3:.1f} GB/s (largest buffer in the sweep)")
        print(f"ridge point    : {RIDGE:.0f} FLOP/byte")
        print(f"element-wise add: {len(add_pts)} measured sizes; AI={add_pts[0]['ai']:.4g} FLOP/byte")
        print(f"softmax         : {len(softmax_pts)} measured row lengths; AI~{softmax_pts[0]['ai']:.4g} FLOP/byte")
        print("softmax AI uses ~5 operations/value and minimum input-read + output-write traffic;")
        print("exp/reduction throughput and extra implementation traffic can move it from the ideal Roofline point.")
        print(f"\nAnything with arithmetic intensity below {RIDGE:.0f} FLOP/byte cannot reach")
        print("your compute roof, no matter how well it is written.")
    else:
        print("Not enough measurements to plot -- rerun Q2 to Q4 first.")
