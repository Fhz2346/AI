from measure.bandwidth import *


if __name__ == "__main__":
    best = max((r["gbs"] for r in BW), default=float("nan"))
    print(f"{'buffer':>10s} {'ms':>9s} {'GB/s':>9s} {'% of best':>10s}")
    print("-" * 42)
    for r in BW:
        print(f"{r['mib']:7d}MiB {r['ms']:9.3f} {r['gbs']:9.1f} {100*r['gbs']/best:9.1f}%")
    if BW:
        print(f"\nbest anywhere in the sweep : {best:9.1f} GB/s  at {max(BW, key=lambda r: r['gbs'])['mib']} MiB")
        print(f"largest buffer measured    : {BW[-1]['gbs']:9.1f} GB/s  at {BW[-1]['mib']} MiB   <-- use THIS in the roofline")
    l2 = INFO.get("L2_cache_size")
    if l2:
        print(f"\nlast-level cache reported by PyTorch: {l2/2**20:.0f} MiB -- any buffer near")
        print("or below this is partly a cache measurement, not a memory measurement.")
