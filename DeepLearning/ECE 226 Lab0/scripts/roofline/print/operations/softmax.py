from measure import SOFTMAX, SOFTMAX_OPS_PER_ELEMENT


if __name__ == "__main__":
    print(f"{'dtype':8s} {'rows':>8s} {'row len':>9s} {'ms':>9s} {'TFLOP/s*':>10s} {'AI*':>10s}")
    print("-" * 67)
    for r in SOFTMAX:
        print(
            f"{r['dtype']:8s} {r['rows']:8,d} {r['row_length']:9,d} "
            f"{r['ms']:9.4f} {r['tflops']:10.4g} {r['ai']:10.4g}"
        )
    print(f"* Approximate count: {SOFTMAX_OPS_PER_ELEMENT:g} ops/value; minimum input read + output write traffic.")
