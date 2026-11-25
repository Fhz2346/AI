from measure import ELEMENTWISE_ADD


if __name__ == "__main__":
    print(f"{'dtype':8s} {'elements':>12s} {'ms':>9s} {'TFLOP/s':>10s} {'AI (F/B)':>10s}")
    print("-" * 55)
    for r in ELEMENTWISE_ADD:
        print(f"{r['dtype']:8s} {r['n']:12,d} {r['ms']:9.4f} {r['tflops']:10.4g} {r['ai']:10.4g}")
    print("AI assumes 1 addition/value and 2 input reads + 1 output write.")
