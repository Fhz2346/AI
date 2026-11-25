from measure import GEMM

if __name__ == "__main__":
    print(f"{'dtype':6s} {'N':>6s} {'ms':>9s} {'TFLOP/s':>10s} {'% of roof':>10s} {'I (F/B)':>9s}")
    print("-" * 56)
    for r in GEMM:
        print(f"{r['dtype']:6s} {r['n']:6d} {r['ms']:9.3f} {r['tflops']:10.4g} {r['pct']:9.1f}% {r['ai']:9.0f}")
