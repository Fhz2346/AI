from measure.roof import *


if __name__ == "__main__":
    print("usable precisions:", [n for n, _ in DTYPES], "\n")
    print(
        f"{'dtype':6s} {'N':>6s} {'ms':>9s} {'+/- ms':>8s} {'TFLOP/s':>10s} "
        f"{'implied FLOP/CU/clk':>21s}"
    )
    print("-" * 66)
    for name, r in ROOF.items():
        implied = f"{r['tflops']*1e12/(CU*CLOCK_GHZ*1e9):21.0f}" if CU and CLOCK_GHZ else " " * 21
        print(
            f"{name:6s} {r['n']:6d} {r['ms']:9.3f} {r['spread']:8.3f} "
            f"{r['tflops']:10.4g} {implied}"
        )
    if BEST_DT:
        print(
            f"\ncompute roof : {ROOF_TFLOPS[BEST_DT]:.4g} TFLOP/s at {BEST_DT} "
            f"(N={ROOF[BEST_DT]['n']})"
        )
    if not (CU and CLOCK_GHZ):
        missing = " and ".join(x for x, ok in (("the CU/SM count", CU), ("the peak clock", CLOCK_GHZ)) if not ok)
        print(f"\nThe last column is blank: {missing} could not be determined here.")
        print("To finish the inversion you would need the vendor product page (clock) and")
        print("the vendor tool or datasheet (core count). Say so in your report -- the")
        print("measured roof above stands on its own, and it is the number the roofline uses.")
