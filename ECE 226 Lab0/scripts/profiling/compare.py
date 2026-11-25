from timing import bench
from categorise import *
from predict import *

ms_step, lo, hi = bench(lambda: train_step(it))
step_ms = grand / 1e3 / ACTIVE  # profiled self time per active step
gap_ms = ms_step - step_ms

if __name__ == "__main__":
    print(f"wall clock per step : {ms_step:8.2f} ms   (p5 {lo:.2f}, p95 {hi:.2f})")
    print(
        f"profiled self time  : {step_ms:8.2f} ms "
        f"({'device' if USED_DEVICE_TIME else 'host'})"
    )
    print(f"unaccounted         : {gap_ms:8.2f} ms ({100*gap_ms/ms_step:.0f}%)")
    if flops_step:
        wall_tflops = flops_step / Tera / (ms_step / MS)
        busy_tflops = flops_step / Tera / (step_ms / MS)
        print(f"\nFLOPs per step      : {flops_step/1e9:8.1f} GFLOP")
        print(f"achieved (wall)     : {wall_tflops:8.2f} TFLOP/s", end="")
        print(f"   = {100*wall_tflops/roof:.1f}% of {MODEL_DT} roof {roof:.2f}" if roof else "")
        print(f"achieved (kernels)  : {busy_tflops:8.2f} TFLOP/s", end="")
        print(f"   = {100*busy_tflops/roof:.1f}% of {MODEL_DT} roof {roof:.2f}" if roof else "")
