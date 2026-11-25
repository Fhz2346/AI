from torchprofile import profile_macs
from init_task import *
from measure.roof import ROOF_TFLOPS
from values import *

macs = profile_macs(model, torch.zeros(1, *image_shape, device=DEV, dtype=model_dt))
flops_step = 3 * 2 * macs * BATCH
n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
roof = ROOF_TFLOPS[MODEL_DT]
if __name__ == "__main__":
    # Rough per-step cost, so we know what to expect before we measure it.
    print(f"MACs per image       : {macs/Mega:.1f} M")
    print(f"est. FLOPs per step  : {flops_step/Giga:.1f} GFLOP")
    print(f"trainable parameters : {n_params/Mega:.2f} M")
    print(f"weights at FP32      : {torch.finfo(torch.float32).bits*n_params/MiB:.1f} MiB")
    # Compare against the ceiling you measured in Q2 -- if you kept those names.
    if macs and "ROOF_TFLOPS" in globals() and globals().get("BEST_DT"):
        est_ms = flops_step / (roof * Tera) * MS
        print(f"at your {MODEL_DT} roof that step would take {est_ms:.1f} ms if it ran at peak;")
        print("Q6 measures what it actually takes, and the gap is the subject of the course.")
