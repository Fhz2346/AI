from init_task import *
from torch.profiler import profile, schedule

model.train()
acts = list(torch.profiler.supported_activities())
ACTIVE = 5
with profile(
    activities=acts,
    schedule=schedule(wait=1, warmup=3, active=ACTIVE, repeat=1),
    record_shapes=True,
) as prof:
    for i in range(1 + 3 + ACTIVE):
        train_step(it)
        if i == 3:
            # The host runs steps ahead of the GPU;
            # drain the queue so warm-up kernels are not recorded as part of the active window.
            sync()
        prof.step()
prof.export_chrome_trace(str(P.out_dir / device_name / "profiling/trace.json"))

if __name__ == "__main__":
    print("profiler activities:", [str(a).split(".")[-1] for a in acts])
    # Attribute names moved between PyTorch versions; try the current one first.
    if HAS_GPU:
        keys = ("self_device_time_total", "self_cuda_time_total", "self_cpu_time_total")
    else:
        keys = ("self_cpu_time_total",)
    for sort_key in keys:
        try:
            print(prof.key_averages().table(sort_by=sort_key, row_limit=18))
            break
        except Exception:
            continue
    print("\nwrote trace.json  (open in Perfetto, or in AMD TraceLens)")
