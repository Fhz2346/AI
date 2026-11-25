from torch.autograd import DeviceType
import json
from device import *
from pathlib import Path


def _self_time(e, use_device):
    if use_device:
        for attr in ("self_device_time_total", "self_cuda_time_total"):
            t = getattr(e, attr, 0) or 0
            if t:
                return t
        return 0.0
    return e.self_cpu_time_total


def _is_kernel(e):
    # aten:: ops also report the device time of the kernels they launch, and
    # ProfilerStep* / Optimizer.step are annotations spanning other kernels;
    # counting either alongside the kernel rows double-counts device time.
    return (
        e.device_type != DeviceType.CPU
        and not getattr(e, "is_user_annotation", False)
        and not e.key.startswith(("ProfilerStep", "Optimizer."))
    )

categories_path = Path(__file__).resolve().parent / "categories.json"
with categories_path.open("r", encoding="utf-8") as f:
    CATEGORIES = json.load(f)


def categorise(prof: torch.profiler.profile):
    evs = list(prof.key_averages())
    use_device = HAS_GPU and sum(_self_time(e, True) for e in evs) > 0
    if use_device:
        evs = [e for e in evs if _is_kernel(e)]
    if not use_device:
        print("No device-side timing available (CPU run, or this backend has no profiler support),")
        print("so the shares below are host-side operator time.")
        print("The categories mean the same thing; say which one you measured.\n")

    tot = {k: 0.0 for k in CATEGORIES}
    tot["other"] = 0.0
    grand = 0.0
    for e in evs:
        t = _self_time(e, use_device)
        if t <= 0:
            continue
        grand += t
        low = e.key.lower()
        for cat, keys in CATEGORIES.items():
            if any(k.lower() in low for k in keys):
                tot[cat] += t
                break
        else:
            tot["other"] += t
    return tot, grand, use_device
