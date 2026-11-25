import os
import numpy as np
import torch

torch.manual_seed(0)
np.random.seed(0)


def pick_device():
    """First accelerator we can actually use; CPU if there is none.

    Set LAB0_DEVICE=cpu to force the host, which is how you compare your GPU
    against your own CPU -- run the notebook twice and keep both tables.
    """
    # select want if LAB0_DEVICE is set
    want = os.environ.get("LAB0_DEVICE", "").strip().lower()
    if want:
        kind = "rocm" if (want == "cuda" and torch.version.hip) else want
        return torch.device(want), kind
    # select available device if LAB0_DEVICE is not set
    if torch.cuda.is_available():
        return torch.device("cuda"), ("rocm" if torch.version.hip else "cuda")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps"), "mps"
    for name in ("xpu", "npu"):  # Intel GPUs, vendor out-of-tree backends
        be = getattr(torch, name, None)
        if be is not None and getattr(be, "is_available", lambda: False)():
            return torch.device(name), name
    return torch.device("cpu"), "cpu"


DEV, ACCEL = pick_device()
HAS_GPU = ACCEL != "cpu"
IS_CUDA = DEV.type == "cuda"  # True on ROCm too: PyTorch shares the namespace
