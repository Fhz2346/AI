import os
import platform
import re
import shutil
import subprocess
import torch

from ..init import *
from values import *


def device_report(idx=0):
    """Everything PyTorch will tell you about the accelerator it found."""
    if not HAS_GPU:
        return {}
    info = {"backend": ACCEL}
    if IS_CUDA:
        p = torch.cuda.get_device_properties(idx)
        info.update({
            "name":            p.name,
            "arch":            getattr(p, "gcnArchName", None) or f"sm{p.major}{p.minor}",
            "cu_or_sm_count":  p.multi_processor_count,
            "wave_or_warp":    getattr(p, "warp_size", None),
            "total_mem_GiB":   round(p.total_memory / 2**30, 2),
            "vendor_runtime":  f"ROCm {torch.version.hip}" if torch.version.hip
                               else f"CUDA {torch.version.cuda}",
            "n_devices":       torch.cuda.device_count(),
        })
        for extra in ("L2_cache_size", "max_threads_per_multi_processor",
                      "regs_per_multiprocessor", "shared_memory_per_block"):
            if hasattr(p, extra):
                info[extra] = getattr(p, extra)
    else:
        # MPS and the out-of-tree backends expose far less. Take what there is.
        info.update({
            "name": host_name() + f"  (integrated GPU via {ACCEL})",
            "arch": platform.machine(),
            "total_mem_GiB": round(free_bytes() / 2**30, 2),
        })
    return info


def host_name():
    """A readable name for the CPU, for labelling plots on a laptop run."""
    try:
        if platform.system() == "Linux":
            for line in open("/proc/cpuinfo"):
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return platform.processor() or platform.machine() or "unknown CPU"


def free_bytes():
    """Roughly how much device memory we may help ourselves to."""
    if IS_CUDA:
        try:
            return torch.cuda.mem_get_info()[0]
        except Exception:
            return torch.cuda.get_device_properties(0).total_memory // 2
    rec = getattr(getattr(torch, "mps", None), "recommended_max_memory", None)
    if rec is not None:
        try:
            return int(rec())
        except Exception:
            pass
    try:  # host RAM, quarter of it, for a CPU run
        return int(0.25 * os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"))
    except Exception:
        return 2 * GiB / Byte  # conservative: somebody in the class has an 8 GB laptop


def query_clock_ghz():
    """PyTorch does not expose the peak clock. Ask the vendor tool, if it is installed."""
    probes = (
        ["nvidia-smi", "--query-gpu=clocks.max.sm", "--format=csv,noheader,nounits"],
        ["rocm-smi", "--showmaxclocks"],
        ["rocm-smi", "--showclocks"],
    )
    for cmd in probes:
        if shutil.which(cmd[0]) is None:
            continue
        try:
            out = subprocess.run(cmd, capture_output=True, text=True, timeout=30).stdout
        except Exception:
            continue
        mhz = [float(m) for m in re.findall(r"(\d{3,5})\s*[Mm][Hh]z", out)]
        if not mhz:
            mhz = [float(m) for m in re.findall(r"^\s*(\d{3,5})\s*$", out, re.M)]
        if mhz:
            return max(mhz) / 1e3
    return None
