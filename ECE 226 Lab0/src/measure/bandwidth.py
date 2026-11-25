from device import *
from timing import bench
from .sizes import bw_sizes_mib
import json

BW_DTYPE = dict(DTYPES).get("fp16") or dict(DTYPES).get("fp32", torch.float32)
_BW_CACHE_FILE = device_measure_dir / "bandwidth.json"
_BW_CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)


# bandwidth = speed of data move
def bandwidth_sweep(sizes_mib=None, dtype=None):
    dtype = dtype or BW_DTYPE
    bytes_per_elem = torch.finfo(dtype).bits // 8
    out = []
    for mib in (sizes_mib or bw_sizes_mib()):
        n = int(mib * MiB // torch.finfo(dtype).bits)  # number of elements to allocate
        src = torch.randn(n, device=DEV, dtype=dtype)
        dst = torch.empty(n, device=DEV, dtype=dtype)
        ms, lo, hi = bench(lambda src=src, dst=dst: dst.copy_(src))
        del src, dst
        free_cache()
        out.append(
            {
                "mib": mib,
                "ms": ms,
                "spread": (hi - lo) / 2,
                "gbs": ((2 * n * bytes_per_elem) * Byte / GiB) / (ms / MS),  # 2=read and write
            }
        )
    return out


def _load_or_measure_bandwidth():
    sizes_mib = bw_sizes_mib()
    def _dtype_name(dtype):
        return next((name for name, dt in DTYPES if dt == dtype), str(dtype))

    dtype_name = _dtype_name(BW_DTYPE)
    try:
        saved = json.loads(_BW_CACHE_FILE.read_text(encoding="utf-8"))
        results = saved.get("results") if isinstance(saved, dict) else None
        valid = (
            isinstance(saved, dict)
            and saved.get("dtype") == dtype_name
            and saved.get("sizes_mib") == sizes_mib
            and isinstance(results, list)
            and len(results) == len(sizes_mib)
            and all(
                isinstance(record, dict)
                and record.get("mib") == mib
                and all(isinstance(record.get(key), (int, float))
                        for key in ("ms", "spread", "gbs"))
                for record, mib in zip(results, sizes_mib)
            )
        )
        if valid:
            return results
    except (OSError, json.JSONDecodeError):
        pass

    results = bandwidth_sweep(sizes_mib=sizes_mib, dtype=BW_DTYPE)
    lines = ["{", f'  "dtype": {json.dumps(dtype_name)},',
             f'  "sizes_mib": {json.dumps(sizes_mib)},', '  "results": [']
    for i, record in enumerate(results):
        comma = "," if i < len(results) - 1 else ""
        lines.append(f"    {json.dumps(record, separators=(',', ': '))}{comma}")
    lines.extend(["  ]", "}"])
    _BW_CACHE_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return results


BW = _load_or_measure_bandwidth()
ACHIEVED_BW_TBS = (BW[-1]["gbs"] / 1e3) if BW else float("nan") # pick the largest buffer measured
