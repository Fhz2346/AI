import json
from device import *
from timing import bench
from ..sizes import gemm_sizes
from ..roof import ROOF_TFLOPS
from ..jsonio import write_compact_result_array


def gemm_sweep(dtypes=None):
    out = []
    for name, dtype in (dtypes or DTYPES):
        bytes_per_elem = torch.finfo(dtype).bits // 8
        roof = ROOF_TFLOPS.get(name)
        for n in gemm_sizes(name, dtype):
            a = torch.randn(n, n, device=DEV, dtype=dtype)
            b = torch.randn(n, n, device=DEV, dtype=dtype)
            c = torch.empty(n, n, device=DEV, dtype=dtype)
            ms, lo, hi = bench(lambda a=a, b=b, c=c: torch.mm(a, b, out=c))
            del a, b, c
            free_cache()
            flop = 2 * n**3
            tflop = flop / 1e12
            tflops = tflop / (ms / MS)
            out.append(
                {
                    "dtype": name,
                    "n": n,
                    "ms": ms,
                    "spread": (hi - lo) / 2,
                    "tflops": tflops,
                    "pct": 100 * tflops / roof if roof else float("nan"),  # percentage
                    "ai": (2 * n) / (3 * bytes_per_elem),  # arithmetic intensity
                }
            )
    return out


_GEMM_CACHE_FILE = device_measure_dir / "operations" / "gemm.json"
_GEMM_CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)


def _load_or_measure_gemm():
    dtypes = DTYPES
    dtype_names = [name for name, _ in dtypes]
    sizes = {name: list(gemm_sizes(name, dtype)) for name, dtype in dtypes}
    roof = {name: ROOF_TFLOPS.get(name) for name in dtype_names}
    expected_count = sum(len(values) for values in sizes.values())
    expected_points = {
        (name, n)
        for name, dtype in dtypes
        for n in sizes[name]
    }

    try:
        saved = json.loads(_GEMM_CACHE_FILE.read_text(encoding="utf-8"))
        results = saved.get("results") if isinstance(saved, dict) else saved
        valid = (
            isinstance(results, list)
            and len(results) == expected_count
            and all(
                isinstance(record, dict)
                and record.get("dtype") in dtype_names
                and isinstance(record.get("n"), int)
                and (record.get("dtype"), record.get("n")) in expected_points
                and all(isinstance(record.get(key), (int, float))
                        for key in ("ms", "spread", "tflops", "pct", "ai"))
                and roof.get(record.get("dtype")) is not None
                and abs(record["pct"] - 100 * record["tflops"] / roof[record["dtype"]]) < 1e-6
                for record in results
            )
            and {(record["dtype"], record["n"]) for record in results} == expected_points
        )
        if valid:
            write_compact_result_array(_GEMM_CACHE_FILE, results)
            return results
    except (OSError, json.JSONDecodeError):
        pass

    results = gemm_sweep(dtypes=dtypes)
    write_compact_result_array(_GEMM_CACHE_FILE, results)
    return results


GEMM = _load_or_measure_gemm()
