import json
from device import *
from timing import bench
from ..sizes import elementwise_add_sizes
from ..jsonio import write_compact_result_array


_ADD_CACHE_FILE = device_measure_dir / "operations" / "elementwise_add.json"
_ADD_CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)


def elementwise_add_sweep(dtypes=None):
    """Measure y = a + b across supported dtypes and size sweeps."""
    results = []
    for name, dtype in (dtypes or DTYPES):
        bytes_per_elem = torch.finfo(dtype).bits // 8
        for n in elementwise_add_sizes(name, dtype):
            a = torch.randn(n, device=DEV, dtype=dtype)
            b = torch.randn(n, device=DEV, dtype=dtype)
            out = torch.empty_like(a)
            ms, p5_ms, p95_ms = bench(lambda a=a, b=b, out=out: torch.add(a, b, out=out))
            flops = n  # one addition per element
            ai = flops / (3 * n * bytes_per_elem)  # 2 input reads + 1 output write
            results.append({
                "dtype": name,
                "n": n,
                "ms": ms,
                "spread": (p95_ms - p5_ms) / 2,
                "tflops": (flops / 1e12) / (ms / MS),
                "ai": ai,
            })
            del a, b, out
            free_cache()
    return results


def _load_or_measure_add():
    dtypes = list(DTYPES)
    dtype_names = [name for name, _ in dtypes]
    sizes_by_dtype = {name: elementwise_add_sizes(name, dtype) for name, dtype in dtypes}
    expected_count = sum(len(sizes) for sizes in sizes_by_dtype.values())
    expected_points = {
        (name, n)
        for name, sizes in sizes_by_dtype.items()
        for n in sizes
    }
    try:
        saved = json.loads(_ADD_CACHE_FILE.read_text(encoding="utf-8"))
        results = saved.get("results") if isinstance(saved, dict) else saved
        if (
            isinstance(results, list)
            and len(results) == expected_count
            and all(
                isinstance(row, dict)
                and row.get("dtype") in dtype_names
                and isinstance(row.get("n"), int)
                and (row.get("dtype"), row.get("n")) in expected_points
                and all(isinstance(row.get(k), (int, float))
                        for k in ("ms", "spread", "tflops", "ai"))
                for row in results
            )
            and {(row["dtype"], row["n"]) for row in results} == expected_points
        ):
            results = [
                {key: value for key, value in row.items()
                 if key not in {"p5_ms", "p95_ms", "tflops_low", "tflops_high", "bytes"}}
                for row in results
            ]
            write_compact_result_array(_ADD_CACHE_FILE, results)
            return results
    except (OSError, json.JSONDecodeError):
        pass

    results = elementwise_add_sweep(dtypes=dtypes)
    write_compact_result_array(_ADD_CACHE_FILE, results)
    return results


ELEMENTWISE_ADD = _load_or_measure_add()
