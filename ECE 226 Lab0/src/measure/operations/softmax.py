import json
from device import *
from timing import bench
from ..sizes import softmax_shapes
from ..jsonio import write_compact_result_array


SOFTMAX_OPS_PER_ELEMENT = 5.0  # max, subtract, exp, sum, normalize (approximate)


def softmax_sweep(dtypes=None):
    """Measure row-wise torch.softmax over row lengths for each supported dtype."""
    results = []
    for name, dtype in (dtypes or DTYPES):
        bytes_per_elem = torch.finfo(dtype).bits // 8
        for rows, row_length in softmax_shapes(name, dtype):
            x = torch.randn((rows, row_length), device=DEV, dtype=dtype)
            ms, p5_ms, p95_ms = bench(lambda x=x: torch.softmax(x, dim=-1))
            elements = rows * row_length
            # Approximate max/sub/exp/sum/div as ~5 operations per input value.
            # Exp and max-reduction are not ordinary FLOPs; this is a labeled model.
            ops = SOFTMAX_OPS_PER_ELEMENT * elements
            traffic = 2 * elements * bytes_per_elem  # minimum: input read + output write
            results.append({
                "dtype": name,
                "rows": rows,
                "row_length": row_length,
                "elements": elements,
                "ms": ms,
                "spread": (p95_ms - p5_ms) / 2,
                "tflops": (ops / 1e12) / (ms / MS),
                "ai": ops / traffic,
            })
            del x
            free_cache()
    return results


_SOFTMAX_CACHE_FILE = device_measure_dir / "operations" / "softmax.json"
_SOFTMAX_CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)


def _load_or_measure_softmax():
    dtypes = list(DTYPES)
    dtype_names = [name for name, _ in dtypes]
    shapes_by_dtype = {
        name: [list(shape) for shape in softmax_shapes(name, dtype)]
        for name, dtype in dtypes
    }
    expected_count = sum(len(shapes) for shapes in shapes_by_dtype.values())
    expected_points = {
        (name, rows, row_length)
        for name, shapes in shapes_by_dtype.items()
        for rows, row_length in shapes
    }
    try:
        saved = json.loads(_SOFTMAX_CACHE_FILE.read_text(encoding="utf-8"))
        results = saved.get("results") if isinstance(saved, dict) else saved
        if (
            isinstance(results, list)
            and len(results) == expected_count
            and all(
                isinstance(row, dict)
                and row.get("dtype") in dtype_names
                and isinstance(row.get("rows"), int)
                and isinstance(row.get("row_length"), int)
                and (row.get("dtype"), row.get("rows"), row.get("row_length")) in expected_points
                and all(isinstance(row.get(k), (int, float))
                        for k in ("elements", "ms", "spread", "tflops", "ai"))
                for row in results
            )
            and {(row["dtype"], row["rows"], row["row_length"]) for row in results}
                == expected_points
        ):
            results = [
                {key: value for key, value in row.items()
                 if key not in {"p5_ms", "p95_ms", "tflops_low", "tflops_high",
                                "bytes_min", "ops_per_element"}}
                for row in results
            ]
            write_compact_result_array(_SOFTMAX_CACHE_FILE, results)
            return results
    except (OSError, json.JSONDecodeError):
        pass

    results = softmax_sweep(dtypes=dtypes)
    write_compact_result_array(_SOFTMAX_CACHE_FILE, results)
    return results


SOFTMAX = _load_or_measure_softmax()
