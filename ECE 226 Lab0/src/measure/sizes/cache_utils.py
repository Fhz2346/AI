import json
from device import *

_SIZE_CACHE_FILE = device_measure_dir / "sizes.json"
_SIZE_CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)


def is_int_list(value):
    return (
        isinstance(value, list)
        and all(isinstance(item, int) and not isinstance(item, bool) for item in value)
    )


def load_size_cache():
    try:
        saved = json.loads(_SIZE_CACHE_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(saved, dict):
        return {}

    operation_names = {"gemm", "bandwidth", "elementwise_add", "softmax"}
    if operation_names.intersection(saved):
        return saved

    # Migrate the previous dtype -> GEMM sizes format in memory.
    if all(isinstance(name, str) and is_int_list(sizes) for name, sizes in saved.items()):
        return {"gemm": saved}
    return {}


SIZE_CACHE = load_size_cache()


def save_size_cache():
    lines = ["{"]
    operations = list(SIZE_CACHE.items())
    for op_index, (operation, values) in enumerate(operations):
        op_comma = "," if op_index < len(operations) - 1 else ""
        if isinstance(values, dict):
            lines.append(f"  {json.dumps(operation)}: {{")
            dtypes = list(values.items())
            for dtype_index, (dtype_name, sizes) in enumerate(dtypes):
                comma = "," if dtype_index < len(dtypes) - 1 else ""
                compact_sizes = json.dumps(sizes, separators=(",", ": "))
                lines.append(f"    {json.dumps(dtype_name)}: {compact_sizes}{comma}")
            lines.append(f"  }}{op_comma}")
        else:
            compact_value = json.dumps(values, separators=(",", ": "))
            lines.append(f"  {json.dumps(operation)}: {compact_value}{op_comma}")
    lines.append("}")
    _SIZE_CACHE_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")


def cached_operation_value(operation, compute):
    value = SIZE_CACHE.get(operation)
    if operation == "bandwidth" and not is_int_list(value):
        value = None
    if value is None:
        value = compute()
        SIZE_CACHE[operation] = value
        save_size_cache()
    return value


def cached_dtype_value(operation, dtype_name, compute):
    values = SIZE_CACHE.setdefault(operation, {})
    if dtype_name not in values or not is_valid_dtype_sizes(operation, values[dtype_name]):
        values[dtype_name] = compute()
        save_size_cache()
    return values[dtype_name]


def is_valid_dtype_sizes(operation, value):
    if operation in {"gemm", "elementwise_add"}:
        return is_int_list(value)
    if operation == "softmax":
        return (
            isinstance(value, list)
            and all(
                isinstance(shape, list)
                and len(shape) == 2
                and all(isinstance(dim, int) and not isinstance(dim, bool) for dim in shape)
                for shape in value
            )
        )
    return False
