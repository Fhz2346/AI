from device import *
from timing import bench
from .sizes import gemm_sizes
from .jsonio import write_compact_mapping
import json

# roof = best
def compute_roof(dtypes):
    """Best sustained GEMM throughput per dtype = this device's attainable roof.

    Take the best over every shape that fits, not just the largest one. 
    On a big GPU the largest always wins; 
    on a CPU a matrix that still fits in cache can beat it, and the ceiling is whichever one actually went fastest. 
    (don't forget to delete tensors and reset the cache on each eval)
    """
    roof = {}
    for name, dtype in dtypes:
        best = None
        for n in gemm_sizes(name, dtype):
            a = torch.randn(n, n, device=DEV, dtype=dtype)
            b = torch.randn(n, n, device=DEV, dtype=dtype)
            c = torch.empty(n, n, device=DEV, dtype=dtype)
            ms, lo, hi = bench(lambda a=a, b=b, c=c: torch.mm(a, b, out=c))
            del a, b, c
            free_cache()
            rec = {
                "n": n,
                "ms": ms,
                "spread": (hi - lo) / 2,
                "tflops": (2 * n**3) / 1e12 / (ms / MS),
            }
            if best is None or rec["tflops"] > best["tflops"]:
                best = rec
        roof[name] = best
    return roof

_ROOF_CACHE_FILE = device_measure_dir / "roof.json"
_ROOF_CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)


def _load_or_measure_roof():
    roof_names = [name for name, _ in DTYPES]
    try:
        saved_roof = json.loads(_ROOF_CACHE_FILE.read_text(encoding="utf-8"))
        valid_roof = (
            isinstance(saved_roof, dict)
            and set(saved_roof) == set(roof_names)
            and all(
                isinstance(record, dict)
                and isinstance(record.get("n"), int)
                and all(isinstance(record.get(key), (int, float))
                        for key in ("ms", "spread", "tflops"))
                for record in saved_roof.values()
            )
        )
        if valid_roof:
            write_compact_mapping(_ROOF_CACHE_FILE, saved_roof)
            return saved_roof
    except (OSError, json.JSONDecodeError):
        pass

    roof = compute_roof(DTYPES)
    write_compact_mapping(_ROOF_CACHE_FILE, roof)
    return roof


ROOF = _load_or_measure_roof()

ROOF_TFLOPS = {k: v["tflops"] for k, v in ROOF.items()}
BEST_DT = max(ROOF_TFLOPS, key=ROOF_TFLOPS.get) if ROOF_TFLOPS else None
