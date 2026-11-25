from .sweep import *
from .cache_utils import *


def gemm_sizes(name, dtype):
    """Load cached GEMM sizes for a dtype, sweeping and caching on a miss."""
    return cached_dtype_value("gemm", name, lambda: sweep_gemm_sizes(dtype))


def bw_sizes_mib(candidates=BANDWIDTH_CANDIDATES):
    """Load cached bandwidth sizes in MiB, calculating and caching on a miss."""
    if tuple(candidates) != BANDWIDTH_CANDIDATES:
        return sweep_bw_sizes_mib(candidates)
    return cached_operation_value(
        "bandwidth", lambda: sweep_bw_sizes_mib(candidates)
    )


def _dtype_name(dtype):
    return next((name for name, dt in DTYPES if dt == dtype), str(dtype))


def elementwise_add_sizes(name, dtype, candidates=ADD_CANDIDATES):
    """Load cached elementwise-add sizes, sweeping and caching on a miss."""
    if tuple(candidates) != ADD_CANDIDATES:
        return sweep_elementwise_add_sizes(dtype, candidates)
    return cached_dtype_value(
        "elementwise_add", name or _dtype_name(dtype),
        lambda: sweep_elementwise_add_sizes(dtype, candidates),
    )


def softmax_shapes(name, dtype, row_lengths=SOFTMAX_ROW_LENGTHS,
                   target_elements=SOFTMAX_TARGET_ELEMENTS):
    """Load cached softmax shapes, calculating and caching on a miss."""
    if tuple(row_lengths) != SOFTMAX_ROW_LENGTHS or target_elements != SOFTMAX_TARGET_ELEMENTS:
        return [tuple(shape) for shape in sweep_softmax_shapes(dtype, row_lengths, target_elements)]
    shapes = cached_dtype_value(
        "softmax", name or _dtype_name(dtype),
        lambda: sweep_softmax_shapes(dtype, row_lengths, target_elements),
    )
    return [tuple(shape) for shape in shapes]
