import os
from device import *
from timing import _time_once

BUDGET_MS = float(os.environ.get("LAB0_BUDGET_MS", "150"))


def sweep_gemm_sizes(dtype=torch.float32, start=256, cap_n=16384):
    """Measure square GEMM sizes that fit memory and the sweep time budget."""
    sizes, n = [], start
    bytes_per_element = torch.finfo(dtype).bits // 8
    while n <= cap_n:
        need = 4 * 3 * n * n * bytes_per_element  # inputs, output, and allocation slack
        if need > 0.35 * free_bytes():
            break
        a = torch.randn(n, n, device=DEV, dtype=dtype)
        ms = _time_once(lambda: a @ a)
        del a
        free_cache()
        if ms > BUDGET_MS:
            if not sizes:
                sizes.append(n)
            break
        sizes.append(n)
        n *= 2
    return sizes or [start]


BANDWIDTH_CANDIDATES = (1, 4, 16, 64, 256, 1024, 4096, 8192)


def sweep_bw_sizes_mib(candidates=BANDWIDTH_CANDIDATES):
    """Choose bandwidth buffer sizes in MiB from the available memory budget."""
    cap = 0.35 * free_bytes()
    return [m for m in candidates if 2 * m * MiB < cap * Byte] or [1]


ADD_CANDIDATES = (2**16, 2**18, 2**20, 2**22, 2**23, 2**24, 2**25)


def sweep_elementwise_add_sizes(dtype=torch.bfloat16, candidates=ADD_CANDIDATES):
    """Choose elementwise-add sizes that fit the available memory budget."""
    bytes_per_elem = torch.finfo(dtype).bits // 8
    cap_bytes = 0.20 * free_bytes()
    sizes = [n for n in candidates if 3 * n * bytes_per_elem <= cap_bytes]
    return sizes or [candidates[0]]


SOFTMAX_ROW_LENGTHS = (128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768)
SOFTMAX_TARGET_ELEMENTS = 16 * 2**10  # 16 * 2**20


def sweep_softmax_shapes(
    dtype=torch.bfloat16,
    row_lengths=SOFTMAX_ROW_LENGTHS,
    target_elements=SOFTMAX_TARGET_ELEMENTS,
):
    """Choose softmax row shapes that fit the available memory budget."""
    bytes_per_elem = torch.finfo(dtype).bits // 8
    cap_elements = int(0.20 * free_bytes() / (2 * bytes_per_elem))
    valid_lengths = [n for n in row_lengths if 128 * n <= cap_elements]
    if not valid_lengths:
        valid_lengths = [max(1, cap_elements // 128)]
    target = max(valid_lengths[0] * 128, min(target_elements, cap_elements))
    return [[max(128, target // n), n] for n in valid_lengths]
