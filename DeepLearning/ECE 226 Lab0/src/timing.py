import time
from device import *
from values import *

"""
a@a 在 CUDA 上是异步的: 这行 Python 返回时, 乘法往往还在设备上跑, CPU 已经继续往下走.
sync() 是让 CPU 停下来, 等 GPU 把已经提交的计算做完.
没有它, _time_once 量到的只是把任务丢给设备的时间, 不是矩阵乘法本身的时间.
"""
def _time_once(fn):
    """Crude single-shot timer"""
    fn(); sync() # one throwaway call
    t0 = time.perf_counter()
    fn(); sync() # one timed call
    return (time.perf_counter() - t0) * MS


def clamp(x, range):
    """Clamp x to the range [lo, hi]"""
    lo, hi = range
    return max(lo, min(hi, x))


def bench(fn, budget_ms=600, rep_range=(3, 50)):
    """Median device time in ms, plus the 5th and 95th percentile.

    The repetition count is chosen from a first timing, so a 20-microsecond
    kernel gets 50 reps and a 200-millisecond laptop GEMM still finishes.
    """
    ms0 = max(_time_once(fn), 1e-4)
    rep = budget_ms / ms0
    rep = int(clamp(rep, rep_range))
    # warm up: clocks, caches, autotune
    for _ in range(clamp(rep // 5, (2, 10))):
        fn()
    sync()

    times = []
    if IS_CUDA:
        for _ in range(rep):
            a = torch.cuda.Event(enable_timing=True)
            b = torch.cuda.Event(enable_timing=True)
            a.record(); fn(); b.record()
            torch.cuda.synchronize()
            times.append(a.elapsed_time(b))
    else:
        for _ in range(rep):  # no events on mps/cpu: sync + clock
            t0 = time.perf_counter(); fn(); sync()
            times.append((time.perf_counter() - t0) * MS)

    t = torch.tensor(times, dtype=torch.float64)
    return (
        t.median().item(),
        t.quantile(0.05).item(),
        t.quantile(0.95).item(),
    )
