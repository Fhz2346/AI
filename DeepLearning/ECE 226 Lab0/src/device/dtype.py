import torch
from .init import *
from .utils.write import *

ALL_DTYPES = [
    ("fp64", torch.float64),
    ("fp32", torch.float32),
    ("bf16", torch.bfloat16),
    ("fp16", torch.float16),
]

name2dtype = dict(ALL_DTYPES)
dtype2name = dtype2name = {dtype: name for name, dtype in ALL_DTYPES}


def supported_dtypes(candidates=None, n=64):
    """Which precisions this device will actually multiply. Probe, do not assume."""
    ok = []
    for name, dt in (candidates or ALL_DTYPES):
        try:
            x = torch.ones(n, n, device=DEV, dtype=dt)
            y = x @ x
            sync()
            if not torch.isfinite(y.float()).all():
                raise RuntimeError("non-finite result")
            ok.append((name, dt))
            del x, y
        except Exception as e:
            print(f"  {name:5s} unusable here: {type(e).__name__}: {str(e)[:70]}")
    free_cache()
    return ok
