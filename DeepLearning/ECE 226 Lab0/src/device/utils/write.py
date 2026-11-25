from ..init import *
import torch


def sync():
    """Accelerators are asynchronous. Never time one without this."""
    if IS_CUDA:
        torch.cuda.synchronize()


def free_cache():
    if IS_CUDA:
        torch.cuda.empty_cache()

