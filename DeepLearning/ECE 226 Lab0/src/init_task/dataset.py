from torch.utils.data import DataLoader
from device import *

USE_SYNTHETIC = os.environ.get("LAB0_SYNTHETIC") == "1"
BATCH = 256 if IS_CUDA else (64 if HAS_GPU else 16)
# Worker processes need fork. On macOS and Windows, Python spawns and re-imports
# the main module, which a notebook cell cannot provide -- so load in-process there.
WORKERS = 2 if (platform.system() == "Linux" and (os.cpu_count() or 1) >= 4) else 0
image_size = 32
image_shape = (3, image_size, image_size)
num_labels = 10

MODEL_DT = os.environ.get("MODEL_DT")
model_dt = name2dtype[MODEL_DT]


def synthetic_dataset(n=512):
    from torch.utils.data import TensorDataset
    x = torch.randn(n, *image_shape).to(dtype=model_dt)
    y = torch.randint(0, num_labels, (n,)).to(dtype=model_dt)
    return {s: TensorDataset(x, y) for s in ("train", "test")}


dataset, SOURCE = None, "CIFAR-10"
if not USE_SYNTHETIC:
    try:
        from torchvision.datasets import CIFAR10
        from torchvision.transforms import (
            Compose,
            RandomCrop,
            RandomHorizontalFlip,
            ToTensor,
            Lambda
        )
        transforms = {
            "train": Compose(
                [
                    RandomCrop(image_size, padding=4),
                    RandomHorizontalFlip(),
                    ToTensor(),
                    Lambda(lambda x: x.to(dtype=model_dt)),
                ]
            ),
            "test": Compose(
                [
                    ToTensor(),
                    Lambda(lambda x: x.to(dtype=model_dt)),
                ]
            ),
        }
        dataset = {
            s: CIFAR10(
                r"E:\Dataset\cifar10",
                train=(s == "train"),
                download=False,
                transform=transforms[s],
            )
            for s in ("train", "test")
        }
    except Exception as e:
        print(f"CIFAR-10 unavailable ({type(e).__name__}: {str(e)[:90]})")
        dataset = None

if dataset is None:
    dataset, SOURCE = synthetic_dataset(), "random tensors (CIFAR-10 not available)"
    print("Falling back to", SOURCE + ".")
    print("The profile below is still valid; the loss value is not.")

loader = {
    s: DataLoader(
        dataset[s],
        batch_size=BATCH,
        shuffle=(s == "train"),
        num_workers=WORKERS,
        pin_memory=IS_CUDA,
        drop_last=(s == "train"),
    )
    for s in ("train", "test")
}


def cycle(dl):
    """Endless batches, so a benchmark loop never trips over StopIteration."""
    while True:
        for batch in dl:
            yield batch


it = cycle(loader["train"])

if __name__ == "__main__":
    print(f"data source : {SOURCE}")
    print(f"batch size  : {BATCH}")
    print(f"dataloader workers: {WORKERS}")
    print({s: len(loader[s]) for s in loader}, "batches")
    images, labels = next(iter(loader["train"]))
    print(f"images shape: {images.shape}")
    print(f"labels shape: {labels.shape}")
