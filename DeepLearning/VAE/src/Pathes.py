from dataclasses import dataclass
import platform
from pathlib import Path

os_name = platform.system()


@dataclass(frozen=True)
class Pathes:
    wsf = Path(r"E:\OneDrive\ProjectCloud\AI\DeepLearning\VAE")
    ckpt_dir = wsf / "ckpt"
    out_dir = wsf / "output"


P = Pathes()
