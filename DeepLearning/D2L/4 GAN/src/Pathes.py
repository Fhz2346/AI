from dataclasses import dataclass
import platform
from pathlib import Path

os_name = platform.system()


@dataclass(frozen=True)
class Pathes:
    wsf = Path("E:\OneDrive\ProjectCloud\AI\DeepLearning\D2L\GAN")
    data_dir = wsf / "data"
    out_dir = wsf / "output"
    ckpt_dir = wsf / "ckpt"
    cfg_dir = wsf / "config"


P = Pathes()