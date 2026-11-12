from dataclasses import dataclass
import platform
from pathlib import Path

os_name = platform.system()


@dataclass(frozen=True)
class ConfigGlobal:
    if os_name == "Windows":
        wsf = Path(r"E:\ProjectLocal\DeepLearning\MLP")
        data_dir = Path(r"E:\Dataset")
    elif os_name == "Linux":
        wsf = Path("/media/fhz/Learning/ProjectLocal/DeepLearning/MLP")
        data_dir = Path("/media/fhz/Learning/Dataset")

    out_dir = wsf / "output"


cfgG = ConfigGlobal()
