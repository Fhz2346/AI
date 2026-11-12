from dataclasses import dataclass
import platform
from pathlib import Path

os_name = platform.system()


@dataclass(frozen=True)
class ConfigGlobal:
    if os_name == "Windows":
        wsf = Path(r"E:\ProjectLocal\ECE 276B\PR2")
    elif os_name == "Linux":
        wsf = Path("/media/fhz/Learning/ProjectLocal/DeepLearning/MLP")
    out_dir = wsf / "output"
    maps_dir = wsf / "maps"

cfgG = ConfigGlobal()
