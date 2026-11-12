from dataclasses import dataclass
import platform
from pathlib import Path

os_name = platform.system()


@dataclass(frozen=True)
class ConfigGlobal:
    if os_name == "Windows":
        wsf = Path(r"E:\ProjectLocal\ECE 276B\PR1")
        data_dir = Path(r"E:\Dataset\D2L")
    elif os_name == "Linux":
        wsf = Path("/media/fhz/Learning/ProjectLocal/DeepLearning/D2L/Sequence")
        data_dir = Path("/media/fhz/Learning/Dataset/D2L")

    out_dir = wsf / "output"

cfgG = ConfigGlobal()
