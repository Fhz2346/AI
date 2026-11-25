from .utils.read import *
from .utils.write import *
from .dtype import *
from Pathes import P
import json
import re

if IS_CUDA:
    DEVICE_NAME = torch.cuda.get_device_properties(0).name
elif HAS_GPU:
    DEVICE_NAME = f"{host_name()}  (integrated GPU via {ACCEL})"
else:
    DEVICE_NAME = host_name()

# Device names can contain characters that are not valid in Windows filenames.
device_name = re.sub(r'[<>:"/\\|?*]+', "_", DEVICE_NAME).rstrip(" .")
device_report_file = P.out_dir / device_name / "report.json"
device_out_dir = P.out_dir / device_name
device_measure_dir = device_out_dir / "roofline" / "json"


INFO = None
saved = {}
cache_hit = False
try:
    saved = json.loads(device_report_file.read_text(encoding="utf-8"))
    if isinstance(saved, dict) and saved:
        INFO = saved
        cache_hit = True
except (OSError, json.JSONDecodeError):
    saved = {}

if INFO is None:
    INFO = device_report()

# Keep all hardware details together at the JSON top level.
if "CLOCK_GHZ" not in INFO:
    INFO["CLOCK_GHZ"] = query_clock_ghz()
    cache_hit = False

cached_dtype_names = INFO.get("DTYPES")
if not isinstance(cached_dtype_names, list) or any(
    not isinstance(name, str) or name not in name2dtype for name in cached_dtype_names
):
    DTYPES_pairs = supported_dtypes()
    INFO["DTYPES"] = [name for name, _ in DTYPES_pairs]
    cache_hit = False
else:
    DTYPES_pairs = [(name, name2dtype[name]) for name in cached_dtype_names]

if not cache_hit:
    device_report_file.parent.mkdir(parents=True, exist_ok=True)
    device_report_file.write_text(
        json.dumps(INFO, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

CLOCK_GHZ = INFO["CLOCK_GHZ"]
CU = INFO.get("cu_or_sm_count")
DTYPES = DTYPES_pairs
