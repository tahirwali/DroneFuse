"""
Measure the machine a Kaggle session actually gives us: CPU, RAM (including the container
limit, which can be lower than what /proc/meminfo reports), free disk on every candidate
folder, GPU, package versions and internet access.
"""
import importlib
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

CANDIDATE_FOLDERS = ("/kaggle/working", "/kaggle/temp", "/kaggle/tmp", "/tmp", "/kaggle/input", "/")
PACKAGES = ("torch", "torchvision", "numpy", "scipy", "pandas", "PIL", "soundfile", "librosa",
            "huggingface_hub", "requests", "rosbags", "lz4")


def _meminfo_gb():
    values = {}
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            key, rest = line.split(":", 1)
            if key in ("MemTotal", "MemAvailable"):
                values[key] = round(int(rest.split()[0]) / 1e6, 2)   # kB -> GB
    except OSError:
        pass
    return values


def _cgroup_limit_gb():
    for path in ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
        try:
            text = Path(path).read_text().strip()
        except OSError:
            continue
        if text.isdigit() and int(text) < 1 << 60:
            return round(int(text) / 1e9, 2)
        return None   # "max" or a huge number: no container limit
    return None


def disk_report(folders=CANDIDATE_FOLDERS):
    report = {}
    for folder in folders:
        if os.path.isdir(folder):
            usage = shutil.disk_usage(folder)
            report[folder] = {"total_gb": round(usage.total / 1e9, 1), "free_gb": round(usage.free / 1e9, 1)}
    return report


def pick_scratch_folder(min_free_gb=5.0):
    """Largest writable folder that Kaggle does NOT save with the notebook (never /kaggle/working)."""
    best, best_free = None, -1.0
    for folder in ("/kaggle/temp", "/kaggle/tmp", "/tmp"):
        if os.path.isdir(folder) and os.access(folder, os.W_OK):
            free = shutil.disk_usage(folder).free / 1e9
            if free > best_free:
                best, best_free = folder, free
    if best is None or best_free < min_free_gb:
        print(f"Warning: largest scratch folder {best} has only {best_free:.1f} GB free.")
    return Path(best or "/tmp") / "dronefuse_scratch"


def _gpu():
    if shutil.which("nvidia-smi") is None:
        return None
    result = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                            capture_output=True, text=True)
    return result.stdout.strip() or None


def _versions():
    versions = {}
    for name in PACKAGES:
        try:
            versions[name] = getattr(importlib.import_module(name), "__version__", "installed")
        except Exception:
            versions[name] = None
    return versions


def _internet(url="https://huggingface.co", timeout=15):
    try:
        import requests
        return requests.get(url, timeout=timeout).status_code
    except Exception as error:
        return f"no ({type(error).__name__})"


def collect():
    memory = _meminfo_gb()
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "kaggle_run_type": os.environ.get("KAGGLE_KERNEL_RUN_TYPE"),   # "Batch" during Save & Run All
        "cpu_count": os.cpu_count(),
        "ram_total_gb": memory.get("MemTotal"),
        "ram_available_gb": memory.get("MemAvailable"),
        "ram_container_limit_gb": _cgroup_limit_gb(),
        "disk": disk_report(),
        "gpu": _gpu(),
        "packages": _versions(),
        "internet_status": _internet(),
    }
