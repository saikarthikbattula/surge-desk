"""Environment check. First cell of every notebook: `from surgedesk.envcheck import run; run()`.

Prints architecture, resources, Telum/NNPA status, package versions, and which
data/model files exist. Warns, never fails, so it also runs on laptops.

CLI: python -m surgedesk.envcheck
Owner: foundation (frozen).
"""
from __future__ import annotations

import glob
import importlib
import importlib.metadata as md
import importlib.util
import os
import platform
import shutil
import sys
from pathlib import Path

from . import config

PACKAGES = [
    "numpy", "pandas", "scikit-learn", "xgboost", "torch", "transformers", "tokenizers",
    "huggingface_hub", "llama_cpp_python", "ipywidgets", "matplotlib", "joblib",
    "torch_nnpa", "nbconvert",
]


def _mem_gb() -> float | None:
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    return int(line.split()[1]) / 1024 / 1024
    except OSError:
        pass
    return None


def _nnpa_flag() -> bool | None:
    try:
        with open("/proc/cpuinfo") as f:
            return "nnpa" in f.read()
    except OSError:
        return None


def _zdnn_libs() -> list[str]:
    found: list[str] = []
    try:
        spec = importlib.util.find_spec("llama_cpp")
        if spec and spec.submodule_search_locations:
            for loc in spec.submodule_search_locations:
                found += glob.glob(os.path.join(loc, "lib", "*zdnn*"))
    except Exception:
        pass
    return found


def collect() -> dict:
    info: dict = {
        "arch": platform.machine(),
        "byteorder": sys.byteorder,
        "python": sys.version.split()[0],
        "cores": os.cpu_count(),
        "ram_gb": _mem_gb(),
        "disk_free_gb": round(shutil.disk_usage(str(config.ROOT)).free / 1e9, 1),
        "nnpa_flag": _nnpa_flag(),
        "zdnn_libs": _zdnn_libs(),
        "root": str(config.ROOT),
        "threads_setting": config.N_THREADS,
        "demo_event": config.DEMO_EVENT,
        "packages": {},
        "paths": {},
    }
    for p in PACKAGES:
        try:
            info["packages"][p] = md.version(p)
        except md.PackageNotFoundError:
            info["packages"][p] = None
    paths = {
        "humaid_dir": config.HUMAID_DIR,
        "gazetteer_dir": config.GAZETTEER_DIR,
        "svi_csv": config.SVI_PATH,
        "minilm_dir": config.MINILM_DIR,
        "granite_gguf": config.GRANITE_PATH,
        "classifier": config.CLASSIFIER_PATH,
        "ranker": config.RANKER_PATH,
        "labels_dir": config.LABELS_DIR,
        "outputs_demo": config.OUTPUTS / config.DEMO_EVENT,
    }
    for k, p in paths.items():
        p = Path(p)
        info["paths"][k] = {"path": str(p), "exists": p.exists()}
    info["on_linuxone"] = info["arch"] == "s390x"
    return info


def run(verbose: bool = True) -> dict:
    info = collect()
    if not verbose:
        return info
    print("=" * 64)
    print("SurgeDesk environment check")
    print("=" * 64)
    print(f"arch            : {info['arch']}  (byte order: {info['byteorder']})")
    if not info["on_linuxone"]:
        print("WARNING: not running on LinuxONE (s390x). Fine for development; numbers must come from the VM.")
    print(f"python          : {info['python']}")
    print(f"cores / RAM     : {info['cores']} / {info['ram_gb'] and round(info['ram_gb'], 1)} GB")
    print(f"disk free       : {info['disk_free_gb']} GB")
    print(f"NNPA flag       : {info['nnpa_flag']}")
    print(f"zDNN libs       : {info['zdnn_libs'] or 'none found'}")
    print(f"ROOT            : {info['root']}")
    print(f"threads setting : {info['threads_setting']}   demo event: {info['demo_event']}")
    print("-" * 64)
    for p, v in info["packages"].items():
        print(f"  {p:<18} {v or 'NOT INSTALLED'}")
    print("-" * 64)
    for k, v in info["paths"].items():
        print(f"  {'OK ' if v['exists'] else '-- '} {k:<14} {v['path']}")
    print("=" * 64)
    return info


if __name__ == "__main__":
    run()
