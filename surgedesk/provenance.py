"""Local, content-addressed provenance without copying private input text."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import threading
import time
import contextlib
import io
from pathlib import Path

from . import config, envcheck
from .textutil import mask_privacy


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def file_hash(path) -> str:
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def tree_hash(path) -> dict:
    root = Path(path)
    if root.is_file():
        return {root.name: file_hash(root)}
    return {str(p.relative_to(root)).replace("\\", "/"): file_hash(p)
            for p in sorted(root.rglob("*")) if p.is_file() and "__pycache__" not in p.parts} if root.exists() else {}


def git(*args):
    try:
        return subprocess.check_output(["git", *args], cwd=config.ROOT,
                                       stderr=subprocess.DEVNULL, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def capture() -> dict:
    info = envcheck.collect()
    try:
        import psutil
        if info["ram_gb"] is None:
            info["ram_gb"] = psutil.virtual_memory().total / (1024 ** 3)
        info["physical_cores"] = psutil.cpu_count(logical=False)
    except ImportError:
        info["physical_cores"] = None
    for package in ("psutil", "joblib"):
        try:
            info["packages"][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            info["packages"][package] = None
    runtime = None
    numpy = sys.modules.get("numpy")
    if numpy is not None and hasattr(numpy, "show_runtime"):
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream):
            numpy.show_runtime()
        runtime = stream.getvalue()
    torch = sys.modules.get("torch")
    return {"commit": git("rev-parse", "HEAD"), "branch": git("branch", "--show-current"),
            "working_tree": git("status", "--porcelain"),
            "source_files": tree_hash(config.PACKAGE_DIR), "environment": info,
            "platform": platform.platform(), "python_executable": sys.executable,
            "processor": platform.processor(), "machine_token": digest(platform.node()),
            "installed_distributions": {d.metadata["Name"]: d.version for d in importlib.metadata.distributions() if d.metadata["Name"]},
            "thread_environment": {key: os.environ.get(key) for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")},
            "numpy_runtime": runtime, "torch_intraop_threads": torch.get_num_threads() if torch else None,
            "seed": config.SEED, "threads": config.N_THREADS,
            "model_identifier": config.RESEARCH_MODEL_ID}


def report_identity(reports):
    return digest([[r.report_id, mask_privacy(r.text)] for r in reports])


class Measurement:
    """Sample this process' RSS; missing psutil stays unavailable, not zero.

    Sampling is an observed peak, not an operating-system high-water mark.
    Child process memory is deliberately not included.
    """
    def __enter__(self):
        self.start = time.perf_counter()
        self.peak = None
        self.stop = threading.Event()
        try:
            import psutil
            self.process = psutil.Process()
        except ImportError:
            self.process = None
        self._sample()
        self.thread = threading.Thread(target=self._poll, daemon=True)
        self.thread.start()
        return self

    def _sample(self):
        if self.process is not None:
            value = self.process.memory_info().rss
            self.peak = max(self.peak or 0, value)

    def _poll(self):
        while not self.stop.wait(config.RESEARCH_MEMORY_POLL_S):
            self._sample()

    def __exit__(self, *_):
        self.stop.set()
        self.thread.join()
        self._sample()
        self.seconds = time.perf_counter() - self.start

    def as_dict(self):
        return {"seconds": self.seconds, "peak_rss_bytes": self.peak,
                "memory_method": "sampled process RSS, excludes children" if self.process else None,
                "memory_unavailable_reason": None if self.process else "psutil is not installed"}
