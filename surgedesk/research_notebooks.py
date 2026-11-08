"""Execute research notebooks with a local kernel and workspace-only caches."""
import os
from pathlib import Path
import subprocess
import sys

from . import config, fileio
from .provenance import capture, file_hash


def main():
    root = config.RESEARCH_NOTEBOOK_DIR.resolve()
    environment = dict(os.environ)
    for variable, name in (("IPYTHONDIR", "ipython"), ("JUPYTER_RUNTIME_DIR", "runtime"),
                           ("JUPYTER_CONFIG_DIR", "config"), ("JUPYTER_DATA_DIR", "data")):
        directory = root / name
        directory.mkdir(parents=True, exist_ok=True)
        environment[variable] = str(directory)
    kernel = Path(environment["JUPYTER_DATA_DIR"]) / "kernels" / "python3" / "kernel.json"
    fileio.write_json(kernel, {"argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
                               "display_name": "SurgeDesk local validation", "language": "python"})
    environment["PYTHONPATH"] = str(config.ROOT) + os.pathsep + environment.get("PYTHONPATH", "")
    status = subprocess.run([sys.executable, "-m", "nbconvert", "--to", "notebook", "--execute",
                             *map(str, config.RESEARCH_NOTEBOOKS), "--output-dir", str(root),
                             "--ExecutePreprocessor.timeout=120"], cwd=config.ROOT, env=environment).returncode
    fileio.write_json(root / config.RESEARCH_NOTEBOOK_CHECKS_FILE, {"status": "passed" if status == 0 else "failed",
                       "notebooks": {str(p.relative_to(config.ROOT)): file_hash(p) for p in config.RESEARCH_NOTEBOOKS},
                       "provenance": capture()})
    return status


if __name__ == "__main__":
    raise SystemExit(main())
