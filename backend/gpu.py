"""CUDA runtime helpers (pip-installed NVIDIA wheels on Windows)."""
from __future__ import annotations

import os
import site
import sys
from pathlib import Path

_done = False


def _site_dirs() -> list[Path]:
    dirs = [Path(p) for p in site.getsitepackages()]
    try:
        dirs.append(Path(site.getusersitepackages()))
    except Exception:
        pass
    dirs += [Path(p) for p in sys.path if p.endswith("site-packages")]
    return dirs


def register_cuda_dlls() -> None:
    """Make DLLs from site-packages/nvidia/*/bin (cublas, cudnn) loadable. No-op off Windows / if absent."""
    global _done
    if _done or sys.platform != "win32":
        return
    _done = True
    seen: set[Path] = set()
    for sp in _site_dirs():
        for bin_dir in sorted((sp / "nvidia").glob("*/bin")):
            if bin_dir in seen or not bin_dir.is_dir():
                continue
            seen.add(bin_dir)
            os.add_dll_directory(str(bin_dir))
            os.environ["PATH"] = str(bin_dir) + os.pathsep + os.environ.get("PATH", "")
