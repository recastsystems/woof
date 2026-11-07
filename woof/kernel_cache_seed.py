"""Seed the writable kernel cache from read-only, prebuilt caches.

THE BREAKAGE THIS PREVENTS.  CuPy reads and writes ONE kernel cache
directory (``CUPY_CACHE_DIR``).  An image that ships a prebuilt cache in
that directory loses it the moment a host directory is mounted over it to
keep compiles between containers: the run then compiles every kernel from
source, measured at one to three minutes per fresh container on a
300 x 300 x 49 member, with the prebuilt images sitting unread underneath.

``WOOF_KERNEL_CACHE_SEED`` names one or more read-only cache directories
(``os.pathsep``-separated).  Before the first kernel loads, every entry a
seed holds that the active cache lacks is copied in -- CuPy's compiled
images (``*.cubin`` and the like) and the engine's direct-NVRTC images
(``gpuwm-direct-nvrtc/*.image``).  Entries are named by a digest of their
complete compile key and each one carries its own content digest, which the
reader checks before use, so an entry copied from a seed is exactly the
entry a compile on this card would have written, or it is ignored.  An
entry already present is never replaced.

Absent -- the default -- nothing is read and nothing is copied.
"""

from __future__ import annotations

import os
import shutil
import tempfile
import time
from pathlib import Path

ENV = "WOOF_KERNEL_CACHE_SEED"
_DIRECT = "gpuwm-direct-nvrtc"


def _entries(root: Path):
    """(relative path, absolute path) of every cache entry under ``root``."""
    try:
        with os.scandir(root) as scan:
            for entry in scan:
                if entry.is_file() and not entry.name.startswith("."):
                    yield entry.name, Path(entry.path)
    except OSError:
        return
    for backend_dir in ("gpuwm-hip","gpuwm-hiprtc"):
        for path in (root/backend_dir).glob("*"):
            if path.is_file() and path.suffix in (".hsaco",".cubin",".json"):
                yield f"{backend_dir}/{path.name}",path
    direct = root / _DIRECT
    try:
        with os.scandir(direct) as scan:
            for entry in scan:
                if entry.is_file() and entry.name.endswith(".image"):
                    yield f"{_DIRECT}/{entry.name}", Path(entry.path)
    except OSError:
        return


def seed(cache_dir: Path | None = None, seeds=None) -> dict:
    """Copy into ``cache_dir`` every seed entry it lacks; return a receipt."""
    if seeds is None:
        raw = os.environ.get(ENV, "").strip()
        seeds = [part for part in raw.split(os.pathsep) if part.strip()]
    if not seeds:
        return {"seeded": 0}
    if cache_dir is None:
        from woof.kernel_compile_notice import cupy_kernel_cache_dir
        cache_dir = cupy_kernel_cache_dir()
    cache_dir = Path(cache_dir)
    began = time.perf_counter()
    copied = present = failed = 0
    copied_bytes = 0
    for source_root in seeds:
        source_root = Path(source_root)
        if source_root.resolve() == cache_dir.resolve():
            continue
        for relative, source in _entries(source_root):
            target = cache_dir / relative
            if target.exists():
                present += 1
                continue
            try:
                target.parent.mkdir(parents=True, exist_ok=True)
                with tempfile.NamedTemporaryFile(dir=target.parent, delete=False,
                                                 prefix=".seed-") as handle:
                    temporary = Path(handle.name)
                    with open(source, "rb") as reader:
                        shutil.copyfileobj(reader, handle, 1 << 20)
                # Never replace an entry a concurrent compile published.
                try:
                    os.link(temporary, target)
                except FileExistsError:
                    present += 1
                    continue
                except OSError:
                    # A filesystem without hard links: publish by rename,
                    # unless an entry appeared meanwhile.
                    if target.exists():
                        present += 1
                        continue
                    os.replace(temporary, target)
                finally:
                    temporary.unlink(missing_ok=True)
                copied += 1
                copied_bytes += source.stat().st_size
            except OSError:
                failed += 1
    return {"cache_dir": str(cache_dir), "seeds": [str(s) for s in seeds],
            "seeded": copied, "seeded_bytes": copied_bytes,
            "already_present": present, "failed": failed,
            "seconds": round(time.perf_counter() - began, 3)}


__all__ = ["ENV", "seed"]
