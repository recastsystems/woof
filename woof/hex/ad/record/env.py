# SPDX-License-Identifier: Apache-2.0
"""Where a recording finds what it needs, from the installed environment only.

Nothing here names a host, a user or a fixed directory. The CUDA toolkit, CUPTI, cuBLAS and CuPy's CUB/Thrust host
modules come from the wheels installed beside hexcore (cupy-cuda13x[ctk] and nvidia-cuda-cupti); a site that keeps
them elsewhere names the toolkit root with HEXAD_CUDA_ROOT. Built recorder libraries and warm caches live under
HEXAD_RECORD_CACHE, else $XDG_CACHE_HOME/gpuwm-hex/ad-record, else ~/.cache/gpuwm-hex/ad-record.
"""
from __future__ import annotations

import hashlib
import importlib.util
import os
import shutil
import subprocess
import sys
import sysconfig
from pathlib import Path

HERE = Path(__file__).resolve().parent
TOOLS = HERE / "tools"
NATIVE = HERE / "native"
DATA = HERE / "data"


def defect(message: str) -> "NoReturn":  # noqa: F821
    """One line, exit 3: the record/seal/replay chain never falls back to an older package or an unchecked path."""
    print(f"DEFECT: woof.hex.ad.record: {message}", file=sys.stderr, flush=True)
    raise SystemExit(3)


def purelib() -> Path:
    return Path(sysconfig.get_paths()["purelib"])


def cuda_root() -> Path:
    named = os.environ.get("HEXAD_CUDA_ROOT")
    if named:
        root = Path(named)
        if not (root / "lib").is_dir():
            defect(f"HEXAD_CUDA_ROOT={named} has no lib/ directory")
        return root
    for base in [purelib(), *[Path(p) for p in sys.path if p.endswith("site-packages")]]:
        for name in ("cu13", "cu12"):
            root = base / "nvidia" / name
            if (root / "lib").is_dir():
                return root
    defect("no CUDA toolkit wheel beside hexcore (install recast-woof[gpu-cu13]) and HEXAD_CUDA_ROOT is not set")


def cupti() -> tuple[Path, Path]:
    """(libcupti, include dir). CUPTI comes from the nvidia-cuda-cupti wheel of the same CUDA major."""
    root = cuda_root()
    libs = sorted(p for p in (root / "lib").glob("libcupti.so*") if p.is_file())
    include = root / "include"
    if not libs or not (include / "cupti.h").is_file():
        defect(f"CUPTI is not installed in {root} (pip install 'nvidia-cuda-cupti>=13,<14')")
    return libs[0], include


def cublas() -> Path:
    libs = sorted(p for p in (cuda_root() / "lib").glob("libcublas.so*") if p.is_file())
    if not libs:
        defect(f"libcublas is not installed in {cuda_root()}")
    return libs[0]


def cupy_host_libs() -> tuple[Path, Path]:
    """CuPy's CUB and Thrust host extension modules (the recorder observes calls into them)."""
    spec = importlib.util.find_spec("cupy")
    if spec is None or not spec.submodule_search_locations:
        defect("cupy is not installed (install recast-woof[gpu-cu13])")
    cuda = Path(list(spec.submodule_search_locations)[0]) / "cuda"
    found = []
    for name in ("cub", "thrust"):
        hits = sorted(cuda.glob(f"{name}.cpython-*.so"))
        if not hits:
            defect(f"cupy/cuda/{name}.cpython-*.so is missing in {cuda}")
        found.append(hits[0])
    return found[0], found[1]


def hexcore_src() -> Path:
    spec = importlib.util.find_spec("woof.hex")
    if spec is None or not spec.submodule_search_locations:
        defect("hexcore is not importable")
    return Path(list(spec.submodule_search_locations)[0])


def cache_root() -> Path:
    named = os.environ.get("HEXAD_RECORD_CACHE")
    if named:
        return Path(named)
    xdg = os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    return Path(xdg) / "woof hex" / "ad-record"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            digest.update(chunk)
    return digest.hexdigest()


def compute_capability(card: str) -> str:
    """'sm_120' for the named card (nvidia-smi index or UUID)."""
    try:
        out = subprocess.run(["nvidia-smi", "-i", str(card), "--query-gpu=compute_cap", "--format=csv,noheader"],
                             check=True, capture_output=True, text=True, timeout=60).stdout.strip().splitlines()[0]
    except (OSError, subprocess.SubprocessError, IndexError) as error:
        defect(f"cannot read the compute capability of card {card}: {error}")
    major, minor = out.strip().split(".")
    return f"sm_{major}{minor}"


def physical_card(card) -> str:
    """The card a chain step runs on, as CUDA_VISIBLE_DEVICES names it for the child: a card this process may see is
    kept; an index within a set CUDA_VISIBLE_DEVICES (the scheduler's cards) is counted inside it.
    THE BREAKAGE THIS PREVENTS: under a queue that gave the job card 3 (CUDA_VISIBLE_DEVICES=3), `record --card 0`
    (the default) set CUDA_VISIBLE_DEVICES=0 for every forward, so the recordings ran on physical card 0, another
    job's card, while the queue counted them on card 3 (18 Nov 2026: three 3-36 km recordings on card 0)."""
    card = str(card)
    seen = [c.strip() for c in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if c.strip()]
    if not seen or seen[0].startswith("-") or card in seen:
        return card
    if card.isdigit() and int(card) < len(seen):
        return seen[int(card)]
    return card


def runtime_env(card: str) -> dict:
    """The process environment every CUDA step of the chain runs in: the named card, the wheel toolkit, no MPS."""
    root = cuda_root()
    libcupti, _ = cupti()
    env = dict(os.environ)
    env.pop("CUDA_HOME", None)
    env["CUDA_VISIBLE_DEVICES"] = physical_card(card)
    env["CUDA_PATH"] = str(root)
    paths = [str(libcupti.parent), str(root / "lib")]
    if env.get("LD_LIBRARY_PATH"):
        paths.append(env["LD_LIBRARY_PATH"])
    env["LD_LIBRARY_PATH"] = ":".join(dict.fromkeys(paths))
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return env


def which_runner(named: str | None) -> Path | None:
    """The replay runner: --runner, else HEXAD_RUNNER, else the wheel's own runner (woof.hex.hexad.binary()), else
    `ad-binder` on PATH."""
    for candidate in (named, os.environ.get("HEXAD_RUNNER")):
        if candidate:
            path = Path(candidate)
            if not (path.is_file() and os.access(path, os.X_OK)):
                defect(f"runner {candidate} is not an executable file")
            return path
    # The wheel's own runner (woof/hex/libexec/ad-binder, located by woof.hex.hexad), the one the doors run.
    # THE BREAKAGE THIS PREVENTS (hex 0.6.0 candidate, fresh [ad] install on an RTX 4090, 2026-11-16): `python -m
    # woof.hex.ad.record record ...` stopped on "no replay runner" although the wheel carries ad-binder, because this
    # looked for a module (woof.hex.ad.runner) that no wheel has; a user recording a plan on their own card, the only
    # way to a plan for a card family the shipped plans were not recorded on, had to find libexec by hand.
    try:
        from woof.hex import hexad
        path = Path(hexad.binary())
        if path.is_file() and os.access(path, os.X_OK):
            return path
    except Exception:  # noqa: BLE001 - the packaged runner is optional (sdist without a build); PATH is the next place
        pass
    hit = shutil.which("ad-binder")
    return Path(hit) if hit else None
