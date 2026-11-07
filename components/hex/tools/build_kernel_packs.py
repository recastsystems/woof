#!/usr/bin/env python3
"""Rebuild every pinned kernel pack from one tree, one compiler environment per NVRTC.

usage: build_kernel_packs.py TREE --out DIR [--base DIR] [--work DIR] [--family NAME ...]
                             [--workers N] [--reserve-gib G] [--tag kernel-packs-vX.Y.Z]

For each family below it makes (once) a venv holding CuPy 14.2.0, the family's NVRTC and the CUDA
headers a fresh ``[gpu-cu12]`` / ``[gpu-cu13]`` install carries, plus the published engine; checks
that the loaded NVRTC library has the family's SHA-256; and runs ``tools/build_hex_aot.py`` from
TREE in it.  A pack in ``--base`` built by the same compiler is reused image by image (only kernels
whose source or options changed compile) and pruned to this tree's inventory.  Compiles run one
process per image under a memory-admitted scheduler on every core, resumable from ``WORK/jobs``.
Families run concurrently.  Finally the packs are pinned into
``TREE/src/hexcore/data/on-demand-assets.json`` under ``--tag`` and ``SHA256SUMS`` is written in OUT.

THE BREAKAGE THIS ENDS (woof hex 0.4.12, 2026-11-05): NVRTC 13.5.37 reached PyPI after the gate,
no pack was built for it, and every fresh ``[gpu-cu13]`` install compiled every kernel for about
100 s before its first step; the five packs were rebuilt by five hand-made environments on rented
boxes, so a kernel change late in a release could not be followed by a pack rebuild in one step.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import tarfile
import time
import urllib.request
from pathlib import Path

ENGINE = "woof==2.9.12"
CUPY = "14.2.0"
BASE_REQUIREMENTS = ("numpy==2.5.4", "scipy==1.18.1", "netcdf4", "packaging", "psutil", "jsonschema",
                     "threadpoolctl")
APT = "https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/"

#: name -> pack file, requirements, the loaded NVRTC's SHA-256, optional system-toolkit deb.
FAMILIES = {
    "12.9.86": {"pack": "kernel-pack-cuda12.json",
                "requirements": (f"cupy-cuda12x[ctk]=={CUPY}", "nvidia-cuda-nvrtc-cu12==12.9.86"),
                "nvrtc_sha256": "7c67c6b51ea0e0279634cebd676ff7efda1674806444520c84430ad5c35fe625"},
    # pip 13.0.88 and the CUDA 13.0 system toolkit of a driver-580 machine carry the same bytes.
    "13.0.88": {"pack": "kernel-pack-cuda13.0.json",
                "requirements": (f"cupy-cuda13x[ctk]=={CUPY}", "nvidia-cuda-nvrtc==13.0.88"),
                "nvrtc_sha256": "a49e67e8e74590f1e98de55c39c6287efd3f59e3c3797464d7bbe0fe01349b11"},
    # A CUDA 13.2 system toolkit (NVIDIA apt cuda-nvrtc-13-2 13.2.51) shadowing no pip NVRTC; pip headers.
    "13.2.51-system": {"pack": "kernel-pack-cuda13.2.json",
                       "requirements": (f"cupy-cuda13x=={CUPY}", "cuda-pathfinder", "nvidia-cuda-runtime==13.2.51"),
                       "nvrtc_sha256": "b93dae6f3435a5b7b52a689af7f769cc7876fcfd7b01381ca7d6a3814d405c89",
                       "deb": "cuda-nvrtc-13-2_13.2.51-1_amd64.deb",
                       "lib": "usr/local/cuda-13.2/targets/x86_64-linux/lib"},
    # 13.4.59: a lane venv on the rented boxes held it; its library bytes differ from 13.4.92, so it matched no
    # pack and a strict hrrrv5 first step compiled for 114 s on one core (2026-11-06).
    "13.4.59": {"pack": "kernel-pack-cuda13.4.59.json",
                "requirements": (f"cupy-cuda13x[ctk]=={CUPY}", "nvidia-cuda-nvrtc==13.4.59"),
                "nvrtc_sha256": "5fc18c75a772c5fe0684d2a1bf3fa6ea964b6a71d0caaa828e5e1785dc16daba"},
    "13.4.92": {"pack": "kernel-pack-cuda13.4.json",
                "requirements": (f"cupy-cuda13x[ctk]=={CUPY}", "nvidia-cuda-nvrtc==13.4.92"),
                "nvrtc_sha256": "d8f82e7019fe990bcd9efe4c5c444c036ddf2d5df95090d3965ae62be2009a2c"},
    "13.5.37": {"pack": "kernel-pack-cuda13.json",
                "requirements": (f"cupy-cuda13x[ctk]=={CUPY}", "nvidia-cuda-nvrtc==13.5.37"),
                "nvrtc_sha256": "98a317ff5d2176dee5a04235c6663f8d553d5ed95b3104cbf7fb198cbf76a10d"},
}

PROBE = ("import json\nfrom woof.hex.cuda_backend.aot import compiler_identity\n"
         "print(json.dumps(compiler_identity(), sort_keys=True))\n")


def sha256(path):
    with open(path, "rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _deb_data(blob):
    """The data member of a .deb (an ar archive) as an open tarfile."""
    if blob[:8] != b"!<arch>\n":
        raise ValueError("not a .deb archive")
    offset = 8
    while offset < len(blob):
        name = blob[offset:offset + 16].decode().strip().rstrip("/")
        size = int(blob[offset + 48:offset + 58].decode().strip())
        body = blob[offset + 60:offset + 60 + size]
        if name.startswith("data.tar"):
            return tarfile.open(fileobj=io.BytesIO(body))
        offset += 60 + size + (size & 1)
    raise ValueError("the .deb has no data member")


def system_library(work, family):
    """Extract a system toolkit's NVRTC from NVIDIA's apt repository; return its library directory."""
    root = work / "system" / family["deb"].removesuffix(".deb")
    lib = root / family["lib"]
    if not lib.is_dir():
        with urllib.request.urlopen(APT + family["deb"], timeout=120) as response:
            blob = response.read()
        partial = root.with_suffix(".partial")
        shutil.rmtree(partial, ignore_errors=True)
        with _deb_data(blob) as archive:
            archive.extractall(partial, filter="tar")
        partial.rename(root)
    return lib


def make_env(work, name, family):
    env = work / "envs" / name
    python = env / "bin" / "python"
    stamp = env / ".requirements.json"
    wanted = [ENGINE, *family["requirements"], *BASE_REQUIREMENTS]
    if python.is_file() and stamp.is_file() and json.loads(stamp.read_text()) == wanted:
        return python
    uv = shutil.which("uv") or str(Path.home() / ".local/bin/uv")
    log = work / "logs" / f"env-{name}.log"
    with log.open("w") as out:
        if Path(uv).is_file():
            if not python.is_file():
                subprocess.run([uv, "venv", "-q", "--python", sys.executable, str(env)], check=True, stdout=out, stderr=out)
            install = [uv, "pip", "install", "-q", "--python", str(python)]
        else:
            if not python.is_file():
                subprocess.run([sys.executable, "-m", "venv", str(env)], check=True, stdout=out, stderr=out)
            install = [str(python), "-m", "pip", "install", "-q"]
        # The engine without its dependencies: the build reads its compile probes, not its CUDA stack.
        subprocess.run([*install, "--no-deps", ENGINE], check=True, stdout=out, stderr=out)
        subprocess.run([*install, *family["requirements"], *BASE_REQUIREMENTS], check=True, stdout=out, stderr=out)
    stamp.write_text(json.dumps(wanted))
    return python


def run_family(tree, out, base, work, name, workers, reserve_gib, helper=None):
    family = FAMILIES[name]
    started = time.time()
    python = make_env(work, name, family)
    environment = {key: value for key, value in os.environ.items()
                   if not key.startswith(("GPUWM", "HEX_", "CUPY_", "CUDA_", "LD_LIBRARY_PATH", "PYTHON"))}
    environment.update(PYTHONPATH=str(tree / "src"), CUDA_VISIBLE_DEVICES="-1", WOOF_HEX_NO_LOCAL_GPU="1",
                       OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
                       CUPY_CACHE_DIR=str(work / "cupy-cache" / name),
                       WOOF_HEX_AOT_LEDGER=str(work / "ledger"))
    if helper is not None:
        # A second machine compiling into a synchronised WORK/jobs: smallest first, nothing over its memory.
        environment.update(WOOF_HEX_AOT_ORDER="small-first", WOOF_HEX_AOT_MAX_JOB_GIB=str(helper))
    if "deb" in family:
        environment["LD_LIBRARY_PATH"] = str(system_library(work, family))
    identity = json.loads(subprocess.run([str(python), "-c", PROBE], env=environment, check=True,
                                         capture_output=True, text=True).stdout.splitlines()[-1])
    if identity["nvrtc_library_sha256"] != family["nvrtc_sha256"]:
        raise SystemExit(f"{name}: loaded NVRTC sha256 {identity['nvrtc_library_sha256']}, "
                         f"expected {family['nvrtc_sha256']}")
    output = out / family["pack"]
    command = [str(python), str(tree / "tools" / "build_hex_aot.py"), "--output", str(output),
               "--job-dir", str(work / "jobs" / name), "--workers", str(workers),
               "--reserve-gib", str(reserve_gib)]
    reuse = None
    candidates = [base / family["pack"], *sorted(base.glob("kernel-pack-cuda*.json"))] if base else []
    for candidate in candidates:
        if candidate.is_file():
            built = subprocess.run([str(python), "-c",
                                    "import json,sys\nfrom woof.hex.kernel_pack import compiler_of\n"
                                    "print(json.dumps(compiler_of(sys.argv[1])))", str(candidate)],
                                   env=environment, capture_output=True, text=True).stdout.strip()
            from_pack = json.loads(built or "null") or {}
            if from_pack.get("nvrtc_library_sha256") == family["nvrtc_sha256"] and from_pack.get("cupy") == identity["cupy"]:
                reuse = candidate
                break
    if reuse is not None:
        command += ["--base-pack", str(reuse), "--prune"]
    log = work / "logs" / f"build-{name}.log"
    with log.open("w") as stream:
        rc = subprocess.run(command, env=environment, stdout=subprocess.PIPE, stderr=stream, text=True)
    if rc.returncode:
        raise SystemExit(f"{name}: build_hex_aot.py rc {rc.returncode}; see {log}")
    summary = json.loads(rc.stdout.strip().splitlines()[-1])
    return {"family": name, "pack": family["pack"], "nvrtc": identity["nvrtc"],
            "nvrtc_package": identity["nvrtc_package"], "base_pack": str(reuse) if reuse else None,
            "bytes": output.stat().st_size, "sha256": sha256(output), "seconds": round(time.time() - started, 1),
            **summary}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("tree", type=Path, help="woof hex tree (the directory holding src/ and tools/)")
    parser.add_argument("--out", type=Path, required=True, help="directory for the pack files and SHA256SUMS")
    parser.add_argument("--base", type=Path, help="earlier packs to reuse unchanged images from")
    parser.add_argument("--work", type=Path, help="environments, compile jobs and logs (default OUT/../work)")
    parser.add_argument("--family", action="append", choices=sorted(FAMILIES), help="build only these")
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--reserve-gib", type=float, default=8.0)
    parser.add_argument("--helper-max-job-gib", type=float, metavar="GIB",
                        help="run as a helper machine: compile smallest images first and none expected over GIB, "
                             "into a WORK/jobs that is synchronised with the main builder's (rsync *.json both ways)")
    parser.add_argument("--tag", help="pin the packs into the tree's on-demand-assets.json under this release tag")
    parser.add_argument("--repository", default="FahrenheitResearch/gpuwm-hex")
    args = parser.parse_args()
    tree = args.tree.resolve()
    out = args.out.resolve()
    work = (args.work or out.parent / "work").resolve()
    for directory in (out, work / "logs", work / "envs", work / "jobs", work / "cupy-cache"):
        directory.mkdir(parents=True, exist_ok=True)
    names = args.family or list(FAMILIES)
    started = time.time()
    # Each family's scheduler admits compiles by the machine's live free memory, so families share it.
    with concurrent.futures.ThreadPoolExecutor(len(names)) as pool:
        futures = {name: pool.submit(run_family, tree, out, args.base and args.base.resolve(), work, name,
                                     args.workers, args.reserve_gib, args.helper_max_job_gib) for name in names}
        results = [futures[name].result() for name in names]
    packs = sorted(path.name for path in out.glob("kernel-pack-cuda*.json"))
    (out / "SHA256SUMS").write_text("".join(f"{sha256(out / name)}  {name}\n" for name in packs))
    if args.tag:
        environment = dict(os.environ, PYTHONPATH=str(tree / "src"))
        python = work / "envs" / names[-1] / "bin" / "python"
        subprocess.run([str(python), "-m", "woof.hex.assets", "pin", "--from", str(out), "--tag", args.tag,
                        "--repository", args.repository,
                        "--output", str(tree / "src/hexcore/data/on-demand-assets.json")],
                       env=environment, check=True)
    receipt = {"tree": str(tree), "tree_head": subprocess.run(["git", "-C", str(tree), "rev-parse", "HEAD"],
                                                              capture_output=True, text=True).stdout.strip(),
               "tag": args.tag, "wall_seconds": round(time.time() - started, 1), "families": results}
    (out / "build-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
