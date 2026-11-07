"""Build large unchanged production units ahead of a forecast.

``--family gfdleta`` (default) holds the GFDL radiation units; ``--family mskf``
holds the MSKF cumulus column (cu_physics 11), whose 255-register unit
otherwise compiles on one core inside the first forecast step.

Run in each supported pinned CuPy/NVRTC environment. No device or forecast
is needed. All four shipped architectures and fast/WRF-exact recipes are
compiled in independent processes; publication uses the common pack codec.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import multiprocessing
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
UNITS = ("gfdleta_column", "gfdleta_co2_init", "gfdleta_table_runtime",
         "gfdleta_tables", "gfdleta_init")
#: family -> (loader, ((unit, integer define tier), ...)). Each request is
#: produced by the forecast's own loader, so the pack key is the exact
#: source/options/filename the forecast hands NVRTC.
FAMILIES = {
    "gfdleta": ("module", tuple((name, ()) for name in UNITS)),
    # woof/core/mskf.py: the column kernel with and without aerosol
    # (GPUWM_MSKF_NO_AEROSOL when aercu_opt=0) and mskf_w0avg, one module.
    "mskf": ("no_ftz", (("mskf", ()), ("mskf", (("GPUWM_MSKF_NO_AEROSOL", 1),)))),
}


#: The numerics a forecast or proof run hands NVRTC. Fast is the default
#: forecast. Strict is the opt-in proof path, once per referee glibc
#: generation (woof.wrf_exact.referee_glibc): 2.39 is the published strict
#: record, and GPUWM_WRF_GLIBC=2.43 (any 2.41+) names the pinned native
#: referees the ra99 paper packets were captured on. Their GFDL column and
#: CO2 sources differ, so a pack holding only the 2.39 recipe left the 2.43
#: replay compiling gfdleta_column on one core (SKEPTIC-3 D: 1981 s).
#: Identical rows (units the glibc generation does not touch) are held once.
CATALOGS = (
    ("fast", {"GPUWM_WRF_EXACT": "0", "GPUWM_NUMERICS": "fast"}),
    ("strict", {"GPUWM_WRF_EXACT": "1", "GPUWM_NUMERICS": "strict", "GPUWM_WRF_GLIBC": "2.39"}),
    ("strict-glibc2.43", {"GPUWM_WRF_EXACT": "1", "GPUWM_NUMERICS": "strict", "GPUWM_WRF_GLIBC": "2.43"}),
)


def compile_one(job):
    from woof.kernel_aot import compiler_identity
    from woof.kernel_aot_build import compile_entry
    from woof.kernel_pack import write
    row, arch, checkpoint = job
    image = compile_entry(row, arch)
    write(checkpoint, compiler_identity(), [image])
    return image


def build(output, catalogs, work, workers):
    """Checkpoint each verified image before reporting it to the parent pool."""
    from woof.kernel_aot import ARCHITECTURES, Pack, compiler_identity, recipe
    from woof.kernel_pack import digest, read, write
    identity = compiler_identity()
    recipes = {}
    for path in catalogs:
        for entry in read(path, identity):
            row = recipe(entry)
            recipes[digest(row)] = row
    held, jobs = [], []
    for key, row in recipes.items():
        for arch in ARCHITECTURES:
            checkpoint = work / "images" / (key + "-" + arch + ".json.xz")
            if checkpoint.exists():
                pack = Pack(checkpoint, identity=identity)
                image, reason = pack.select(row['module_key'], row['source'],
                                           row['options'], arch,
                                           metadata=row['compile_request'])
                if reason or pack.image(image) is None:
                    raise ValueError(f"invalid completed image: {checkpoint}: {reason}")
                held.append(image)
            else:
                jobs.append((row, arch, checkpoint))
    started, reused = time.perf_counter(), len(held)
    if jobs:
        with concurrent.futures.ProcessPoolExecutor(
                min(workers, len(jobs)), mp_context=multiprocessing.get_context("spawn")) as pool:
            for future in concurrent.futures.as_completed([pool.submit(compile_one, job) for job in jobs]):
                image = future.result()
                held.append(image)
                print(json.dumps(dict(completed=len(held), total=len(recipes)*len(ARCHITECTURES),
                                      unit=image['compile_request']['filename'], arch=image['arch'],
                                      compile_seconds=image['compile_seconds'])), flush=True)
    held.sort(key=lambda image: (image['module_key'], image['arch'], image['options']))
    write(output, identity, held)
    return dict(modules=len(recipes), entries=len(held), compiled=len(jobs), reused=reused,
                architectures=list(ARCHITECTURES), workers=workers,
                wall_seconds=time.perf_counter()-started)


class _Request(BaseException):
    """Raised by the capturing compiler: no NVRTC or device work is done."""


def _no_ftz_request(name, defines):
    """The exact request woof.core.kernels.no_ftz hands the direct compiler."""
    import woof.nvrtc_ptx_cache as direct
    from woof.core.kernels.no_ftz import load_module_no_ftz
    captured = []
    def capture(source, options=(), arch=None, filename="kern.cu", *args, **kwargs):
        captured.append((source, tuple(options), filename))
        raise _Request()
    original = direct.compile_using_nvrtc
    direct.compile_using_nvrtc = capture
    try:
        load_module_no_ftz.__wrapped__(name, tuple(defines))
    except _Request:
        pass
    finally:
        direct.compile_using_nvrtc = original
    if len(captured) != 1:
        raise ValueError(f"{name}: expected one direct compiler request, saw {len(captured)}")
    return captured[0]


def catalog(output, family="gfdleta"):
    from woof.core.kernels import module_options, module_source
    from woof.kernel_aot import compiler_identity, effective_options, DIRECT_ROUTE
    from woof.kernel_pack import write
    loader, units = FAMILIES[family]
    rows = []
    for name, defines in units:
        if loader == "module":
            source, options, filename = module_source(name), module_options(name), name + ".cu"
        else:
            source, options, filename = _no_ftz_request(name, defines)
        rows.append(dict(route=DIRECT_ROUTE, source=source,
                         options=list(effective_options(options)), filename=filename))
    write(output, compiler_identity(), rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--catalog", type=Path)
    parser.add_argument("--family", choices=sorted(FAMILIES), default="gfdleta")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error('--workers must be positive')
    if args.catalog:
        catalog(args.catalog, args.family)
        return
    args.work.mkdir(parents=True, exist_ok=True)
    catalogs = []
    for label, settings in CATALOGS:
        path = args.work / (label + ".json.xz")
        env = {key: value for key, value in os.environ.items()
               if key not in ("GPUWM_WRF_EXACT", "GPUWM_NUMERICS", "GPUWM_WRF_GLIBC")}
        env.update(settings)
        subprocess.run([sys.executable, __file__, "--work", str(args.work),
                        "--catalog", str(path), "--family", args.family], env=env, check=True)
        catalogs.append(path)
    from woof.kernel_aot import compiler_identity, unit_pack_path
    output = unit_pack_path(args.family, compiler_identity())
    receipt = build(output, catalogs, args.work, args.workers)
    receipt.update(output=str(output), compiler=compiler_identity(), family=args.family,
                   units=[[name, [list(d) for d in defines]] for name, defines in FAMILIES[args.family][1]],
                   command=sys.argv)
    (args.work / "build.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    main()
