"""The hail-aware Thompson table set: build, verify and stage it.

Hail-aware Thompson (``mp_physics = 38``) reads WRF v4.6.1's nine-slab
rain-graupel table, ``qr_acr_qg_mp38V1.dat`` -- one slab per graupel bulk
density rho_g(1:9) = 50..800 kg m-3 (module_mp_thompson.F:4117-4125,
:4192-4273) -- beside the three v4.6.1 caches it shares with mp=8/28.  WRF
does not ship it: ``thompson_init`` computes it on first use when
``write_thompson_mp38table`` is set, and so does WOOF, through the same
Thompson table path the WRF 3.9 fork set takes
(:mod:`woof.thompson_fork_build`): hash-pinned public WRF source, a small
pinned Fortran harness, unchanged ``thompson_init`` arithmetic, and the result held to its size and SHA-256 pin
(:data:`woof.core.thompson_contract.HAIL_TABLE_ASSETS`) before anything
reads it.  Bytes that differ from the pin are refused, never installed.

THE ONE DIFFERENCE FROM THE FORK BUILD IS PARALLELISM, AND IT CANNOT MOVE A
BIT.  The table integrates 9 x 37 x 37 x 37 x 37 size-distribution pairs
over 100 x 100 size bins, about three quarters of an hour on one CPU.  WRF
itself distributes exactly that loop over MPI ranks (DM_PARALLEL: each rank
computes a ``wrf_dm_decomp1d`` share of the rain index pairs and
``wrf_dm_gatherv`` assembles them, :4201-4285); every element depends only
on its own indices.  The harness (``stub_wrf_hail_dm.F90``) gives each of
``N`` processes its share through the environment instead of MPI, and
:func:`splice_table_shares` assembles the shares byte for byte.  The
spliced file is then verified against the same pin a single-process
generation of WRF's code produces (tools/thompson_wrf461_oracle/
build_hail.sh generates that one in one process).

Where the set lives: :func:`woof.physics_compat.thompson_hail_table_root`
(``WOOF_THOMPSON_HAIL_TABLE_ROOT``, else ``~/.woof/tables/
thompson-wrf461-hail``).  The three shared caches are linked (or copied
where the filesystem refuses a link) from the verified v4.6.1 root, so the
hail root is one complete, verifiable directory.  A directory holding an
already built ``qr_acr_qg_mp38V1.dat`` can be named with
``WOOF_THOMPSON_HAIL_TABLE_SOURCE_ROOT``; its bytes are held to the same
pin.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import struct
import subprocess
import tempfile
from functools import lru_cache

from woof import fetch_guard
from woof.core.thompson_contract import (
    HAIL_GENERATED_TABLE_FILES, HAIL_MP38_TABLE_FILE, HAIL_TABLE_ASSETS,
    TableAsset, validate_table_assets,
)
from woof.table_assets import (
    TableAssetError, classify_assets, fetch_asset_from_url,
)

HAIL_TABLE_SOURCE_ROOT_ENV = "WOOF_THOMPSON_HAIL_TABLE_SOURCE_ROOT"
HAIL_TABLE_WORKERS_ENV = "WOOF_THOMPSON_HAIL_TABLE_WORKERS"

#: WRF v4.6.1 at the commit every v4.6.1 pin in the tree names.
SOURCE_BASE = ("https://raw.githubusercontent.com/wrf-model/WRF/"
               "d66e442fccc04111067e29274c9f9eaccc3cef28/phys")
PUBLIC_SOURCES = (
    TableAsset("module_mp_thompson.F", 240308,
               "fabf19e2a9073cff886e882b187080bfdf089d3fd40c0fce1d19bc93b1e5e802"),
    TableAsset("module_mp_radar.F", 23668,
               "aa99da858be41efa579966680708d230123a7417560af0eb2e24f4c94e253688"),
)
# The Fortran fixture generator uses a pinned scalar reference runtime.
# A host runtime can change log10(real32), and therefore freeze-table words.
REFERENCE_RUNTIME_ASSETS = (
    (TableAsset("libc6_2.35-0ubuntu3.15_amd64.deb", 3234380,
                "79e35256227e16a607c154cdeb8d76ff12d20e31de286ea7fd9ad3b96fe0452d"),
     "https://archive.ubuntu.com/ubuntu/pool/main/g/glibc/"),
    (TableAsset("libgfortran5_12.3.0-1ubuntu1~22.04.3_amd64.deb", 878662,
                "14dbb269458f60ca64af646c9300b8b85dd962ddbb434ba362062a54c07c4db6"),
     "https://archive.ubuntu.com/ubuntu/pool/main/g/gcc-12/"),
    (TableAsset("libquadmath0_12.3.0-1ubuntu1~22.04.3_amd64.deb", 153976,
                "c4c59e67e76674c8c494ce5e0415b71d770a759b48893c0d3cbabddceeb690b2"),
     "https://archive.ubuntu.com/ubuntu/pool/main/g/gcc-12/"),
)

REFERENCE_RUNTIME_SCRIPT = r"""#!/usr/bin/env bash
set -euo pipefail
work=$1
cd "$work"
mkdir -p runtime/pkg runtime/lib
for package in libc6_2.35-0ubuntu3.15_amd64.deb \
 libgfortran5_12.3.0-1ubuntu1~22.04.3_amd64.deb \
 libquadmath0_12.3.0-1ubuntu1~22.04.3_amd64.deb; do
 dpkg-deb -x "sources/$package" runtime/pkg
done
lib="$work/runtime/lib"
ln -s ../pkg/lib/x86_64-linux-gnu/libm.so.6 "$lib/libm.so.6"
ln -s libm.so.6 "$lib/libm.so"
ln -s ../pkg/usr/lib/x86_64-linux-gnu/libgfortran.so.5 "$lib/libgfortran.so.5"
ln -s libgfortran.so.5 "$lib/libgfortran.so"
ln -s ../pkg/usr/lib/x86_64-linux-gnu/libquadmath.so.0 "$lib/libquadmath.so.0"
ln -s libquadmath.so.0 "$lib/libquadmath.so"
"""

#: ntb_r*ntb_r1, the rain index pairs WRF's decomposition splits.
RAIN_PAIRS = 37 * 37
#: Bytes of one rain index pair's slab of one record: ntb_g1 x ntb_g x NRHG
#: float64 values (the record is Fortran-ordered (37, 37, 9, 37, 37), so a
#: pair's share is one contiguous run).
PAIR_BYTES = 37 * 37 * 9 * 8

BUILD_SCRIPT = r'''#!/usr/bin/env bash
set -euo pipefail
work=$1
workers=$2
cd "$work"
command -v gfortran >/dev/null || { echo 'GNU Fortran (gfortran) is required'; exit 2; }
test "$(uname -m)" = x86_64 || { echo 'The pinned scalar table build needs Linux x86_64'; exit 2; }
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
flags=(-O2 -fallow-argument-mismatch -fno-tree-vectorize -ffree-form -ffree-line-length-none)
lib="$work/runtime/lib"
linkflags=(-L"$lib" -Wl,--disable-new-dtags,-rpath="$lib")
gfortran -c "${flags[@]}" stub_wrf_hail_dm.F90
gfortran -c "${flags[@]}" sources/module_mp_radar.F
gfortran -c "${flags[@]}" -cpp -DWRF_CHEM=0 -DDM_PARALLEL sources/module_mp_thompson.F -o module_mp_thompson.o
gfortran -c "${flags[@]}" generate_tables_hail.F90
gfortran "${flags[@]}" "${linkflags[@]}" -o generate_tables_hail stub_wrf_hail_dm.o module_mp_radar.o module_mp_thompson.o generate_tables_hail.o
if nm -D generate_tables_hail | grep -q _ZGV; then echo 'Refused vector libm generation'; exit 2; fi
printf 'compiler: '; gfortran --version | head -1
mkdir -p common
if test -f classic/qr_acr_qsV2.dat; then
  ln -s ../classic/qr_acr_qsV2.dat common/qr_acr_qsV2.dat
  ln -s ../classic/freezeH2O.dat common/freezeH2O.dat
fi
( cd common && THOMPSON_TABLE_SLICE=0 THOMPSON_TABLE_SLICES=1369 \
  nice -n 15 timeout 7200s "$work/generate_tables_hail" > generate.log 2>&1 )
# The shared caches also have MPI decomposition. Rebuild those in a serial
# module, reading the unused partial collection cache only to avoid its loop.
rm common/qr_acr_qsV2.dat common/freezeH2O.dat
gfortran -c "${flags[@]}" -cpp -DWRF_CHEM=0 sources/module_mp_thompson.F -o module_mp_thompson_serial.o
gfortran "${flags[@]}" "${linkflags[@]}" -o generate_shared_hail stub_wrf_hail_dm.o module_mp_radar.o module_mp_thompson_serial.o generate_tables_hail.o
( cd common && nice -n 15 timeout 7200s "$work/generate_shared_hail" >> generate.log 2>&1 )
# A partial collection file is not a coefficient asset.
rm common/qr_acr_qg_mp38V1.dat

pids=()
for ((slice = 0; slice < workers; slice++)); do
  mkdir -p "share-$slice"
  # The two shared caches are READ by thompson_init, never recomputed.
  ln -s ../common/qr_acr_qsV2.dat "share-$slice/qr_acr_qsV2.dat"
  ln -s ../common/freezeH2O.dat "share-$slice/freezeH2O.dat"
  ( cd "share-$slice" && THOMPSON_TABLE_SLICE=$slice THOMPSON_TABLE_SLICES=$workers \
      nice -n 15 timeout 7200s "$work/generate_tables_hail" > generate.log 2>&1 ) &
  pids+=($!)
done
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
exit $status
'''


def _harness_dir() -> Path:
    return Path(__file__).parent / "data" / "thompson" / "hail-build"


def _harness_pins() -> dict[str, str]:
    from woof.core.thompson_contract import HAIL_HARNESS_SHA256
    return dict(HAIL_HARNESS_SHA256)


def worker_count() -> int:
    """How many share processes the build runs (env override, else CPUs)."""
    override = os.environ.get(HAIL_TABLE_WORKERS_ENV, "").strip()
    if override:
        count = int(override)
    else:
        count = os.cpu_count() or 1
    return max(1, min(count, 16, RAIN_PAIRS))


def share_bounds(slice_index: int, slices: int) -> tuple[int, int]:
    """The [km_s, km_e] share stub_wrf_hail_dm.F90 hands one process."""
    chunk = (RAIN_PAIRS + slices - 1) // slices
    start = slice_index * chunk
    return start, min(RAIN_PAIRS, (slice_index + 1) * chunk) - 1


def splice_table_shares(share_files: list[Path], output: Path) -> None:
    """Assemble the shares' ``qr_acr_qg_mp38V1.dat`` into one file.

    Each share file is a complete five-record WRF sequential file in which
    only that share's rain index pairs were computed.  The output takes every
    record's markers from share 0 and each pair's slab from the share that
    computed it.  Fortran order puts the rain pair in the two slowest
    dimensions, so a share is one contiguous byte run per record.
    """
    records = HAIL_GENERATED_TABLE_FILES[HAIL_MP38_TABLE_FILE]
    slices = len(share_files)
    handles = [path.open("rb") for path in share_files]
    try:
        with output.open("wb") as out:
            for record in records:
                markers = []
                for handle in handles:
                    head = handle.read(4)
                    markers.append(head)
                if len(set(markers)) != 1 or struct.unpack(
                        "<i", markers[0])[0] != record.payload_bytes:
                    raise TableAssetError(
                        f"share files disagree on the {record.name} record")
                out.write(markers[0])
                for slice_index, handle in enumerate(handles):
                    start, end = share_bounds(slice_index, slices)
                    payload = handle.read(record.payload_bytes)
                    if len(payload) != record.payload_bytes:
                        raise TableAssetError(
                            f"truncated {record.name} in share {slice_index}")
                    if end >= start:
                        out.write(payload[start * PAIR_BYTES:
                                          (end + 1) * PAIR_BYTES])
                tails = [handle.read(4) for handle in handles]
                if len(set(tails)) != 1 or tails[0] != markers[0]:
                    raise TableAssetError(
                        f"mismatched trailing marker for {record.name}")
                out.write(tails[0])
            for handle in handles:
                if handle.read(1):
                    raise TableAssetError("bytes after the final record")
    finally:
        for handle in handles:
            handle.close()


def _command(script: Path, work: Path, workers: int) -> list[str]:
    if os.name == "nt":
        distro = os.environ.get("WOOF_THOMPSON_FORK_WSL_DISTRO",
                                "Ubuntu-24.04")
        if shutil.which("wsl.exe") is None:
            raise FileNotFoundError(
                "The hail-aware Thompson table build needs WSL Ubuntu with "
                "gfortran and dpkg-deb, or a directory holding the pinned "
                f"{HAIL_MP38_TABLE_FILE} ({HAIL_TABLE_SOURCE_ROOT_ENV})")
        prefix = ["wsl.exe", "-d", distro, "--"]
        convert = lambda path: subprocess.check_output(  # noqa: E731
            [*prefix, "wslpath", "-a", "-u",
             str(path.resolve()).replace("\\", "/")], text=True).strip()
        return [*prefix, "bash", convert(script), convert(work),
                str(workers)]
    if platform.system() != "Linux" or platform.machine() not in (
            "x86_64", "AMD64"):
        raise FileNotFoundError(
            "The hail-aware Thompson table build needs Linux x86_64 with "
            "gfortran and dpkg-deb, or a directory holding the pinned "
            f"{HAIL_MP38_TABLE_FILE} ({HAIL_TABLE_SOURCE_ROOT_ENV})")
    if shutil.which("gfortran") is None or shutil.which("dpkg-deb") is None:
        raise FileNotFoundError(
            "The hail-aware Thompson table build needs GNU Fortran "
            "(gfortran) and dpkg-deb, or a directory holding the pinned "
            f"{HAIL_MP38_TABLE_FILE} ({HAIL_TABLE_SOURCE_ROOT_ENV})")
    return ["bash", str(script), str(work), str(workers)]


def prepare_reference_runtime(work: Path) -> Path:
    """Prepare the hash-pinned Fortran reference libraries inside owned work."""
    work = Path(work).resolve()
    work.mkdir(parents=True, exist_ok=True)
    sources = work / "sources"
    sources.mkdir(exist_ok=True)
    for asset, base_url in REFERENCE_RUNTIME_ASSETS:
        fetch_asset_from_url(sources, asset, base_url + asset.filename)
    script = work / "prepare-reference-runtime.sh"
    script.write_text(REFERENCE_RUNTIME_SCRIPT, encoding="utf-8", newline="\n")
    done = subprocess.run(_command(script, work, 1), capture_output=True,
                          text=True, timeout=120)
    if done.returncode:
        raise TableAssetError(
            "Fortran table reference runtime could not be prepared; "
            + (done.stdout + done.stderr).strip())
    return work / "runtime" / "lib"


def build_hail_table(work: Path, classic_root: Path, *, log_path: Path,
                     workers: int | None = None) -> Path:
    """Run WRF's own hail-aware ``thompson_init`` and splice its shares.

    ``work`` is a fresh directory this call owns.  Returns the spliced
    ``qr_acr_qg_mp38V1.dat`` inside it, already held to its pin.
    """

    work = Path(work).resolve()
    work.mkdir(parents=True, exist_ok=False)
    workers = worker_count() if workers is None else max(1, int(workers))
    script = work / "build.sh"
    script.write_text(BUILD_SCRIPT, encoding="utf-8", newline="\n")
    command = _command(script, work, workers)
    sources = work / "sources"
    sources.mkdir()
    for asset in PUBLIC_SOURCES:
        fetch_asset_from_url(sources, asset,
                             SOURCE_BASE + "/" + asset.filename)
    harness = _harness_dir()
    for name, expected in _harness_pins().items():
        original = harness / name
        if hashlib.sha256(original.read_bytes()).hexdigest() != expected:
            raise TableAssetError(
                f"packaged hail table harness {original} differs from its "
                "pinned source")
        shutil.copyfile(original, work / name)
    prepare_reference_runtime(work)
    classic = work / "classic"
    classic.mkdir()
    shared_assets = tuple(a for a in HAIL_TABLE_ASSETS if a.filename in
                          ("qr_acr_qsV2.dat", "freezeH2O.dat"))
    valid, invalid, absent = classify_assets(Path(classic_root), shared_assets)
    if not invalid and not absent:
        for name in ("qr_acr_qsV2.dat", "freezeH2O.dat"):
            _link_or_copy(Path(classic_root) / name, classic / name)
    print("woof hail-aware Thompson tables: building WRF v4.6.1's nine-"
          f"slab rain-graupel table on {workers} CPU process(es); first use "
          "only", flush=True)
    try:
        done = subprocess.run(command, capture_output=True, text=True,
                              timeout=7500)
    except (OSError, subprocess.SubprocessError) as error:
        raise TableAssetError(
            f"hail-aware Thompson table build failed: {error}") from error
    logs = [done.stdout, done.stderr]
    for slice_index in range(workers):
        share_log = work / f"share-{slice_index}" / "generate.log"
        if share_log.is_file():
            logs.append(share_log.read_text(encoding="utf-8",
                                            errors="replace"))
    log_path.write_text("\n".join(logs), encoding="utf-8")
    if done.returncode:
        raise TableAssetError(
            f"hail-aware Thompson table build exited {done.returncode}; "
            f"see {log_path}")
    spliced = work / HAIL_MP38_TABLE_FILE
    splice_table_shares(
        [work / f"share-{slice_index}" / HAIL_MP38_TABLE_FILE
         for slice_index in range(workers)], spliced)
    validate_table_assets(work, tuple(
        asset for asset in HAIL_TABLE_ASSETS
        if asset.filename == HAIL_MP38_TABLE_FILE))
    for asset in HAIL_TABLE_ASSETS:
        if asset.filename != HAIL_MP38_TABLE_FILE:
            _link_or_copy(work / "common" / asset.filename, work / asset.filename)
    validate_table_assets(work, HAIL_TABLE_ASSETS)
    receipt = {
        "schema": 1, "command": command, "workers": workers,
        "public_sources": [
            {"filename": a.filename, "sha256": a.sha256, "bytes": a.bytes}
            for a in PUBLIC_SOURCES],
        "reference_runtime": [
            {"filename": a.filename, "sha256": a.sha256, "bytes": a.bytes,
             "url": base + a.filename} for a, base in REFERENCE_RUNTIME_ASSETS],
        "harness_sha256": _harness_pins(),
        "build_script_sha256": hashlib.sha256(
            BUILD_SCRIPT.encode()).hexdigest(),
    }
    (work / "source-build-receipt.json").write_text(
        json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    return spliced


def _link_or_copy(source: Path, target: Path) -> None:
    try:
        os.link(source, target)
    except OSError:
        shutil.copyfile(source, target)


def ensure_thompson_hail_tables(root=None, *, source_dir=None) -> Path:
    """Verify or build the complete hail-aware set under one root lock.

    All four hail-aware generated assets are checked against their own
    size and SHA-256 pins. A supplied source must match the complete set.
    First use rebuilds WRF's scalar tables locally. Classic caches are
    reused only when their bytes also match the hail-aware pins.
    Existing wrong bytes are refused without overwrite.
    """
    from woof.physics_compat import (
        thompson_hail_table_root, thompson_table_root)

    if root is None:
        root = thompson_hail_table_root()
    root = Path(root)
    with fetch_guard.hold("fetch-tables", root):
        _valid, invalid, absent = classify_assets(root, HAIL_TABLE_ASSETS)
        if invalid:
            raise TableAssetError(
                "hail-aware Thompson cache contains different bytes; "
                "refusing without overwrite: " + "; ".join(invalid))
        if not absent:
            return root
        root.mkdir(parents=True, exist_ok=True)
        source = source_dir or os.environ.get(HAIL_TABLE_SOURCE_ROOT_ENV)
        if source:
            source = Path(source)
            validate_table_assets(source, HAIL_TABLE_ASSETS)
            for asset in absent:
                _link_or_copy(source / asset.filename, root / asset.filename)
        else:
            work = root / f".hail-build-{os.getpid()}"
            try:
                build_hail_table(work, Path(thompson_table_root()),
                                log_path=root / "hail-build.log")
                for asset in absent:
                    os.replace(work / asset.filename, root / asset.filename)
                shutil.copyfile(work / "source-build-receipt.json",
                                root / "source-build-receipt.json")
            finally:
                if work.exists() and work.parent.resolve() == root.resolve():
                    shutil.rmtree(work, ignore_errors=True)
        validate_table_assets(root, HAIL_TABLE_ASSETS)
        return root


#: The ``woof fetch-tables --schemes`` family that stages this set.
HAIL_TABLE_FAMILY = "thompson38"
HAIL_STAGE_COMMAND = f"woof fetch-tables --schemes {HAIL_TABLE_FAMILY}"


@lru_cache(maxsize=None)
def stage_hail_table_for_run(root: str, *, force_read: bool = False) -> Path:
    """Stage the verified hail table before WRF's cache policy is applied.

    Named breakage (2.9.12 pass 17 R11): on a bare install nothing put
    ``qr_acr_qg_mp38V1.dat`` in the table root, so WRF's
    ``write_thompson_mp38table = .false.`` refusal (the default, :4160-4163)
    stopped every shipped mp_physics = 38 profile.  WRF's flag governs
    whether ``thompson_init`` computes and writes the table at run time; it
    does not forbid a table that was staged beforehand, and WRF itself reads
    any present file.  So the forecast door stages the set the way
    ``woof fetch-tables --schemes thompson38`` does
    (:func:`ensure_thompson_hail_tables`: ``WOOF_THOMPSON_HAIL_TABLE_SOURCE_ROOT``,
    else the hash-pinned public WRF v4.6.1 build), and installs it only after
    it matches its size and SHA-256 pins.  A failure is a refusal that names
    the table and the staging commands; nothing unverified is ever installed.

    ``force_read_thompson = .true.`` keeps WRF's read-only meaning: nothing
    is staged, and :func:`prepare_thompson_hail_tables` refuses a missing
    table by name.  A present table costs one ``stat`` here; its bytes are
    verified once by :func:`prepare_thompson_hail_tables`.
    """
    path = Path(root)
    if force_read or (path / HAIL_MP38_TABLE_FILE).is_file():
        return path
    import sys
    mib = next(a.bytes for a in HAIL_TABLE_ASSETS
               if a.filename == HAIL_MP38_TABLE_FILE) / 2 ** 20
    print(f"woof: mp_physics = 38 needs {HAIL_MP38_TABLE_FILE} ({mib:.0f} MiB), "
          f"absent from {path}; staging the verified table ({HAIL_STAGE_COMMAND})",
          file=sys.stderr, flush=True)
    try:
        staged = ensure_thompson_hail_tables(path)
    except (OSError, ValueError, TableAssetError) as error:
        raise TableAssetError(
            f"mp_physics = 38 needs the hail-aware Thompson table "
            f"{HAIL_MP38_TABLE_FILE} ({mib:.0f} MiB) in {path}, and staging it "
            f"failed: {error}. Stage it with `{HAIL_STAGE_COMMAND}` (or "
            f"`{HAIL_STAGE_COMMAND} --from DIR` for a directory holding the pinned "
            f"files), or name such a directory with {HAIL_TABLE_SOURCE_ROOT_ENV}"
            ) from error
    print(f"woof: verified and installed {staged / HAIL_MP38_TABLE_FILE}",
          file=sys.stderr, flush=True)
    return staged


# Keep a non-persistent build alive until process exit. CUDA table owners read
# these paths once; deleting them earlier would break a later domain's owner.
_TRANSIENT_HAIL_TABLES = []


@lru_cache(maxsize=None)
def prepare_thompson_hail_tables(root: str, *, force_read: bool = False,
                                write_tables: bool = True,
                                write_mp38table: bool = False) -> Path:
    """Apply WRF :4113-4164 startup cache policy, once per configuration.

    A complete verified cache needs no write under any of the three controls.
    Missing tables obey force_read_thompson; the hail table additionally
    requires write_thompson_mp38table. With writes disabled, native-generated
    missing records live in process-owned temporary storage, never the cache.
    """
    path = Path(root)
    _valid, invalid, absent = classify_assets(path, HAIL_TABLE_ASSETS)
    if invalid:
        raise TableAssetError("hail-aware Thompson cache contains different bytes; "
                              "refusing without overwrite: " + "; ".join(invalid))
    if not absent:
        return path
    missing = {asset.filename for asset in absent}
    if force_read:
        raise TableAssetError("force_read_thompson is true and verified native "
                              "table records are missing: " + ", ".join(sorted(missing)))
    if HAIL_MP38_TABLE_FILE in missing and not write_mp38table:
        raise TableAssetError("write_thompson_mp38table is false and the native "
                              "hail table is missing: " + HAIL_MP38_TABLE_FILE)
    if write_tables:
        return ensure_thompson_hail_tables(path)
    temporary = tempfile.TemporaryDirectory(prefix="gpuwm-thompson-hail-")
    work = Path(temporary.name)
    try:
        for asset in HAIL_TABLE_ASSETS:
            if asset.filename not in missing:
                _link_or_copy(path / asset.filename, work / asset.filename)
        result = ensure_thompson_hail_tables(work)
    except BaseException:
        temporary.cleanup()
        raise
    _TRANSIENT_HAIL_TABLES.append(temporary)
    return result


__all__ = [
    "HAIL_STAGE_COMMAND",
    "HAIL_TABLE_FAMILY",
    "HAIL_TABLE_SOURCE_ROOT_ENV",
    "HAIL_TABLE_WORKERS_ENV",
    "PUBLIC_SOURCES",
    "build_hail_table",
    "ensure_thompson_hail_tables",
    "prepare_thompson_hail_tables",
    "stage_hail_table_for_run",
    "share_bounds",
    "splice_table_shares",
    "worker_count",
]
