"""Build the MPAS hex-cuda ``--physics-resources`` packet from this installation.

THE BREAKAGE THIS PREVENTS.  Every MPAS hex-cuda slot whose scheme reads
tables or shared device maths (Thompson, MYNN, Noah, RRTMG LW and SW) refused
to run without a ``--physics-resources`` packet, and the only packets that
existed named files in lane scratch directories on one build host.  A clean
``pip install woof hex`` could not reach the MPAS physics suites at all.

The MPAS route fetches pinned release resources on first use, without a host
compiler (resources_fetch). ``woof hex physics-resources build`` remains the
explicit source-build alternative, using the inputs below:

* device maths: assembled from the sources vendored under
  ``woof/hex/data/mpasphys-maths`` by the recipes the scheme lanes certified,
  refused unless each header equals its pinned SHA-256;
* table decoders: the Rust executables and library built into the wheel from
  ``rust/physics-tables`` (``woof/hex/libexec``), refused unless each decodes
  its pinned table to the same words as the certified decoder;
* MPAS-Data tables: fetched by SHA-256 at the pinned MPAS-Data commit, or read
  from a local ``--mpas-source`` tree;
* Thompson strict tables and auxiliary arrays: generated only by the source-checkout
  referee tool in tools/native_dry_reference/thompson_tables.py, using GNU Fortran from
  the pinned MPAS sources (fetched by SHA-256) with the strict recipe and the
  vendored Thompson host maths and CORE-MATH log10f (PINNED_LOG10F), refused unless every byte equals
  the strict native reference's.

The default MPAS route reads its Registry declarations from packaged metadata
and requests only the Rust pool reader plus the CUDA resources packet. Release
assets contain DATA only; wheel resources are assembled locally. The
explicit source-build command still records those proof resources under the
portable RESOURCE_KEYS for native referee tools and verification decks.

RRFS-MPAS runs on the same ported framework (woof.hex.mpasphys.rrfs.framework_rrfs)
with no native library: its route needs the pool reader, the physics packet and
the fork's DATA (Registry, TEMPO v3 tables, ccn_activate.bin; assemble_rrfs_source).
No native library or Fortran runtime is distributed; the RRFS native library is a
referee built only when asked for, as v8.4.2's.

Every artifact lives in a cache directory under its own SHA-256 and every step
reuses what is already there and verified, so a repeat build costs only the
hashing.  The packet file is named by the SHA-256 of its own text.
"""
from __future__ import annotations
from woof.hex import numerics as _hex_numerics

import argparse
import hashlib
import http.client
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from typing import Any, Callable
import urllib.request

SCHEMA = "gpuwm-hex.physics-resources-build/v1"

# Upstream pins (the same commits the native reference is built from).
MPAS_MODEL_COMMIT = "f34984b2c89353cf31c235b08d3b4acd96b37bb3"   # MPAS-Model v8.4.2
MMM_PHYSICS_COMMIT = "a4baf7f3243d1db0dbc5f63473f895bdbdc05c30"  # MMM-physics 20250616-MPASv8.3
MPAS_DATA_COMMIT = "c57dbc7be629802c6e848770a9e44b9bc602be41"     # MPAS-Data v8.2

UGWP_COMMIT = "c1c893edcf171af5639af60e3a3a528816f6cc2b"          # NOAA-GSL UGWP MPAS_20241223

# The RRFS pin set: the GSL v8.3.1 fork as rrfs-workflow pins it, and its submodule pins (the same MMM-physics
# and UGWP commits as v8.4.2).  woof.hex.mpasphys.native_backend checks these pins' digest on load.
RRFS_MODEL_COMMIT = "1d6e370e7b43a332074809f170532a216c2212fb"     # RRFSx/MPAS-Model (MPAS-v8.3.1-2.26)
RRFS_SUBMODULES = {
    "tempo": ("NCAR/TEMPO", "9adf9ef2827104640125f0c322e1218fce74e94c", "physics_noaa/TEMPO"),
    "mynn-edmf": ("NCAR/MYNN-EDMF", "c942418a47a29dee09b51b377286da6bb25160e3", "physics_noaa/MYNN-EDMF"),
    "mynn-sfc": ("NCAR/MYNN-SFC", "b66187f54a0c169e36592552634867474e603ac0", "physics_noaa/MYNN-SFC"),
    "ruclsm": ("NOAA-GSL/RUCLSM", "8c9f1a8bb2cfb5fb8d115440fced566361fe0f20", "physics_noaa/RUCLSM"),
    "gfl": ("NOAA-GSL/GFL", "bf6a4b0d58f112322fbf9b11d43ed2027aeb56fa", "physics_noaa/GFL"),
    "smoke": ("NOAA-GSL/SMOKE", "9652bbf152f8b946bf04f72a2b48ff67e15d7973", "physics_noaa/SMOKE"),
}
# NOAA's published RRFS fix files (the release rrfs-workflow stages), fetched by SHA-256.
NOAA_GSL_FIX = "https://noaa-gsl-experimental-pds.s3.amazonaws.com/fix_rrfs2/v2.1.4-3.4/main/{path}"

URLS = {
    "mpas-model": "https://raw.githubusercontent.com/MPAS-Dev/MPAS-Model/" + MPAS_MODEL_COMMIT + "/{path}",
    "mmm-physics": "https://raw.githubusercontent.com/NCAR/MMM-physics/" + MMM_PHYSICS_COMMIT + "/{path}",
    "ugwp": "https://raw.githubusercontent.com/NOAA-GSL/UGWP/" + UGWP_COMMIT + "/{path}",
    "mpas-data": "https://raw.githubusercontent.com/MPAS-Dev/MPAS-Data/" + MPAS_DATA_COMMIT + "/{path}",
    "rrfs-model": "https://raw.githubusercontent.com/RRFSx/MPAS-Model/" + RRFS_MODEL_COMMIT + "/{path}",
    **{origin: f"https://raw.githubusercontent.com/{repository}/{commit}/{{path}}"
       for origin, (repository, commit, _) in RRFS_SUBMODULES.items()},
    "noaa-gsl-fix": NOAA_GSL_FIX,
}

# Where an origin's file sits inside an assembled MPAS-Model v8.4.2 tree
# (MPAS's own build copies the MPAS-Data tables into physics_wrf/files).
_PHYSICS_FILES = "src/core_atmosphere/physics/physics_wrf/files/"


def _local_path(origin: str, path: str) -> str:
    if origin in ("mpas-model", "rrfs-model"):
        return path
    if origin in RRFS_SUBMODULES:
        return "src/core_atmosphere/physics/" + RRFS_SUBMODULES[origin][2] + "/" + path
    if origin == "mmm-physics":
        return "src/core_atmosphere/physics/physics_mmm/" + path
    if origin == "ugwp":
        return "src/core_atmosphere/physics/physics_noaa/UGWP/" + path
    return _PHYSICS_FILES + Path(path).name


# name -> (origin, upstream path, sha256).  The Noah rows equal
# noah_v842.TABLE_FILE_SHA256 (tests/test_physics_resources_build.py holds it).
TABLE_FILES = {
    "RRTMG_LW_DATA": ("mpas-data", "atmosphere/physics_wrf/files/RRTMG_LW_DATA",
                      "bcfdee24b63a4c909522a329b8e16c539f0173c7e5aea2caf933ab4fe28c5c97"),
    "RRTMG_SW_DATA": ("mpas-data", "atmosphere/physics_wrf/files/RRTMG_SW_DATA",
                      "a7d25f5b4d33be8629cbef7ecacc1ff413bf398a021297793e843ba1cc627baf"),
    "LANDUSE.TBL": ("mpas-data", "atmosphere/physics_wrf/files/LANDUSE.TBL",
                    "7f661318ff5f06aed6b4b5508e4d077dc794cff03d5fb4447fe59c88d0ed2ece"),
    "VEGPARM.TBL": ("mpas-data", "atmosphere/physics_wrf/files/VEGPARM.TBL",
                    "ed5478afbe49af51492256c1eb6cf88b3948590308525b32df1ddec28687b40e"),
    "SOILPARM.TBL": ("mpas-data", "atmosphere/physics_wrf/files/SOILPARM.TBL",
                     "1e2275a32d8cd3b48ca693d22c0816df0013f83b6594ac632716361db337d58f"),
    "GENPARM.TBL": ("mpas-data", "atmosphere/physics_wrf/files/GENPARM.TBL",
                    "9c02832a0e4a2ecaf47fcee485539aad95cd732c379c5c258161a88eb3d25ea2"),
    "CCN_ACTIVATE_DATA": ("mpas-data", "atmosphere/physics_wrf/files/CCN_ACTIVATE_DATA",
                          "f2b8d3916560f9046f89f8ac5f32c5292a1800498fd75301e422f147c82a3dbd"),
}

# The pristine Fortran the Thompson strict tables and auxiliary arrays are
# generated from (module_mp_thompson.F:433-993; build_tables.F).
THOMPSON_SOURCES = {
    "module_mp_thompson.F": ("mpas-model", "src/core_atmosphere/physics/physics_wrf/module_mp_thompson.F",
                             "1ba1f30d486bc8e2a2e6295f8495877f963f7995a047558f8a68ff80012a70e5"),
    "mpas_atmphys_functions.F": ("mpas-model", "src/core_atmosphere/physics/mpas_atmphys_functions.F",
                                 "2cf3763c2781468ed59134acecfeb04a903859356ab30c18d48e595e5688e37e"),
    "build_tables.F": ("mpas-model", "src/core_atmosphere/utils/build_tables.F",
                       "ae965b9406b743f867d3eae90c6d7bb524a5c9cbe056fcd3d67d8fafd60f0200"),
    "atmphys_build_tables_thompson.F": ("mpas-model", "src/core_atmosphere/utils/atmphys_build_tables_thompson.F",
                                        "6341d37bea31a64fac76743bc674e04ba1293bc7b299dc41bd628cf7a5b34e9c"),
    "mp_radar.F90": ("mmm-physics", "mp_radar.F90",
                     "f349afc8f5068173320856f11e9b4b1f769761803ce282f2832340b96e6f90b3"),
}
# The strict recipe of the native reference build (tools/native_physics_reference),
# without the optional coverage instrumentation the lane oracle added.
STRICT_FFLAGS = ("-O3", "-ffp-contract=off", "-fno-tree-vectorize", "-fno-unsafe-math-optimizations",
                 "-ffree-form", "-ffree-line-length-none", "-fPIC", "-cpp", "-Dmpas", "-fconvert=big-endian")
_THOMPSON_OBJECTS = ("framework_stubs", "mpas_atmphys_functions", "mp_radar", "module_mp_thompson")
# The generators' one host-maths dependence, pinned.  freezeH2O's T_adjust calls
# single-precision ALOG10 (module_mp_thompson.F, subroutine freezeH2O).  The
# reference's glibc 2.43 log10f is CORE-MATH's correctly rounded one; older C
# libraries round some of those values differently (glibc 2.39, Ubuntu 24.04),
# MP_THOMPSON_freezeH2O_DATA.DBL followed the host, and the build refused on a
# standard Linux install, so no MPAS suite could run from it.  The generators
# link the vendored CORE-MATH log10f (woof/hex/data/mpasphys-maths, MIT, pinned
# in SOURCES.json) as log10f. The other transcendental calls use the same
# pinned Thompson host maths as the device ports. This also preserves every
# strict table byte with GNU Fortran 14.2.1 on the release's glibc 2.28.
PINNED_LOG10F = "core-math/log10f.c"
PINNED_LOG10F_CFLAGS = ("-O2", "-ffp-contract=off", "-fno-builtin", "-fPIC", "-Dcr_log10f=log10f")
HOST_MATH_FUNCTIONS = ("expf", "logf", "powf", "exp", "log", "pow", "log10")
HOST_MATH_LINK = tuple("-Wl,--wrap=" + name for name in HOST_MATH_FUNCTIONS)
HOST_MATH_COMPILE = ("g++", "-O2", "-ffp-contract=off", "-fno-fast-math", "-fPIC",
                     "-std=c++17", "-c", "thompson_host_math.cpp", "-o", "thompson_host_math.o")

# The shared device maths each scheme binds, and the SHA-256 its lane certified.
MATHS = {
    "bl_mynn": ("bl-mynn-shared-maths.cuh",
                "90d9c883180c2edc7ca26f9067d06519228c55e88252edc1e857f0351c302fea"),
    "mp_thompson": ("mp-thompson-glibc243-log10-maths.cuh",
                    "1819ce9648539d700544837ff3f81e017f045f153dc51725fbca08920258a49e"),
    # The RRFS fork's TEMPO v3.0.3 port (rrfs.tempo_v303): the maths its real
    # replay and forecast packet certified (tests/tempo_rrfs/prepare_shared_math.py
    # plus gpu_replay.shared_math).
    "mp_tempo": ("mp-tempo-rrfs-maths.cuh",
                 "13b194710cacf584f00a794dd311dcb6bf6d351179609a5feb1c10e23bd233c4"),
}

DECODERS = ("lookup_decode", "table_decode", "rw_mpas_rrtmg_lw_tables", "rw_mpas_rrtmg_sw_tables",
            "ruc_table_decode", "tempo_table_compute",
            "tempo_table_init")

# What the certified decoders (the binaries the hl-3 headline packet pins)
# decode the pinned tables to; a rebuilt decoder must produce the same words.
# Measured with decoded_digests() on that packet.  Noah's depends on the
# landuse section, so only the rows below are pinned; others are recorded.
DECODED_SHA256 = {
    "mp_thompson": "9f6470e055c5fc7a0cc0109e8fb09590159dff61aaf7e4d72a6876d385cf57fc",
    "rrtmg_lw": "3ad7bf1093d536d238501e7556ef2d35147e651ca1e92b3eb66c05dd41d7caef",
    "rrtmg_sw": "eccacfa5da899b5afc23d10dc55e21078548ee361cf396ebf970e44bb9a5b917",
    "sf_noah:MODIFIED_IGBP_MODIS_NOAH": "708f0313fc237755a32bb4bf8e9d853adb98c3e98f794892e6ce5399542cd67f",
    "sf_noah:USGS": "cdceda3b430f264f9d7767c7a659db48dd54f92f6ff42fbaa08b666b677368e3",
}

DEFAULT_LANDUSE = "MODIFIED_IGBP_MODIS_NOAH"   # MPAS init's default config_landuse_data
SOIL_TYPE = "STAS"                             # mpas_atmphys_lsm_noahinit.F mminsl

# Slots with a resource descriptor this command does not build, and why.
NOT_BUILT = {
    # Breakage named (G0 on integrate 9b9c141c): p17 landed a hex-cuda sf_noahmp
    # entry whose resource is a certified offline CUDA image, and this table did
    # not name it, so the builder silently left it out of not_built.
    "sf_noahmp": "its packet is a certified offline Noah-MP CUDA image plus its compilation receipt, "
                 "compiled by NVRTC for one device architecture and one workspace capacity "
                 "(noahmp_v842.load_cuda_resources, compile_cuda; tests/noahmp_v842/"
                 "compile_cuda_artifact.py); this command does not compile device images yet",
}
# Schemes whose slot descriptor binds the Thompson loader too; each gets the
# mp_thompson entry when its slot module is in this tree.  Aerosol-aware Thompson
# binds its own loader and auxiliary (build_thompson_aero_auxiliary): the shared
# entry carried the unit CCN table, which that loader refuses, so the slot could
# not run from an install.
THOMPSON_SHARING: tuple[str, ...] = ()


def _module_present(scheme: str) -> bool:
    import importlib.util
    from ..physics_suites import load_table
    for row in load_table()["mpasphys_v842"]["slots"].values():
        descriptor = row.get("cuda_entries", {}).get(scheme)
        if descriptor is not None:
            return importlib.util.find_spec(f"{__package__}.{descriptor['module']}") is not None
    return False


class PhysicsResourceError(ValueError):
    """A resource that cannot be built, named with what closes it."""


def _log(message: str) -> None:
    print(f"woof hex physics-resources: {message}", file=sys.stderr, flush=True)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def default_cache() -> Path:
    """``$WOOF_HEX_CACHE_ROOT/physics-resources``, else the XDG cache."""
    root = os.environ.get("WOOF_HEX_CACHE_ROOT")
    if root:
        return Path(root) / "physics-resources"
    base = os.environ.get("XDG_CACHE_HOME")
    return (Path(base) if base else Path.home() / ".cache") / "woof hex" / "physics-resources"


def _link_relative(link: Path, source: Path) -> None:
    """Point ``link`` at ``source`` by a path relative to the link's own folder.

    THE BREAKAGE THIS PREVENTS: the MPAS trees' physics_wrf/files links (OZONE_PLEV.TBL and the other tables)
    were absolute, so a physics-resources cache copied to another path (an image's builder stage into its
    final stage, a box's disk) still pointed at the original; with the original moved or deleted every
    MPAS-physics run stopped in oznini ("failure opening OZONE_PLEV.TBL", 2026-10-17).  A relative link
    moves with the cache.  An existing absolute link is replaced by its relative form on the next build.
    """
    target = os.path.relpath(Path(source).resolve(), Path(link).parent.resolve())
    if link.is_symlink() and os.readlink(link) == target:
        return
    if link.is_symlink() or link.exists():
        link.unlink()
    link.symlink_to(target)


def _recorded_root(record: dict[str, Any]) -> Path | None:
    """The cache folder an installed record was written in (older records: inferred from the packet row)."""
    root = record.get("cache_root")
    if root:
        return Path(root)
    row = record.get("resources", {}).get("physics-resources")
    if row and Path(row["path"]).parent.name == "packets":
        return Path(row["path"]).parent.parent
    return None


def _rebased(path: Path, recorded: Path | None, current: Path) -> Path:
    """``path`` under the cache's current folder when it was recorded under the folder the record was written in.

    THE BREAKAGE THIS PREVENTS: installed.json named every resource by the absolute path of the folder it was
    built in, so a cache copied elsewhere resolved to the original (or, with the original gone, rebuilt the
    whole native stack on first use).  Paths outside the recorded folder (an adopted --mpich, a named
    library) are kept as they are.
    """
    if recorded is None or not path.is_absolute():
        return path if path.is_absolute() else current / path
    try:
        relative = path.relative_to(recorded)
    except ValueError:
        return path
    return current / relative


def _packet_paths(value, transform):
    if isinstance(value, dict):
        return {key: _packet_paths(item, transform) for key, item in value.items()}
    if isinstance(value, list):
        return [_packet_paths(item, transform) for item in value]
    return transform(value) if isinstance(value, str) else value


def portable_packet(packet: dict[str, Any], cache: Path) -> dict[str, Any]:
    """Store internal packet paths relative to its packets directory."""
    cache = Path(cache).resolve()
    def relative(value):
        path = Path(value)
        if path.is_absolute() and path.is_relative_to(cache):
            return os.path.relpath(path, cache / "packets")
        return value
    return _packet_paths(packet, relative)


def load_packet(path: Path) -> dict[str, Any]:
    """Resolve portable packets and rebase older cache packets after a move."""
    if (os.environ.get('WOOF_HEX_MOVING_RESIDENT') == '1'
            and os.environ.get('WOOF_HEX_MOVING_RESIDENT_TABLES','1') != '0'):
        from .moving_tables_v842 import packet
        return packet(path, _read_packet)
    return _read_packet(path)


def _read_packet(path: Path) -> dict[str, Any]:
    path = Path(path).resolve()
    recorded = None
    receipt = path.with_suffix(".receipt.json")
    if path.parent.name == "packets" and receipt.is_file():
        root = json.loads(receipt.read_text()).get("cache")
        recorded = Path(root) if root else None
    def absolute(value):
        item = Path(value)
        if item.is_absolute():
            return str(_rebased(item, recorded, path.parent.parent))
        if value.startswith(("../", "./")):
            return str((path.parent / item).resolve())
        return value
    return _packet_paths(json.loads(path.read_text()), absolute)


def _data_root(name: str) -> Path:
    return Path(__file__).resolve().parents[1] / "data" / name


def _replace_into(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    os.replace(source, target)


def _copy_verified(source: Path, target: Path, sha256: str) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(dir=target.parent, prefix=".partial-")
    os.close(handle)
    shutil.copyfile(source, temporary)
    if sha256_file(Path(temporary)) != sha256:
        os.unlink(temporary)
        raise PhysicsResourceError(f"{source} changed while it was copied")
    _replace_into(Path(temporary), target)
    return target


def fetch(name: str, origin: str, path: str, sha256: str, target: Path, *,
          mpas_source: Path | None = None, offline: bool = False, origins: dict | None = None) -> Path:
    """One pinned upstream file at ``target``: reused, copied or downloaded, always by SHA-256."""
    if target.is_file() and sha256_file(target) == sha256:
        return target
    origin_row = (origins or {}).get(origin)
    if mpas_source is not None:
        relative = origin_row["prefix"] + path if origin_row else _local_path(origin, path)
        local = Path(mpas_source) / relative
        if local.is_file():
            if sha256_file(local) != sha256:
                raise PhysicsResourceError(
                    f"{name}: {local} is not the pinned file (sha256 {sha256}); --mpas-source must be "
                    f"MPAS-Model {MPAS_MODEL_COMMIT[:7]} with MPAS-Data {MPAS_DATA_COMMIT[:7]} tables")
            return _copy_verified(local, target, sha256)
    return fetch_url(name, (origin_row["url"] if origin_row else URLS[origin]).format(path=path), sha256, target, offline=offline,
                     remedy="pass --mpas-source with a local MPAS-Model v8.4.2 tree, or retry with network access")


def fetch_url(name: str, url: str, sha256: str, target: Path, *, offline: bool = False,
              remedy: str = "retry with network access") -> Path:
    """``url``'s bytes at ``target``, refused unless they hash to ``sha256``; a verified copy is reused."""
    if target.is_file() and sha256_file(target) == sha256:
        return target
    if offline:
        raise PhysicsResourceError(f"{name}: not found locally and --offline forbids fetching {url}")
    target.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(dir=target.parent, prefix=".partial-")
    os.close(handle)
    try:
        with urllib.request.urlopen(url, timeout=300) as response, open(temporary, "wb") as out:
            shutil.copyfileobj(response, out)
    except OSError as error:
        os.unlink(temporary)
        raise PhysicsResourceError(f"{name}: could not fetch {url} ({error}); {remedy}") from error
    actual = sha256_file(Path(temporary))
    if actual != sha256:
        os.unlink(temporary)
        raise PhysicsResourceError(f"{name}: {url} has sha256 {actual}, not the pinned {sha256}")
    _replace_into(Path(temporary), target)
    return target


# ---------------------------------------------------------------- concurrent pinned fetch

# THE BREAKAGE THIS PREVENTS (rrfs/table-fetch-parallel, 2026-10-28): a plain install's first hrrrv5 run fetched the
# RRFS fork's table tree (999 MB, 970 MB of it the three TEMPO v3 lookups) one file at a time over one connection:
# 7.9 min on an RTX 5090 host while the physics-assets fetch beside it used 8 workers, and a fetch stopped part way began the
# 675 MB qr_acr_qg table again from byte 0.  fetch_pinned places many pinned files at once: files fetch concurrently,
# a file larger than FETCH_CHUNK_BYTES as concurrent byte ranges written in place into one partial file whose chunk
# journal lets a later fetch resume it, and every file is published only when its whole bytes hash to the pinned
# SHA-256, so the published bytes are the pinned bytes whatever order they arrived in.
FETCH_CHUNK_BYTES = 16 << 20
FETCH_ATTEMPTS = 4
FETCH_JOBS_ENV = "WOOF_HEX_FETCH_JOBS"
_FETCH_JOURNAL = "gpuwm-hex.ranged-fetch/v1"
_FETCH_REMEDY = "pass --mpas-source with a local MPAS-Model v8.4.2 tree, or retry with network access"


def fetch_jobs(jobs: int | None = None) -> int:
    """Concurrent fetch workers: ``jobs``, else $WOOF_HEX_FETCH_JOBS, else 8 (the physics-assets fetch's count)."""
    if jobs is None:
        value = os.environ.get(FETCH_JOBS_ENV, "").strip()
        try:
            jobs = int(value) if value else 8
        except ValueError:
            raise PhysicsResourceError(f"{FETCH_JOBS_ENV}={value!r} is not a worker count") from None
    if not 1 <= jobs <= 64:
        raise PhysicsResourceError(f"fetch workers must be between 1 and 64, not {jobs}")
    return jobs


class _RangesRefused(Exception):
    """The server answered a byte-range request with something other than that range (no range support)."""


def _retrying(name: str, url: str, attempt: Callable[[], Any]) -> Any:
    """``attempt()``, repeated on a transient network failure (reset, timeout, short read, 408/429/5xx)."""
    import urllib.error
    for number in range(FETCH_ATTEMPTS):
        try:
            return attempt()
        except (OSError, http.client.HTTPException) as error:
            transient = (error.code in (408, 425, 429) or error.code >= 500
                         if isinstance(error, urllib.error.HTTPError) else True)
            if not transient or number + 1 == FETCH_ATTEMPTS:
                raise PhysicsResourceError(f"{name}: could not fetch {url} ({error}); {_FETCH_REMEDY}") from error
            time.sleep(min(2 ** number, 15))


def _download_whole(name: str, url: str, sha256: str, target: Path) -> Path:
    """``url`` over one connection into a private temporary, published only if it hashes to ``sha256``."""
    target.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(dir=target.parent, prefix=".partial-")
    os.close(handle)
    try:
        def attempt():
            with urllib.request.urlopen(url, timeout=300) as response, open(temporary, "wb") as out:
                shutil.copyfileobj(response, out, 1 << 20)
                expected = response.headers.get("Content-Length")
                if expected and expected.isdigit() and out.tell() != int(expected):
                    raise http.client.IncompleteRead(b"", int(expected) - out.tell())   # cut short: retried
        _retrying(name, url, attempt)
        actual = sha256_file(Path(temporary))
        if actual != sha256:
            raise PhysicsResourceError(f"{name}: {url} has sha256 {actual}, not the pinned {sha256}")
        _replace_into(Path(temporary), target)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return target


class _RangedFile:
    """One large pinned file fetched as concurrent byte ranges into a resumable partial file beside its target.

    The partial file (``.partial-<name>``) is sized up front and each range is written at its own offset; the
    journal (``.partial-<name>.chunks``) names the fetch (URL, SHA-256, size, chunk) on its first line and then one
    finished chunk per line, so a fetch stopped part way resumes with the chunks it lacks.  The whole file is
    hashed once every chunk is in: equal to the pin, it replaces the target; otherwise nothing is published.
    """

    def __init__(self, name: str, url: str, sha256: str, size: int, target: Path, chunk: int):
        self.name, self.url, self.sha256, self.size, self.target, self.chunk = name, url, sha256, size, target, chunk
        self.partial = target.parent / f".partial-{target.name}"
        self.journal = target.parent / f".partial-{target.name}.chunks"
        self.count = -(-size // chunk)
        self.lock = threading.Lock()
        self.refused = False      # the server sent no ranges: fetched over one connection instead
        self.restart = False      # resumed chunks did not hash to the pin: fetched again from byte 0
        self.done = self._resume()
        self.resumed = len(self.done)
        self.resumed_size = self.resumed_bytes()

    def _header(self) -> bytes:
        return (json.dumps({"schema": _FETCH_JOURNAL, "url": self.url, "sha256": self.sha256, "bytes": self.size,
                            "chunk": self.chunk}, sort_keys=True) + "\n").encode()

    def _resume(self) -> set[int]:
        self.target.parent.mkdir(parents=True, exist_ok=True)
        done: set[int] = set()
        try:
            text = self.journal.read_bytes()
            header = self._header()
            if text.startswith(header) and self.partial.stat().st_size == self.size:
                # Only newline-terminated entries count: a line cut short by a stop is not a finished chunk.
                body = text[len(header):]
                for line in body[:body.rfind(b"\n") + 1].split(b"\n"):
                    if line.isdigit() and int(line) < self.count:
                        done.add(int(line))
                tail = header + b"".join(f"{index}\n".encode() for index in sorted(done))
                handle, temporary = tempfile.mkstemp(dir=self.target.parent, prefix=".journal-")
                with os.fdopen(handle, "wb") as stream:
                    stream.write(tail)
                os.replace(temporary, self.journal)
                return done
        except OSError:
            pass
        self.discard()
        descriptor = os.open(self.partial, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | getattr(os, "O_BINARY", 0), 0o600)
        try:
            os.ftruncate(descriptor, self.size)
        finally:
            os.close(descriptor)
        self.journal.write_bytes(self._header())
        return done

    def discard(self) -> None:
        self.partial.unlink(missing_ok=True)
        self.journal.unlink(missing_ok=True)

    def resumed_bytes(self) -> int:
        return sum(min(self.chunk, self.size - index * self.chunk) for index in self.done)

    def missing(self) -> list[int]:
        return [index for index in range(self.count) if index not in self.done]

    def fetch_range(self, index: int) -> None:
        start = index * self.chunk
        end = min(self.size, start + self.chunk) - 1

        def attempt():
            request = urllib.request.Request(self.url, headers={"Range": f"bytes={start}-{end}"})
            with urllib.request.urlopen(request, timeout=300) as response:
                answered = re.fullmatch(r"bytes\s+(\d+)-(\d+)/(\d+|\*)",
                                         (response.headers.get("Content-Range") or "").strip())
                if getattr(response, "status", None) != 206 or answered is None:
                    raise _RangesRefused(self.url)
                if answered.group(3) not in ("*", str(self.size)):
                    raise PhysicsResourceError(f"{self.name}: {self.url} is {answered.group(3)} bytes, not the "
                                               f"pinned {self.size} (sha256 {self.sha256})")
                if (int(answered.group(1)), int(answered.group(2))) != (start, end):
                    raise _RangesRefused(self.url)
                with open(self.partial, "r+b") as out:
                    out.seek(start)
                    remaining = end - start + 1
                    while remaining:
                        block = response.read(min(remaining, 1 << 20))
                        if not block:
                            raise http.client.IncompleteRead(b"", remaining)   # cut short: retried
                        out.write(block)
                        remaining -= len(block)

        try:
            _retrying(self.name, self.url, attempt)
        except _RangesRefused:
            self.refused = True
            return
        with self.lock:
            self.done.add(index)
            with open(self.journal, "ab") as stream:
                stream.write(f"{index}\n".encode())
            last = len(self.done) == self.count
        if last:
            self._publish()

    def _publish(self) -> None:
        actual = sha256_file(self.partial)
        if actual != self.sha256:
            self.discard()
            if self.resumed:
                self.restart = True
                return
            raise PhysicsResourceError(f"{self.name}: {self.url} has sha256 {actual}, not the pinned {self.sha256}")
        _replace_into(self.partial, self.target)
        self.journal.unlink(missing_ok=True)


def _gather(futures: list) -> list:
    """Every future's result; on the first failure the queued work is cancelled and the failure raised."""
    from concurrent.futures import FIRST_EXCEPTION, wait
    finished, pending = wait(futures, return_when=FIRST_EXCEPTION)
    failed = [future for future in futures if future in finished and future.exception() is not None]
    if failed:
        for future in pending:
            future.cancel()
        wait(pending)
        raise failed[0].exception()
    return [future.result() for future in futures]


def _origin_url(row, origins):
    """A pinned row's download URL: the source manifest's own origin table (a pin set's), else URLS."""
    origin_row = (origins or {}).get(row["origin"])
    return (origin_row["url"] if origin_row else URLS[row["origin"]]).format(path=row["path"])


def fetch_pinned(rows: list[dict[str, Any]], *, mpas_source: Path | None = None, offline: bool = False,
                 jobs: int | None = None, log: Callable[[str], None] = _log,
                 origins: dict | None = None) -> list[Path]:
    """Place every pinned row as ``fetch`` places one (reused, copied or downloaded, always by SHA-256), concurrently.

    ``rows`` carry name, origin, path, sha256, bytes (None if unknown) and target; the result is each row's target.
    A download larger than FETCH_CHUNK_BYTES goes as concurrent byte ranges (_RangedFile, resumable); a server
    that sends no ranges gets one connection for that file.  ``jobs`` workers (fetch_jobs) share all of it.  The
    ranged partial files have fixed names, so a caller holds a lock on the cache while it fetches.
    """
    from concurrent.futures import ThreadPoolExecutor
    started = time.perf_counter()
    jobs = fetch_jobs(jobs)
    pool = ThreadPoolExecutor(max_workers=jobs, thread_name_prefix="hex-fetch")
    try:
        def settle(row):
            # fetch()'s own steps for a verified copy, a local tree or --offline; None means download it.
            target = Path(row["target"])
            if target.is_file() and sha256_file(target) == row["sha256"]:
                return target
            origin_row = (origins or {}).get(row["origin"])
            relative = origin_row["prefix"] + row["path"] if origin_row else _local_path(row["origin"], row["path"])
            local = Path(mpas_source) / relative if mpas_source is not None else None
            if offline or (local is not None and local.is_file()):
                return fetch(row["name"], row["origin"], row["path"], row["sha256"], target,
                             mpas_source=mpas_source, offline=offline, origins=origins)
            return None

        placed = _gather([pool.submit(settle, row) for row in rows])
        wanted = sorted((index for index, path in enumerate(placed) if path is None),
                        key=lambda index: -(rows[index]["bytes"] or 0))
        if not wanted:
            return placed
        ranged, whole = [], []
        for index in wanted:
            row = rows[index]
            url = _origin_url(row, origins)
            if row["bytes"] and row["bytes"] > FETCH_CHUNK_BYTES:
                ranged.append((index, _RangedFile(row["name"], url, row["sha256"], int(row["bytes"]),
                                                  Path(row["target"]), FETCH_CHUNK_BYTES)))
            else:
                whole.append(index)
        resumed = sum(state.resumed_size for _, state in ranged)
        while ranged or whole:
            work = [pool.submit(state.fetch_range, chunk) for _, state in ranged for chunk in state.missing()]
            work += [pool.submit(_download_whole, rows[index]["name"],
                                 _origin_url(rows[index], origins),
                                 rows[index]["sha256"], Path(rows[index]["target"])) for index in whole]
            _gather(work)
            whole, again = [], []
            for index, state in ranged:
                if state.refused:
                    state.discard()
                    whole.append(index)
                elif state.restart:
                    log(f"{state.name}: the resumed partial file did not hash to its pin; fetching it again from "
                        "byte 0")
                    state.discard()
                    again.append((index, _RangedFile(state.name, state.url, state.sha256, state.size,
                                                     state.target, state.chunk)))
            ranged = again
        for index in wanted:
            placed[index] = Path(rows[index]["target"])
        size = sum(rows[index]["bytes"] or Path(rows[index]["target"]).stat().st_size for index in wanted) / 1e6
        moved = size - resumed / 1e6
        seconds = time.perf_counter() - started
        log(f"fetched {len(wanted)} pinned files ({size:,.0f} MB) in {seconds:.1f} s "
            f"({moved / max(seconds, 1e-9):,.1f} MB/s over {jobs} workers"
            + (f"; {resumed / 1e6:,.0f} MB resumed from an earlier fetch" if resumed else "") + "), each SHA-256 checked")
        return placed
    finally:
        pool.shutdown(wait=True, cancel_futures=True)


# ---------------------------------------------------------------- device maths

def _vendored_maths() -> dict[str, str]:
    """The vendored maths sources, each checked against SOURCES.json."""
    root = _data_root("mpasphys-maths")
    manifest = json.loads((root / "SOURCES.json").read_text(encoding="utf-8"))
    texts = {}
    for name, row in manifest["files"].items():
        path = root / name
        if not path.is_file():
            raise PhysicsResourceError(f"vendored maths source {name} is missing from this install")
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != row["sha256"]:
            raise PhysicsResourceError(f"vendored maths source {name} differs from SOURCES.json")
        texts[name] = raw.decode("utf-8")
    return texts


def _cuda_core_math(source: str, name: str) -> str:
    """CORE-MATH C as NVRTC device code; the bytes of the bl_mynn lane's assembler."""
    text = re.sub(r'^#include[^\n]*\n', '', source, flags=re.M)
    text = re.sub(r'^#pragma[^\n]*\n', '', text, flags=re.M)
    text = re.sub(r'^\s*feraiseexcept\([^;]*;\s*', '\n', text, flags=re.M)
    text = text.replace('static __attribute__((noinline)) float as_special',
                        '__device__ __noinline__ float as_special')
    text = text.replace('float cr_' + name + '(', '__device__ __noinline__ float cr_' + name + '(')
    text = re.sub(r'(b(?:32u32|64u64)_u)\s+(\w+)\s*=\s*\{\.(f|u)\s*=\s*([^;]*)\};',
                  r'\1 \2; \2.\3 = \4;', text)
    replacements = {
        '__builtin_expect': 'p12_expect', '__builtin_fabsf': 'fabsf',
        '__builtin_copysignf': 'copysignf', '__builtin_copysign': 'copysign',
        '__builtin_floor': 'floor', '__builtin_clz': '__clz',
        '__builtin_fmaf': '__fmaf_rn',
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return 'namespace p12_core_' + name + ' {\n' + text + '\n}\n'


_MYNN_TAIL = '''
#undef p12_expect
__device__ float mynn_pow(float a,float b) { return gfk_pow(a,b); }
__device__ double mynn_pow(double a,double b) { return glibc_pow(a,b); }
__device__ double mynn_pow(double a,float b) { return glibc_pow(a,(double)b); }
__device__ double mynn_pow(float a,double b) { return glibc_pow((double)a,b); }
__device__ float mynn_exp(float a) { return gfk_exp(a); }
__device__ double mynn_exp(double a) { return glibc_exp(a); }
__device__ float mynn_log(float a) { return gfk_log(a); }
__device__ double mynn_log(double a) { return glibc_log(a); }
__device__ float mynn_log10(float a) { return p12_core_log10f::cr_log10f(a); }
__device__ float mynn_atan(float a) { return glibc_atanf(a); }
__device__ float mynn_tanh(float a) { return p12_core_tanhf::cr_tanhf(a); }
__device__ float mynn_erf(float a) { return p12_core_erff::cr_erff(a); }
// No narrowing fallback for a binary64 call not covered by this resource.
__device__ double mynn_log10(double) = delete;
__device__ double mynn_atan(double) = delete;
__device__ double mynn_tanh(double) = delete;
__device__ double mynn_erf(double) = delete;
'''


def mynn_maths(texts: dict[str, str]) -> str:
    """bl_mynn shared device maths (tests/mpas_mynn_v842/assemble_shared_math.py, no FMA blocks)."""
    pieces = ['// Explicit whole-MYNN replay resource; upstream source hashes in adjacent receipt.\n']
    pieces += [texts[name] for name in ('glibc_flt64.cuh', 'glibc_flt32.cuh', 'glibc_trig_flt32.cuh')]
    pieces.append('#define p12_expect(value, expected) (value)\n')
    pieces += [_cuda_core_math(texts[f'core-math/{name}.c'], name) for name in ('log10f', 'tanhf', 'erff')]
    pieces.append(_MYNN_TAIL)
    return '\n'.join(pieces)


def thompson_maths(texts: dict[str, str]) -> str:
    """mp_thompson shared device maths with glibc 2.43's CORE-MATH log10f.

    tests/thompson_v842_oracle/replay.py engine_math_source(native_log10=True):
    the engine's float32, binary64 and Thompson libm headers, the mpas_* maths
    aliases, and the CORE-MATH log10f block glibc 2.43 carries.
    """
    source = '\n'.join(texts[name] for name in ('glibc_flt32.cuh', 'glibc_flt64.cuh', 'thompson_aerosol_libm.cuh'))
    for name in ('expf', 'exp', 'logf', 'log', 'log10f', 'log10', 'powf', 'pow'):
        source += f'\n#define mpas_{name} thompson_aa_{name}\n'
    for name in ('atan', 'atanf', 'atan2', 'hypot'):
        source += f'\n#define mpas_{name} {name}\n'
    source += '\n' + texts['ubm_log10f.cuh']
    source += '\n#undef mpas_log10f\n#define mpas_log10f ubm_log10f\n'
    return source


def tempo_maths(texts: dict[str, str]) -> str:
    """RRFS TEMPO shared device maths: the Thompson aerosol libm aliases with
    hypot from that header, glibc 2.43's CORE-MATH log10f, then glibc cosf."""
    source = '\n'.join(texts[name] for name in ('glibc_flt32.cuh', 'glibc_flt64.cuh', 'thompson_aerosol_libm.cuh'))
    for name in ('expf', 'exp', 'logf', 'log', 'log10f', 'log10', 'powf', 'pow'):
        source += f'\n#define mpas_{name} thompson_aa_{name}\n'
    for name in ('atan', 'atanf', 'atan2'):
        source += f'\n#define mpas_{name} {name}\n'
    source += '\n#define mpas_hypot thompson_aa_hypot\n'
    source += '\n' + texts['ubm_log10f.cuh'] + '\n#undef mpas_log10f\n#define mpas_log10f ubm_log10f\n'
    source += texts['glibc_trig_flt32.cuh'] + '\n#define mpas_cosf glibc_cosf\n'
    source += '\n#define mpas_tgammaf tgammaf\n#define mpas_tgamma tgamma\n#define mpas_sin sin\n#define mpas_cos cos\n'
    return source


_ASSEMBLERS = {"bl_mynn": mynn_maths, "mp_thompson": thompson_maths, "mp_tempo": tempo_maths}


def build_maths(cache: Path) -> dict[str, dict[str, str]]:
    texts = _vendored_maths()
    out = {}
    for scheme, (filename, expected) in MATHS.items():
        text = _ASSEMBLERS[scheme](texts)
        actual = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if actual != expected:
            raise PhysicsResourceError(f"{scheme} device maths assembled to sha256 {actual}, not the certified "
                                       f"{expected}; the vendored sources or the recipe changed")
        target = cache / "maths" / expected / filename
        if not (target.is_file() and sha256_file(target) == expected):
            target.parent.mkdir(parents=True, exist_ok=True)
            handle, temporary = tempfile.mkstemp(dir=target.parent, prefix=".partial-")
            with os.fdopen(handle, "wb") as stream:
                stream.write(text.encode("utf-8"))
            _replace_into(Path(temporary), target)
        out[scheme] = {"path": str(target), "sha256": expected}
    return out


# ---------------------------------------------------------------- decoders

def decoder_filename(name: str) -> str:
    if name == "lookup_decode":
        return {"win32": "lookup_decode.dll", "darwin": "liblookup_decode.dylib"}.get(
            sys.platform, "liblookup_decode.so")
    return name + (".exe" if sys.platform == "win32" else "")


def _source_crate() -> Path | None:
    crate = Path(__file__).resolve().parents[3] / "rust" / "physics-tables" / "Cargo.toml"
    return crate if crate.is_file() else None


def locate_decoders(cache: Path, log: Callable[[str], None] = _log) -> dict[str, Path]:
    """The wheel's decoders (woof/hex/libexec), else built from a source checkout's crate."""
    from woof.hex.helpers import path as helper_path
    found = {name: helper_path(decoder_filename(name)) for name in DECODERS}
    libexec = next(iter(found.values())).parent
    if all(path.is_file() for path in found.values()):
        return found
    crate = _source_crate()
    cargo = shutil.which("cargo")
    if crate is None or cargo is None:
        missing = sorted(name for name, path in found.items() if not path.is_file())
        raise PhysicsResourceError(
            f"this install has no MPAS table decoders ({', '.join(missing)} absent from {libexec}); "
            "reinstall the woof hex wheel, or build from a source checkout with cargo on PATH")
    target = cache / "cargo-target"
    log(f"building the table decoders from {crate.parent} with cargo")
    subprocess.run([cargo, "build", "--release", "--locked", "--manifest-path", str(crate),
                    "--target-dir", str(target)], check=True)
    return {name: target / "release" / decoder_filename(name) for name in DECODERS}


def stage_decoders(cache: Path, log: Callable[[str], None] = _log) -> dict[str, dict[str, str]]:
    """Copy each decoder under its own SHA-256, so a reinstall never moves a pinned byte."""
    out = {}
    for name, path in locate_decoders(cache, log).items():
        digest = sha256_file(path)
        target = cache / "decoders" / digest / decoder_filename(name)
        if not (target.is_file() and sha256_file(target) == digest):
            _copy_verified(path, target, digest)
            target.chmod(0o755)
        out[name] = {"path": str(target), "sha256": digest}
    return out


# ---------------------------------------------------------------- Thompson strict tables

def _toolchain(fortran: str | None) -> dict[str, str]:
    version = ""
    if fortran:
        try:
            version = subprocess.run([fortran, "--version"], capture_output=True, text=True,
                                     check=False).stdout.splitlines()[0]
        except (OSError, IndexError):
            version = ""
    return {"fortran": fortran or "", "fortran_version": version,
            "libc": " ".join(platform.libc_ver()), "machine": platform.machine()}


def _stage_pinned_log10f(build: Path) -> Path:
    """The vendored CORE-MATH log10f source copied into ``build``, refused unless it is the pinned file."""
    root = _data_root("mpasphys-maths")
    sha = json.loads((root / "SOURCES.json").read_text(encoding="utf-8"))["files"][PINNED_LOG10F]["sha256"]
    source = root / PINNED_LOG10F
    if not source.is_file() or sha256_file(source) != sha:
        raise PhysicsResourceError(f"vendored maths source {PINNED_LOG10F} is missing from this install or "
                                   "differs from SOURCES.json; reinstall the woof hex wheel")
    return _copy_verified(source, build / "cr_log10f.c", sha)


def _stage_pinned_host_maths(build: Path) -> None:
    # The same pinned maths already used by the CUDA Thompson port. The
    # manylinux 2.28 libm changes all five generated outputs, not just log10f.
    texts = _vendored_maths()
    for name in ("cuda_host_shim.h", "thompson_host_math.cpp", "glibc_flt32.cuh",
                 "glibc_flt64.cuh", "thompson_aerosol_libm.cuh"):
        (build / name).write_text(texts[name], encoding="utf-8")


def _thompson_referee():
    from .native_library_build import NativeLibraryError, referee_builder
    try:
        return referee_builder("thompson_tables")
    except NativeLibraryError as error:
        raise PhysicsResourceError(str(error)) from error


def thompson_generator_commands(fortran: str) -> list[list[str]]:
    """Explicit referee command inspection requires a source checkout."""
    return _thompson_referee().thompson_generator_commands(fortran)


def _thompson_expected() -> dict[str, str]:
    from .table_resources_v842 import THOMPSON_AUXILIARY_SHA256
    from .thompson_v842 import TABLE_FILE_SHA256
    return {**TABLE_FILE_SHA256, "auxiliary.bin": THOMPSON_AUXILIARY_SHA256}


def build_thompson_strict(cache: Path, *, mpas_source: Path | None = None,
                          thompson_tables: Path | None = None, offline: bool = False,
                          log: Callable[[str], None] = _log) -> Path:
    """Build strict tables only through the source-checkout referee tool."""
    expected = _thompson_expected()
    final = cache / "thompson-strict" / "tables"
    if all((final / name).is_file() and sha256_file(final / name) == sha for name, sha in expected.items()):
        return final
    if thompson_tables is not None:
        for name, sha in expected.items():
            source = Path(thompson_tables) / name
            if name == "auxiliary.bin" and not source.is_file():
                continue
            if not source.is_file() or sha256_file(source) != sha:
                raise PhysicsResourceError(f"--thompson-tables {thompson_tables}: {name} is absent or is not the "
                                           f"strict native reference's bytes (sha256 {sha})")
            if not ((final / name).is_file() and sha256_file(final / name) == sha):
                _copy_verified(source, final / name, sha)
        if (final / "auxiliary.bin").is_file():
            return final
    return _thompson_referee().build_thompson_strict(
        cache, mpas_source=mpas_source, thompson_tables=thompson_tables,
        offline=offline, log=log)


def build_thompson_aero_auxiliary(cache: Path, tables: Path, ccn_activate: Path,
                                  *, mpas_source: Path | None = None, offline: bool = False,
                                  log: Callable[[str], None] = _log) -> Path:
    """Build aerosol arrays only through the source-checkout referee tool."""
    from .thompson_aero_v842 import AUXILIARY_SHA256
    final = cache / "thompson-strict" / "aerosol" / "auxiliary.bin"
    if final.is_file() and sha256_file(final) == AUXILIARY_SHA256:
        return final
    return _thompson_referee().build_thompson_aero_auxiliary(
        cache, tables, ccn_activate, mpas_source=mpas_source, offline=offline, log=log)


def _mkdir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


# ---------------------------------------------------------------- self-check

def decoded_digests(packet: dict[str, Any]) -> dict[str, str]:
    """SHA-256 of what each packet decoder decodes its pinned table to.

    The words, not the decoder binary, are what a slot consumes: a decoder
    rebuilt by another compiler is a different file and must still decode
    every table to the certified decoder's words.
    """
    out = {}
    if "mp_thompson" in packet:
        import numpy as np
        from . import thompson_v842 as scheme
        from .table_resources_v842 import RustLookupRecords, certified_path
        row = packet["mp_thompson"]
        decoder = RustLookupRecords(row["rust_decoder"])
        digest = hashlib.sha256()
        groups = (("MP_THOMPSON_QRacrQG_DATA.DBL", scheme.TABLE_ARGUMENTS[:6]),
                  ("MP_THOMPSON_QRacrQS_DATA.DBL", scheme.TABLE_ARGUMENTS[6:18]),
                  ("MP_THOMPSON_freezeH2O_DATA.DBL", scheme.TABLE_ARGUMENTS[18:24]),
                  ("MP_THOMPSON_QIautQS_DATA.DBL", scheme.TABLE_ARGUMENTS[24:27]))
        for filename, names in groups:
            path = Path(row["table_directory"]) / filename
            arrays = decoder.arrays(path, names, scheme.TABLE_SHAPES, records=True, little=False)
            for name in names:
                digest.update(name.encode() + b"\0" + np.asarray(arrays[name]).tobytes(order="F"))
        auxiliary = ("t_efrw", "t_efsw", "tnr_rev", "tpc_wev", "tnc_wev", "tnccn_act")
        arrays = decoder.arrays(Path(row["auxiliary"]), auxiliary, scheme.TABLE_SHAPES, records=False, little=True)
        for name in auxiliary:
            digest.update(name.encode() + b"\0" + np.asarray(arrays[name]).tobytes(order="F"))
        out["mp_thompson"] = digest.hexdigest()
    if "rrtmg_lw" in packet:
        row = packet["rrtmg_lw"]
        output = subprocess.run([row["decoder"]["path"], row["table"]["path"]], check=True,
                                stdout=subprocess.PIPE).stdout
        out["rrtmg_lw"] = hashlib.sha256(output).hexdigest()
    if "rrtmg_sw" in packet:
        row = packet["rrtmg_sw"]
        with tempfile.TemporaryDirectory() as scratch:
            directory = Path(scratch) / "decoded"
            subprocess.run([row["decoder"]["path"], row["data"]["path"], str(directory)], check=True)
            digest = hashlib.sha256()
            for path in sorted(directory.iterdir()):
                digest.update(path.name.encode() + b"\0" + path.read_bytes())
        out["rrtmg_sw"] = digest.hexdigest()
    if "sf_noah" in packet:
        row = packet["sf_noah"]
        output = subprocess.run([row["rust_decoder"]["path"], row["table_directory"], row["landuse"],
                                 row["soil_type"]], check=True, stdout=subprocess.PIPE).stdout
        out["sf_noah:" + row["landuse"]] = hashlib.sha256(output).hexdigest()
    return out


# ---------------------------------------------------------------- the MPAS route's own resources
#
# The default framework route binds the packet and wheel's Registry pool reader.
# Native libraries and source trees are explicit local referee-build resources.
# Verification decks can name every installed resource as {resource:KEY}.

POOL_READER = "rw_mpas_physics_pool"
POOL_READER_ABI = "gpuwm-rw-mpas-physics-pool-v1"
INSTALLED_SCHEMA = "gpuwm-hex.installed-physics-resources/v1"
RESOURCE_KEYS = ("physics-resources", "thompson-tables", "pool-reader", "mpas-source", "native-library",
                 "mpas-source-rrfs", "native-library-rrfs")
# The keys each pin set's route binds (woof.hex.mpasphys.forecast_v842.run_forecast).
from .native_library_build import PIN_SETS as NATIVE_PIN_SETS
PIN_SETS = tuple(NATIVE_PIN_SETS)
RRFS_PIN_SETS = tuple(name for name, row in NATIVE_PIN_SETS.items() if row["model"] == "rrfs-model")
ROUTE_KEYS = {"v8.4.2": {"native-library": "native-library", "mpas-source": "mpas-source"},
              **{name: {"native-library": "native-library-" + name, "mpas-source": "mpas-source-" + name}
                 for name in RRFS_PIN_SETS}}
RESOURCE_KEYS = tuple(dict.fromkeys((*RESOURCE_KEYS, *(key for row in ROUTE_KEYS.values() for key in row.values()))))
# Registry.xml and the files it #includes (pool_v842.registry_text).
REGISTRY_PREFIXES = ("src/core_atmosphere/Registry.xml", "src/core_atmosphere/diagnostics/Registry",
                     "src/core_atmosphere/physics/Registry_noahmp.xml")


def _pool_reader_crate() -> Path | None:
    crate = Path(__file__).resolve().parents[3] / "rust" / "physics-pool" / "Cargo.toml"
    return crate if crate.is_file() else None


def wheel_pool_reader() -> Path:
    """Where this install carries rw_mpas_physics_pool: the wheel, else the release's hex helper archive (the file
    may be absent)."""
    from woof.hex.helpers import path as helper_path
    return helper_path(POOL_READER + (".exe" if sys.platform == "win32" else ""))


def stage_pool_reader(cache: Path, log: Callable[[str], None] = _log) -> dict[str, str]:
    """The wheel's rw_mpas_physics_pool (else built from a checkout's crate), copied under its SHA-256."""
    path = wheel_pool_reader()
    name = path.name
    if not path.is_file():
        crate, cargo = _pool_reader_crate(), shutil.which("cargo")
        if crate is None or cargo is None:
            raise PhysicsResourceError(
                f"this install has no MPAS Registry pool reader ({name} is absent from {path.parent}: the wheel "
                "was built without the netCDF-C development files); reinstall the manylinux woof hex wheel, "
                "or pass --native-pool-reader")
        log(f"building the Registry pool reader from {crate.parent} with cargo")
        subprocess.run([cargo, "build", "--release", "--locked", "--manifest-path", str(crate),
                        "--target-dir", str(cache / "cargo-target")], check=True)
        path = cache / "cargo-target" / "release" / name
    abi = subprocess.run([str(path), "--abi"], capture_output=True, text=True, check=False).stdout.strip()
    if abi != POOL_READER_ABI:
        raise PhysicsResourceError(f"{path} answers ABI {abi!r}, not {POOL_READER_ABI}")
    digest = sha256_file(path)
    target = cache / "pool-reader" / digest / name
    if not (target.is_file() and sha256_file(target) == digest):
        _copy_verified(path, target, digest)
        target.chmod(0o755)
    return {"path": str(target), "sha256": digest}


def _source_manifest() -> dict[str, Any]:
    from .native_library_build import source_manifest
    return source_manifest()


def assemble_mpas_source(cache: Path, *, mpas_source: Path | None = None, offline: bool = False,
                         log: Callable[[str], None] = _log) -> dict[str, str]:
    """The pinned MPAS-Model tree the route reads at run time, from fetched files.

    Its Registry (with every #include) is what pool_v842 decodes the pools from;
    physics_wrf/files holds the MPAS-Data tables, which runtime_v842 links into
    the run directory as an MPAS build places them.  Every file is checked by
    SHA-256 (mpas-source-v842.json).
    """
    manifest = _source_manifest()
    root = cache / "mpas-source" / MPAS_MODEL_COMMIT[:12]
    tables = cache / "mpas-data" / MPAS_DATA_COMMIT[:12]
    for relative, row in manifest["files"].items():
        if relative.startswith(REGISTRY_PREFIXES):
            fetch(relative, row["origin"], row["path"], row["sha256"], root / relative,
                  mpas_source=mpas_source, offline=offline)
    files = _mkdir(root / _PHYSICS_FILES)
    for name, row in manifest["tables"].items():
        source = fetch(name, row["origin"], row["path"], row["sha256"], tables / name,
                       mpas_source=mpas_source, offline=offline).resolve()
        _link_relative(files / name, source)
    digest = hashlib.sha256(json.dumps(
        {"files": {k: v["sha256"] for k, v in manifest["files"].items() if k.startswith(REGISTRY_PREFIXES)},
         "tables": {k: v["sha256"] for k, v in manifest["tables"].items()}},
        sort_keys=True).encode()).hexdigest()
    return {"path": str(root), "sha256": digest}


def assemble_rrfs_source(cache: Path, *, rrfs_source: Path | None = None, offline: bool = False,
                         log: Callable[[str], None] = _log, jobs: int | None = None,
                         pin_set: str = "rrfs") -> dict[str, str]:
    """The RRFS fork tree the RRFS route reads at run time: its Registry and its physics tables.

    The Registry (with every #include) is what the pool reader decodes the fork's pools from; physics_wrf/files
    holds every table the fork's physics open by name: MPAS-Data v8.2, the fork's ugwp_limb_tau.nc, and the
    TEMPO v3 lookup tables and ccn_activate.bin from NOAA's published RRFS fix files (mpas-source-rrfs.json).
    Every file is placed by fetch_pinned: concurrently, the large TEMPO tables as resumable byte ranges, each
    published only when it hashes to its pinned SHA-256.
    """
    from ..file_lock import held
    from .native_library_build import source_manifest
    manifest = source_manifest(pin_set)
    root = cache / "rrfs-source" / manifest["pins"]["rrfs-model"][:12]
    tables = cache / "rrfs-tables"
    # A Registry file the route never reads is never fetched: the manifest's run_time_skip names each such
    # path prefix and why (44c0849's stochastic_physics/Registry_stoch_physics.xml is LGPL-3.0 and is
    # included only under #ifdef STOCHASTIC_PHYSICS, which pool_v842.registry_text never defines).
    skip = tuple(manifest.get("run_time_skip", {}))
    registry = {relative: row for relative, row in manifest["files"].items()
                if re.fullmatch(r"src/core_atmosphere/(?:[^/]+/)*[Rr]egistry[^/]*\.xml", relative)
                and not relative.startswith(skip)}
    rows = [{"name": relative, "origin": row["origin"], "path": row["path"], "sha256": row["sha256"],
             "bytes": row.get("bytes"), "target": root / relative} for relative, row in registry.items()]
    rows += [{"name": name, "origin": row["origin"], "path": row["path"], "sha256": row["sha256"],
              "bytes": row.get("bytes"), "target": tables / row["sha256"] / name}
             for name, row in manifest["tables"].items()]
    files = _mkdir(root / _PHYSICS_FILES)
    pending = [name for name, row in manifest["tables"].items()
               if not ((tables / row["sha256"] / name).is_file())]
    if pending:
        size = sum(manifest["tables"][name]["bytes"] for name in pending) / 1e6
        log(f"fetching {len(pending)} RRFS physics tables ({size:,.0f} MB, NOAA's published RRFS fix files and "
            f"MPAS-Data) into {tables}")
    # One fetch per cache at a time: the ranged partial files have fixed names so a later fetch can resume them.
    with held(tables / ".fetch.lock"):
        placed = fetch_pinned(rows, mpas_source=rrfs_source, offline=offline, jobs=jobs, log=log,
                              origins=manifest.get("origins"))
    for name, source in zip(manifest["tables"], placed[len(registry):]):
        _link_relative(files / name, Path(source).resolve())
    digest = hashlib.sha256(json.dumps(
        {"files": {k: v["sha256"] for k, v in registry.items()},
         "tables": {k: v["sha256"] for k, v in manifest["tables"].items()}}, sort_keys=True).encode()).hexdigest()
    return {"path": str(root), "sha256": digest}


def installed_record_path(cache: Path | None = None) -> Path:
    return (Path(cache) if cache is not None else default_cache()) / "installed.json"


def read_installed(cache: Path | None = None) -> dict[str, Any]:
    path = installed_record_path(cache)
    if not path.is_file():
        return {"schema": INSTALLED_SCHEMA, "resources": {}}
    record = json.loads(path.read_text(encoding="utf-8"))
    if record.get("schema") != INSTALLED_SCHEMA:
        raise PhysicsResourceError(f"{path} is not a {INSTALLED_SCHEMA} record")
    return record


def record_installed(cache: Path, rows: dict[str, dict[str, str]]) -> Path:
    """Merge ``rows`` into the cache's installed-resources record (atomically)."""
    record = read_installed(cache)
    current = (Path(cache) if cache is not None else default_cache()).resolve()
    recorded = _recorded_root(record)
    if recorded is not None and recorded != current:
        # A record copied with its cache: rows written before the copy move with it.
        for row in record["resources"].values():
            if isinstance(row, dict) and row.get("path"):
                row["path"] = str(_rebased(Path(row["path"]), recorded, current))
                for architectures in row.get("variants", {}).values():
                    for variant in architectures.values():
                        variant["path"] = str(_rebased(Path(variant["path"]), recorded, current))
    record["resources"].update(rows)
    record["cache_root"] = str(current)
    path = installed_record_path(cache)
    handle, temporary = tempfile.mkstemp(dir=_mkdir(path.parent), prefix=".partial-")
    with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(record, indent=2, sort_keys=True) + "\n")
    _replace_into(Path(temporary), path)
    return path


def resolve_resource(key: str, cache: Path | None = None) -> Path:
    """The installed path behind a portable resource key, checked against its recorded SHA-256."""
    if key not in RESOURCE_KEYS:
        raise PhysicsResourceError(f"unknown physics resource key {key!r}; the keys are {', '.join(RESOURCE_KEYS)}")
    record = read_installed(cache)
    row = record["resources"].get(key)
    where = installed_record_path(cache)
    if row is None:
        raise PhysicsResourceError(f"physics resource {key!r} is not installed ({where}); run "
                                   "`woof hex physics-resources fetch` or explicitly build from source")
    path = _rebased(Path(row["path"]), _recorded_root(record), where.parent.resolve())
    if path.is_file():
        if sha256_file(path) != row["sha256"]:
            raise PhysicsResourceError(f"physics resource {key!r} at {path} changed since installation; run "
                                       "`woof hex physics-resources fetch` or explicitly build from source")
    elif not path.is_dir():
        raise PhysicsResourceError(f"physics resource {key!r} at {path} is gone; run "
                                   "`woof hex physics-resources fetch` or explicitly build from source")
    return path


def assemble_framework_resources(directory: Path, *, log: Callable[[str], None] = _log) -> dict:
    """Bind verified DATA to this wheel's maths, decoders and Registry reader."""
    receipt = build_packet(directory, offline=True, include_mynnedmf=False, data_only=True, log=log)
    return {'physics-resources': {'path': receipt['packet'], 'sha256': receipt['packet_sha256']},
            'pool-reader': stage_pool_reader(directory, log)}


def source_tables_missing(path: Path, pin_set: str = "v8.4.2") -> list[str]:
    """The pinned physics tables an installed MPAS or RRFS source tree links that are gone (or the wrong size).

    THE BREAKAGE THIS PREVENTS (hrrrv5 anywhere, RTX 5090, 2026-10-30): a cleaned cache had lost the three big TEMPO v3
    tables its rrfs-source tree links, while installed.json still recorded the tree as installed; the route trusted
    the record, never fetched again, and the forecast died 12 s in on
    ``[Errno 2] No such file or directory: .../physics_wrf/files/freeze_water_data_tempo_v3``. The size check reads
    no file contents, so it costs a stat per table on every run; the refetch hashes what it places.
    """
    from .native_library_build import source_manifest
    try:
        manifest = source_manifest(pin_set)
    except Exception:  # an unknown pin set is refused where its route is resolved, not here
        return []
    files = Path(path) / _PHYSICS_FILES
    gone = []
    for name, row in manifest.get("tables", {}).items():
        try:
            size = (files / name).stat().st_size  # follows the link into the table store
        except OSError:
            gone.append(name)
            continue
        if row.get("bytes") and size != int(row["bytes"]):
            gone.append(name)
    return gone


def ensure_route_resources(keys: tuple[str, ...], *, cache: Path | None = None,
                           log: Callable[[str], None] = _log, pin_set: str = "v8.4.2") -> dict[str, Path]:
    """Resolve installed keys, fetching the pinned release if any are missing."""
    out, missing = {}, []
    if 'pool-reader' in keys and wheel_pool_reader().is_file():
        # THE BREAKAGE THIS PREVENTS (U-H3): the installed record kept the reader copied when the cache was
        # first filled (a first run or `physics-resources fetch`), so upgrading the wheel left every forecast on
        # the old binary; hex 0.4.8's reader ran the whole pool transport on one thread.  The reader is a pure
        # transport of this wheel's request format, so the route always takes the one this wheel carries.
        root = Path(cache if cache is not None else default_cache()).resolve()
        staged = stage_pool_reader(root, log)
        if read_installed(root)["resources"].get('pool-reader', {}).get('sha256') != staged['sha256']:
            record_installed(root, {'pool-reader': staged})
    if keys == ('pool-reader',):
        try:
            return {'pool-reader': resolve_resource('pool-reader', cache)}
        except PhysicsResourceError:
            root = Path(cache if cache is not None else default_cache()).resolve()
            record_installed(root, {'pool-reader': stage_pool_reader(root, log)})
            return {'pool-reader': resolve_resource('pool-reader', root)}
    broken = set()
    for key in keys:
        try:
            out[key] = resolve_resource(key, cache)
        except PhysicsResourceError:
            missing.append(key)
            continue
        if key.startswith('mpas-source'):
            gone = source_tables_missing(out[key], pin_set)
            if gone:
                log(f"physics resource {key}: {len(gone)} pinned table file(s) it links are gone from the cache "
                    f"({', '.join(gone[:3])}{', ...' if len(gone) > 3 else ''}); fetching them again")
                del out[key]
                missing.append(key)
                broken.add(key)
    if not missing:
        return out
    if 'mpas-source' in broken and not pin_set.startswith('rrfs'):
        # The v8.4.2 tree is installed and only its MPAS-Data tables went: place them again by SHA-256.
        root = Path(cache if cache is not None else default_cache()).resolve()
        record_installed(root, {'mpas-source': assemble_mpas_source(root, log=log)})
        out['mpas-source'] = resolve_resource('mpas-source', root)
        missing.remove('mpas-source')
        if not missing:
            return out
    source_key = 'mpas-source-' + pin_set
    if pin_set.startswith('rrfs') and source_key in missing:
        # The fork's DATA (Registry and tables, fetched by SHA-256): no compiler, no native library.
        root = Path(cache if cache is not None else default_cache()).resolve()
        record_installed(root, {source_key: assemble_rrfs_source(root, log=log, pin_set=pin_set)})
        out[source_key] = resolve_resource(source_key, root)
        missing.remove(source_key)
        if not missing:
            return out
    if set(missing) & {'native-library', 'mpas-source', 'native-library-rrfs'}:
        raise PhysicsResourceError('native referee resources require an explicit compiler-requiring '
                                   f'`woof hex physics-resources build --pin-set {pin_set}`, or pass '
                                   '--native-physics-library, --native-pool-reader and --mpas-source '
                                   'from your site build')
    if os.environ.get("WOOF_HEX_PHYSICS_RESOURCES_FIRST_USE", "") == "off":
        raise PhysicsResourceError(
            "the MPAS physics route needs " + ", ".join(missing) + " and first-use fetching is off "
            "(WOOF_HEX_PHYSICS_RESOURCES_FIRST_USE=off); run `woof hex physics-resources fetch` "
            "(or an explicit `woof hex physics-resources build`), or pass "
            + ", ".join({"pool-reader": "--native-pool-reader", "mpas-source": "--mpas-source",
                         "native-library": "--native-physics-library"}.get(k, k) for k in missing))
    log("the MPAS physics route needs " + ", ".join(missing) + ": fetching pinned release resources "
        "(no host compiler needed)")
    from .resources_fetch import fetch
    fetch(cache, pin_sets=(pin_set,), log=log)
    return {key: resolve_resource(key, cache) for key in keys}


# ---------------------------------------------------------------- packet

def build(cache: Path | None = None, *, mpas_source: Path | None = None, thompson_tables: Path | None = None,
          landuse: str = DEFAULT_LANDUSE, offline: bool = False, native: bool = True,
          native_library: Path | None = None, mpich: Path | None = None, jobs: int | None = None,
          licence_opt_in: dict[str, str] | None = None, log: Callable[[str], None] = _log,
          pin_sets: tuple[str, ...] = ("v8.4.2",), rrfs_source: Path | None = None,
          native_pin_sets: tuple[str, ...] | None = None) -> dict[str, Any]:
    """Build (or reuse) every resource, write the packet and the installed record; return the receipt.

    ``pin_sets`` names the native pin sets to build: ``v8.4.2`` (the hex-cuda packet, the MPAS-Model v8.4.2
    tree and library) and ``rrfs`` (the RRFS fork's tree and library); the CLI builds both by default.

    ``native`` delegates to the source checkout's proof builder under
    tools/native_dry_reference; installed forecasts use prebuilt resources.
    ``native_library`` adopts an existing build instead.  ``native_pin_sets`` (None: every built pin set)
    limits which pin sets ``native`` compiles a library for.  ``licence_opt_in`` maps each licence term the
    caller opts in to (woof.hex.licence_gate) to what named it; without ``ucar-noahmp``, the native library
    compiles the licence stub for Noah-MP v5.0 (the default), so it contains none of it.  Terms that name
    no code the library compiles are not the library's concern and are dropped here.
    """
    native_terms = set(_native_licence_terms())
    licence_opt_in = {terms: named_by for terms, named_by in (licence_opt_in or {}).items() if terms in native_terms}
    for terms, named_by in sorted(licence_opt_in.items()):
        log(f"LICENCE OPT-IN {terms} ({named_by}): the native framework library is built with the code under "
            f"those terms, NOT Apache-2.0 (woof.hex.licence_gate)")
    unknown = sorted(set(pin_sets) - set(PIN_SETS))
    if unknown or not pin_sets:
        raise PhysicsResourceError(f"unknown pin set {', '.join(unknown) or '(none)'}; the pin sets are "
                                   + ", ".join(PIN_SETS))
    native_sets = set(pin_sets if native_pin_sets is None else native_pin_sets) if native else set()
    rows = {}
    if "v8.4.2" in pin_sets:
        receipt = build_packet(cache, mpas_source=mpas_source, thompson_tables=thompson_tables, landuse=landuse,
                               offline=offline, log=log)
        cache = Path(receipt["cache"])
        rows = {"physics-resources": {"path": receipt["packet"], "sha256": receipt["packet_sha256"]},
                "thompson-tables": {"path": receipt["thompson_tables"], "sha256": receipt["thompson_tables_sha256"]}}
    else:
        cache = _mkdir(Path(cache) if cache is not None else default_cache()).resolve()
        receipt = {"schema": SCHEMA, "cache": str(cache)}
    receipt["pin_sets"] = list(pin_sets)
    rows["pool-reader"] = stage_pool_reader(cache, log)
    from . import native_library_build as nlb
    from functools import partial
    for selected in (name for name in pin_sets if name in RRFS_PIN_SETS):
        keys = ROUTE_KEYS[selected]
        rows[keys["mpas-source"]] = assemble_rrfs_source(cache, rrfs_source=rrfs_source, offline=offline, log=log, pin_set=selected)
        if selected in native_sets:
            try:
                rows[keys["native-library"]] = nlb.build_native_library(
                    cache, fetch=fetch_url, fetch_file=partial(fetch, origins=nlb.source_manifest(selected).get("origins")),
                    mpas_source=rrfs_source, mpich=mpich, jobs=jobs, offline=offline,
                    licence_opt_in=sorted(licence_opt_in), log=log, pin_set=selected)
            except nlb.NativeLibraryError as error:
                raise PhysicsResourceError(str(error)) from error
            if licence_opt_in:
                rows[keys["native-library"]]["licence"]["named_by"] = {
                    terms: licence_opt_in[terms] for terms in rows[keys["native-library"]]["licence"].get("opt_in", [])}
        receipt[selected + "_native_library_licence"] = (rows[keys["native-library"]].get("licence")
                                                          if keys["native-library"] in rows else None)
    if "v8.4.2" not in pin_sets:
        receipt["installed"] = str(record_installed(cache, rows))
        receipt["resources"] = {key: row for key, row in read_installed(cache)["resources"].items()}
        receipt["native_library_toolchain"] = nlb.toolchain() if any(ROUTE_KEYS[name]["native-library"] in rows for name in RRFS_PIN_SETS) else None
        return receipt
    rows["mpas-source"] = assemble_mpas_source(cache, mpas_source=mpas_source, offline=offline, log=log)
    if native_library is not None:
        try:
            rows["native-library"] = nlb.stage_library(cache, Path(native_library))
        except nlb.NativeLibraryError as error:
            raise PhysicsResourceError(str(error)) from error
    elif "v8.4.2" in native_sets:
        try:
            rows["native-library"] = nlb.build_native_library(
                cache, fetch=fetch_url, fetch_file=fetch, mpas_source=mpas_source, mpich=mpich, jobs=jobs,
                offline=offline, licence_opt_in=sorted(licence_opt_in), log=log)
        except nlb.NativeLibraryError as error:
            raise PhysicsResourceError(str(error)) from error
    if "native-library" in rows and licence_opt_in:
        rows["native-library"]["licence"]["named_by"] = {terms: licence_opt_in[terms]
                                                         for terms in rows["native-library"]["licence"]
                                                         .get("opt_in", [])}
    receipt["installed"] = str(record_installed(cache, rows))
    receipt["resources"] = {key: row for key, row in read_installed(cache)["resources"].items()}
    receipt["native_library_toolchain"] = nlb.toolchain() if "native-library" in rows else None
    # The build flavour: stubbed (the default), opted-in or partly stubbed, with each stub's source SHA-256.
    receipt["native_library_licence"] = rows["native-library"].get("licence") if "native-library" in rows else None
    return receipt


def rrfs_mynnedmf_images() -> tuple[tuple[str, str], ...]:
    """(packet key, module) of every RRFS MYNN-EDMF image the route table names, one per pin.

    A pin is a table row (bl_mynnedmf cuda_entries: module, resources.packet_key), so a new pin's image is built
    and packed without a code path of its own. THE BREAKAGE THIS PREVENTS: with hrrrv5 defaulting to the
    rrfs-44c0849 row (MYNN-EDMF c6d08709), the kernel pack and the physics-resources packet carried only the
    1d6e370e image (c942418), so every first hrrrv5 run compiled its MYNN-EDMF image on the start path.
    """
    from collections.abc import Mapping
    from ..physics_suites import load_table
    found = {}
    for row in load_table().values():
        slots = row.get("slots") if isinstance(row, Mapping) else None
        for slot in (slots or {}).values():
            entry = ((slot.get("cuda_entries") or {}).get("bl_mynnedmf") if isinstance(slot, Mapping) else None)
            module = entry.get("module") if isinstance(entry, Mapping) else None
            if module and module.startswith("rrfs."):
                key = (entry.get("resources") or {}).get("packet_key", "bl_mynnedmf")
                found.setdefault(key, module)
    return tuple(sorted(found.items()))


def _rrfs_module_present(module: str) -> bool:
    import importlib.util
    return importlib.util.find_spec(f"{__package__}.rrfs.{module}") is not None


def build_mynnedmf_image(cache: Path, log: Callable[[str], None] = _log, *,
                        arch: str | None = None,
                        module: str = "rrfs.bl_mynnedmf_c942418") -> dict[str, str]:
    """The RRFS MYNN-EDMF CUDA image for this host's device, compiled once and cached.

    rrfs.bl_mynnedmf_c942418.load_cuda_packet accepts it only when its source,
    image and NVRTC options (strict flags plus this device's architecture)
    match the metadata written beside it (tests/rrfs_mynnedmf/prepare_cuda.py).
    """
    explicit_arch = arch is not None
    try:
        if arch is None:
            import cupy
            arch = "sm_" + cupy.cuda.Device().compute_capability
    except Exception as error:  # no CUDA device or no cupy in this process
        raise PhysicsResourceError(f"the RRFS MYNN-EDMF CUDA image compiles for the run's device and this "
                                   f"process has none ({type(error).__name__}); the forecast's first use "
                                   "builds it") from error
    from ..cuda_fp32 import CUDA_FTZ_HELPERS
    from ..mynn_glibc_math import mynn_glibc_source
    import importlib
    port = importlib.import_module("." + module, __package__)
    source = CUDA_FTZ_HELPERS + mynn_glibc_source() + port.ARRAY_SUPPORT + port.MODULE_CUDA
    options = list(_hex_numerics.options(port.CUDA_OPTIONS)) + ["--gpu-architecture=" + arch]
    source_sha = hashlib.sha256(source.encode()).hexdigest()
    # The options (numerics mode) are part of the entry's name: without them a strict run and a fast run
    # (WOOF_HEX_NUMERICS) wrote one directory, each overwriting the other's image on every switch.
    options_sha = hashlib.sha256(json.dumps(options).encode()).hexdigest()
    image = cache / "rrfs-mynnedmf" / f"{source_sha[:16]}-{arch}-{options_sha[:12]}" / "mynnedmf.cubin"
    metadata = image.with_suffix(".json")
    if image.is_file() and metadata.is_file():
        try:
            row = json.loads(metadata.read_text(encoding="utf-8"))
        except ValueError:
            row = {}
        if (row.get("source_sha256") == source_sha and row.get("options") == options
                and row.get("image_sha256") == sha256_file(image)):
            return {"image": str(image), "metadata": str(metadata)}
    log(f"loading the RRFS MYNN-EDMF CUDA image for {arch} (verified kernel pack, NVRTC fallback)")
    compile_options = _hex_numerics.options(port.CUDA_OPTIONS)
    if explicit_arch:
        # Release construction has no GPU. Reuse the existing AOT compiler,
        # whose explicit-target route avoids CuPy's per-device memoizer.
        import base64
        import lzma
        from ..cuda_backend.aot import Pack, direct_metadata, recipe, source_hash
        pack = Pack()
        entry, _ = pack.select("nvrtc.direct." + source_hash(source), source,
                                compile_options, arch, metadata=direct_metadata("rrfs_mynnedmf.cu"))
        cubin = None if entry is None else pack.image(entry)
        if cubin is None:
            from ..cuda_backend.aot_build import compile_entry
            row = recipe({"source":source,"options":list(compile_options),
                          "route":"direct-nvrtc","filename":"rrfs_mynnedmf.cu"})
            compiled = compile_entry(row, arch)
            cubin = lzma.decompress(base64.b64decode(compiled["cubin_xz_base64"]))
    else:
        from ..cuda_backend.aot import direct_image
        cubin = direct_image(source, compile_options, arch, filename="rrfs_mynnedmf.cu")
    _mkdir(image.parent)
    for path, data in ((image, cubin), (metadata, (json.dumps(
            {"source_sha256": source_sha, "image_sha256": hashlib.sha256(cubin).hexdigest(), "options": options},
            indent=2) + "\n").encode())):
        handle, temporary = tempfile.mkstemp(dir=path.parent, prefix=".partial-")
        with os.fdopen(handle, "wb") as stream:
            stream.write(data)
        _replace_into(Path(temporary), path)
    return {"image": str(image), "metadata": str(metadata)}


def build_packet(cache: Path | None = None, *, mpas_source: Path | None = None, thompson_tables: Path | None = None,
                 landuse: str = DEFAULT_LANDUSE, offline: bool = False,
                 include_mynnedmf: bool = True,
                 log: Callable[[str], None] = _log, image_arch: str | None = None, data_only: bool = False) -> dict[str, Any]:
    """Build the resource packet; omit the RRFS device image when its slot is unused.

    Explicit installation builds retain the complete packet by default. First-use
    registration knows the selected CUDA slots and can skip this device compiler.
    """
    cache = _mkdir(Path(cache) if cache is not None else default_cache()).resolve()
    if not re.fullmatch(r"[A-Za-z0-9_\-]+", landuse):
        raise PhysicsResourceError(f"--landuse {landuse!r} is not a LANDUSE.TBL section name")
    if data_only:
        # A fetch may never fall through to the compiler-requiring referee.
        for name, sha in _thompson_expected().items():
            path = cache / 'thompson-strict' / 'tables' / name
            if not path.is_file() or sha256_file(path) != sha:
                raise PhysicsResourceError(f'fetched Thompson DATA is missing or changed: {path}')
        from .thompson_aero_v842 import AUXILIARY_SHA256
        aerosol = cache / 'thompson-strict' / 'aerosol' / 'auxiliary.bin'
        if not aerosol.is_file() or sha256_file(aerosol) != AUXILIARY_SHA256:
            raise PhysicsResourceError(f'fetched aerosol DATA is missing or changed: {aerosol}')
    maths = build_maths(cache)
    decoders = stage_decoders(cache, log)
    tables = cache / "mpas-data" / MPAS_DATA_COMMIT[:12]
    table = {}
    for name, (origin, path, sha) in TABLE_FILES.items():
        table[name] = {"path": str(fetch(name, origin, path, sha, tables / name, mpas_source=mpas_source,
                                         offline=offline)), "sha256": sha}
    thompson = build_thompson_strict(cache, mpas_source=mpas_source, thompson_tables=thompson_tables,
                                     offline=offline, log=log)
    packet = {
        "bl_mynn": {"device_math_source": maths["bl_mynn"]},
        "mp_thompson": {"auxiliary": str(thompson / "auxiliary.bin"),
                        "device_math_source": maths["mp_thompson"],
                        "rust_decoder": decoders["lookup_decode"],
                        "table_directory": str(thompson)},
        "rrtmg_lw": {"decoder": decoders["rw_mpas_rrtmg_lw_tables"], "table": table["RRTMG_LW_DATA"]},
        "rrtmg_sw": {"cache_root": str(_mkdir(cache / "rrtmg-sw-decoded")),
                     "data": table["RRTMG_SW_DATA"], "decoder": decoders["rw_mpas_rrtmg_sw_tables"]},
        "sf_noah": {"landuse": landuse, "rust_decoder": decoders["table_decode"],
                    "soil_type": SOIL_TYPE, "table_directory": str(tables)},
    }
    # Kain-Fritsch's packet is its warm_rain argument word.  Native never sets it, but reads it only
    # without cloud ice, so an install writes the word every ice-carrying run leaves unread and the slot
    # refuses the rest by name (kain_fritsch_v842.install_packet, check_warm_rain).  This was listed as
    # not built, so every hex-cuda Kain-Fritsch run on a plain install was refused.
    from .kain_fritsch_v842 import install_packet as kain_fritsch_packet
    packet["cu_kain_fritsch"] = kain_fritsch_packet()
    not_built = dict(NOT_BUILT)
    # RRFS fork GPU ports (the hrrrv5 stage's default hex-cuda slots).
    if _rrfs_module_present("tempo_v303"):
        packet["mp_tempo"] = {"device_math": maths["mp_tempo"], "session_tables": True}
    if not include_mynnedmf:
        not_built["bl_mynnedmf"] = "not selected by this run's active CUDA slots"
    else:
        for key, module in rrfs_mynnedmf_images():
            if not _rrfs_module_present(module.split(".", 1)[1]):
                continue
            try:
                packet[key] = build_mynnedmf_image(cache, log, arch=image_arch, module=module)
            except PhysicsResourceError as error:
                not_built[key] = str(error)
    for scheme in THOMPSON_SHARING:
        if _module_present(scheme):
            packet[scheme] = packet["mp_thompson"]
        else:
            not_built[scheme] = "its slot module is not in this install"
    if _module_present("mp_thompson_aerosols"):
        # A host whose toolchain cannot reproduce the reference's aerosol bytes loses this one slot, by
        # name, not the whole packet: every other MPAS suite still builds and runs.
        try:
            aerosol = build_thompson_aero_auxiliary(cache, thompson, Path(table["CCN_ACTIVATE_DATA"]["path"]),
                                                    mpas_source=mpas_source, offline=offline, log=log)
        except (PhysicsResourceError, subprocess.CalledProcessError) as error:
            not_built["mp_thompson_aerosols"] = f"its aerosol auxiliary arrays could not be generated here: {error}"
            log(f"NOT BUILT mp_thompson_aerosols: {not_built['mp_thompson_aerosols']}")
        else:
            packet["mp_thompson_aerosols"] = dict(packet["mp_thompson"], auxiliary=str(aerosol))
    else:
        not_built["mp_thompson_aerosols"] = "its slot module is not in this install"
    text = json.dumps(portable_packet(packet, cache), indent=2, sort_keys=True) + "\n"
    digest = hashlib.sha256(text.encode()).hexdigest()
    path = _mkdir(cache / "packets") / f"physics-resources-{digest}.json"
    # The packet text pins every decoder and table by SHA-256, so a receipt
    # written for this exact text already holds the decoded-word check.
    previous = path.with_suffix(".receipt.json")
    decoded = None
    if path.is_file() and previous.is_file():
        try:
            decoded = json.loads(previous.read_text(encoding="utf-8")).get("decoded_sha256")
        except ValueError:
            decoded = None
    if not decoded:
        decoded = decoded_digests(packet)
    differing = {key: value for key, value in decoded.items()
                 if key in DECODED_SHA256 and DECODED_SHA256[key] != value}
    if differing:
        raise PhysicsResourceError("a table decoder in this install decodes differently from the certified "
                                   "decoder: " + ", ".join(f"{key} sha256 {value}" for key, value in differing.items()))
    if not (path.is_file() and sha256_file(path) == digest):
        handle, temporary = tempfile.mkstemp(dir=path.parent, prefix=".partial-")
        with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(text)
        _replace_into(Path(temporary), path)
    receipt = {
        "schema": SCHEMA, "packet": str(path), "packet_sha256": digest, "cache": str(cache),
        "schemes": sorted(packet), "not_built": not_built, "landuse": landuse,
        "pins": {"mpas_model": MPAS_MODEL_COMMIT, "mmm_physics": MMM_PHYSICS_COMMIT, "mpas_data": MPAS_DATA_COMMIT},
        "decoded_sha256": decoded,
        "decoded_pinned": sorted(key for key in decoded if key in DECODED_SHA256),
        "toolchain": None if data_only else _toolchain(shutil.which("gfortran")),
        "thompson_tables": str(thompson),
        "thompson_tables_sha256": hashlib.sha256(json.dumps(_thompson_expected(), sort_keys=True)
                                                 .encode()).hexdigest(),
    }
    (path.with_suffix(".receipt.json")).write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n",
                                                   encoding="utf-8")
    return receipt


def ensure_packet(*, landuse: str | None = None, cache: Path | None = None,
                  include_mynnedmf: bool = True) -> Path:
    """First use from the MPAS route: build or reuse this install's packet, loudly."""
    landuse = landuse or DEFAULT_LANDUSE
    from .resources_fetch import fetch, packet_path
    try:
        resolve_resource("physics-resources", cache)
    except PhysicsResourceError:
        pass
    else:
        return packet_path(cache, landuse=landuse)
    if os.environ.get("WOOF_HEX_PHYSICS_RESOURCES_FIRST_USE", "") == "off":
        raise PhysicsResourceError("no --physics-resources packet was given and first-use fetching is off "
                                   "(WOOF_HEX_PHYSICS_RESOURCES_FIRST_USE=off); run `woof hex physics-resources "
                                   "fetch` (or an explicit `woof hex physics-resources build`)")
    fetch(cache, pin_sets=("v8.4.2",))
    return packet_path(cache, landuse=landuse)


# ---------------------------------------------------------------- CLI

def _build_command(arguments: argparse.Namespace) -> int:
    from ..licence_gate import LicenceRefusal, requested
    try:
        named = requested(arguments)
    except LicenceRefusal as error:
        raise PhysicsResourceError(str(error)) from error
    # A term the forecast door knows but the library does not compile (the RRFS stage's, set for its runs in
    # WOOF_HEX_LICENCE_OPT_IN) is not refused here; it changes nothing in the library, and build() drops it.
    pin_sets = tuple(dict.fromkeys(part.strip() for value in (arguments.pin_set or ["v8.4.2", "rrfs"])
                                   for part in value.split(",") if part.strip()))
    thompson_tables = arguments.thompson_tables
    if thompson_tables is None:
        # THE BREAKAGE THIS PREVENTS: on a pip install (no source checkout) `build` regenerated the strict
        # Thompson tables through the source-checkout referee and refused, although `fetch` had already
        # installed the same pinned bytes. build_thompson_strict re-verifies every table's SHA-256.
        try:
            thompson_tables = resolve_resource("thompson-tables", arguments.cache)
        except PhysicsResourceError:
            thompson_tables = None
    receipt = build(arguments.cache, mpas_source=arguments.mpas_source, thompson_tables=thompson_tables,
                    landuse=arguments.landuse, offline=arguments.offline,
                    native=not arguments.no_native_library, native_library=arguments.native_library,
                    mpich=arguments.mpich, jobs=arguments.jobs, licence_opt_in=named, pin_sets=pin_sets,
                    rrfs_source=arguments.rrfs_source,
                    # Both pin sets run on the ported framework, so each native library is a referee only,
                    # built when --build-native-library asks for it (or a licence opt-in names it).
                    native_pin_sets=tuple(p for p in pin_sets
                                          if arguments.build_native_library
                                          # a licence opt-in changes only the native library it names
                                          or (p == "v8.4.2" and set(named or ()) & set(_native_licence_terms()))))
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


def _native_licence_terms() -> tuple[str, ...]:
    from .native_library_build import LICENCE_COMPONENTS
    return tuple(LICENCE_COMPONENTS)


def _resolve_command(arguments: argparse.Namespace) -> int:
    print(resolve_resource(arguments.key, arguments.cache))
    return 0


def _verify_command(arguments: argparse.Namespace) -> int:
    packet = load_packet(arguments.packet)
    print(json.dumps({"packet": str(arguments.packet), "decoded_sha256": decoded_digests(packet),
                      "pinned": DECODED_SHA256}, indent=2, sort_keys=True))
    return 0


def add_physics_resources_parser(commands: Any) -> None:
    parser = commands.add_parser(
        "physics-resources",
        help="fetch MPAS DATA resources, or explicitly build native referee tools from source",
        description=(
            "Fetches verified Thompson DATA and pinned upstream tables, then assembles the packet "
            "with this wheel's maths and Rust helpers. No host compiler is needed. Use fetch "
            "--from DIR for offline installation. Native referee libraries require explicit source "
            "builds with C and Fortran compilers; no native library or Fortran runtime is distributed."),
    )
    verbs = parser.add_subparsers(dest="physics_resources_verb", required=True)
    from .resources_fetch import add_parser
    add_parser(verbs)
    build_parser = verbs.add_parser("build", help="rebuild the packet and the MPAS-Model tree from pinned sources; "
                                    "--build-native-library also compiles the native referee (C and Fortran compilers)")
    build_parser.add_argument("--cache", type=Path, default=None,
                              help="cache directory (default $WOOF_HEX_CACHE_ROOT/physics-resources, "
                                   "else ~/.cache/gpuwm-hex/physics-resources)")
    build_parser.add_argument("--mpas-source", type=Path, default=None,
                              help="a local MPAS-Model v8.4.2 tree (with MPAS-Data tables in physics_wrf/files) "
                                   "to read pinned files from instead of fetching them")
    build_parser.add_argument("--thompson-tables", type=Path, default=None,
                              help="a directory holding the strict native MP_THOMPSON_*.DBL tables (and "
                                   "optionally auxiliary.bin) instead of generating them with gfortran")
    build_parser.add_argument("--landuse", default=DEFAULT_LANDUSE,
                              help=f"the Noah LANDUSE.TBL section the case's static uses (default {DEFAULT_LANDUSE})")
    build_parser.add_argument("--offline", action="store_true", help="never fetch; fail by name instead")
    build_parser.add_argument("--native-library", type=Path, default=None,
                              help="adopt an existing libmpasphys_native.so (with its abi.json and "
                                   "build-receipt.json beside it) instead of compiling one here")
    # The v8.4.2 GPU route, the headline proof deck included, runs the ported framework and loads no native
    # library (runtime_v842.framework_only), so `build` compiles none unless asked: a pip install needs no C or
    # Fortran compiler and no Fortran runtime for any hex-cuda run or proof.
    build_parser.add_argument("--build-native-library", action="store_true",
                              help="also compile the native referee library (libmpasphys_native.so, and "
                                   "libmpasphys_native_rrfs.so for --pin-set rrfs) from source with C and Fortran "
                                   "compilers; only native-cpu referee runs and module replays use it")
    build_parser.add_argument("--no-native-library", action="store_true",
                              help="build no native library for any pin set (none is built unless "
                                   "--build-native-library asks for it)")
    build_parser.add_argument("--mpich", type=Path, default=None,
                              help="an existing MPICH install prefix (bin/mpif90, bin/mpicc) instead of building "
                                   "the pinned MPICH 4.3.2")
    build_parser.add_argument("--jobs", type=int, default=None, help="parallel make jobs (default: cores, max 16)")
    build_parser.add_argument("--pin-set", action="append", default=None, metavar="SET",
                              help="build only these native pin sets (repeatable or comma-separated): 'v8.4.2' "
                                   "(the MPAS-A v8.4.2 suites) and 'rrfs' (RRFS-MPAS: the GSL v8.3.1 fork); "
                                   "default both")
    build_parser.add_argument("--rrfs-source", type=Path, default=None,
                              help="a local RRFSx/MPAS-Model 1d6e370e checkout with its submodules to read pinned "
                                   "files from instead of fetching them")
    build_parser.add_argument(
        "--licence-opt-in", action="append", default=None, metavar="TERMS",
        help="compile the native library WITH the upstream code under these terms instead of its licence stub "
             "(repeatable or comma-separated; or WOOF_HEX_LICENCE_OPT_IN): 'ucar-noahmp' builds MPAS's Noah-MP "
             "v5.0 (UCAR terms: no sale of it or of any work that contains it). Default: the library carries a "
             "stub that stops a run selecting Noah-MP v5.0, and contains no code under those terms")
    build_parser.set_defaults(handler=_build_command)
    verify = verbs.add_parser("verify", help="decode a packet's tables and print the decoded-word digests")
    verify.add_argument("packet", type=Path)
    verify.set_defaults(handler=_verify_command)
    resolve = verbs.add_parser("resolve", help="print the installed path behind a portable resource key")
    resolve.add_argument("key", choices=RESOURCE_KEYS)
    resolve.add_argument("--cache", type=Path, default=None)
    resolve.set_defaults(handler=_resolve_command)
