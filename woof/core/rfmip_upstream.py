"""The RFMIP clear-sky files, fetched from upstream rather than shipped.

The five RFMIP NetCDFs the RRTMGP acceptance gates read are not part of
any WOOF-ArWen artifact or of the public tree:

* the four reference-result files are CC-BY-NC-SA-4.0, and carrying them
  would attach a non-commercial restriction to whatever carries them;
* the input file's embedded licence attribute names CC-BY-SA-4.0 while
  linking the CC-BY-4.0 deed, and redistributing it would mean choosing
  one of two readings its producer never resolved.

They stay exactly where they are published, the rrtmgp-data v1.9 commit
pinned below, and this module fetches each one on demand into a user
cache, refusing any byte that does not match its SHA-256.  The model
itself never calls it: RRTMGP runs on the 136 numbers derived from the
input file by ``tools/derive_rrtmgp_trace_climatology.py``, shipped as
``rrtmgp-trace-gas-climatology.json`` under CC-BY-SA-4.0.  The callers are
the acceptance tests, :func:`woof.core.rrtmgp.rfmip_clear_sky` and
``tools/derive_rrtmgp_trace_climatology.py``.
"""

from __future__ import annotations

import hashlib
import os
import urllib.error
import urllib.request
from pathlib import Path

#: rrtmgp-data tag v1.9.
UPSTREAM_COMMIT = "eff0433faf9cbac3ad14fbf608bef0c26ebc4c79"
_RAW = ("https://raw.githubusercontent.com/earth-system-radiation/"
        f"rrtmgp-data/{UPSTREAM_COMMIT}/examples/rfmip-clear-sky/")

#: local name -> (upstream URL, bytes, SHA-256, licence as published).
RFMIP_FILES: dict[str, tuple[str, int, str, str]] = {
    "rfmip-clear-sky-inputs.nc": (
        _RAW + "inputs/multiple_input4MIPs_radiation_RFMIP_"
               "UColorado-RFMIP-1-2_none.nc",
        1859666,
        "b8dc05d7cd2e0e6354b4a6198771ddf3bc09f18d72b49f20a41e2024e2fd51f4",
        "embedded attribute names CC-BY-SA-4.0 and links CC-BY-4.0"),
    "rfmip-clear-sky-reference-lw-down.nc": (
        _RAW + "reference/rld_Efx_RTE-RRTMGP-181204_rad-irf_r1i1p1f1_gn.nc",
        484462,
        "8629ec4b1caaea5a5c1756f25b432637369725ee684c61af0dad5c9ca37556b5",
        "CC-BY-NC-SA-4.0"),
    "rfmip-clear-sky-reference-lw-up.nc": (
        _RAW + "reference/rlu_Efx_RTE-RRTMGP-181204_rad-irf_r1i1p1f1_gn.nc",
        484460,
        "254569d9bb0934fb510306c3e22e13ea826bd918727b477fc20600213923493c",
        "CC-BY-NC-SA-4.0"),
    "rfmip-clear-sky-reference-sw-down.nc": (
        _RAW + "reference/rsd_Efx_RTE-RRTMGP-181204_rad-irf_r1i1p1f1_gn.nc",
        485284,
        "f9b0313fdf74598859a7caf27a5d1395b7fe1e445c9620a66856cc19eaf5e5b9",
        "CC-BY-NC-SA-4.0"),
    "rfmip-clear-sky-reference-sw-up.nc": (
        _RAW + "reference/rsu_Efx_RTE-RRTMGP-181204_rad-irf_r1i1p1f1_gn.nc",
        485284,
        "0ea3f4272d9ef088db6ffd07153587863a052e3cf8b3bf04bfb7c9288ed8b324",
        "CC-BY-NC-SA-4.0"),
}

#: SHA-256 of ``rrtmgp-trace-gas-climatology.json``, the table
#: ``tools/derive_rrtmgp_trace_climatology.py`` writes.  A table whose bytes differ was
#: not derived from the pinned input file, and
#: :func:`woof.core.rrtmgp.load_trace_climatology` refuses it.
TRACE_CLIMATOLOGY_SHA256 = (
    "71d7f85758fda8cf05df66100ccd0a974fed71216a1b7facad2083bc3dd3b70e")

#: The RFMIP input file, as a restart manifest records it under the
#: ``rrtmgp_rfmip`` role.  Until 2.8.0 the RRTMGP driver opened this file
#: itself, so every RRTMGP checkpoint written before then names these
#: bytes.  The driver now reads the same float64 numbers from the derived
#: table, so the role keeps naming the source while the table matches its
#: pin (:func:`woof.io.restart._active_asset_identity`) and those
#: checkpoints stay resumable.
TRACE_CLIMATOLOGY_SOURCE = {
    "path": "data/rrtmgp/rfmip-clear-sky-inputs.nc",
    "bytes": 1859666,
    "sha256": "b8dc05d7cd2e0e6354b4a6198771ddf3bc09f18d72b49f20a41e2024e2fd51f4",
}

#: Overrides the cache directory (a shared fixture store, an offline copy).
CACHE_ENV = "WOOF_RFMIP_DIR"


class RfmipUnavailable(RuntimeError):
    """A pinned RFMIP file could not be produced; the text says why."""


def rfmip_cache_dir() -> Path:
    """``$WOOF_RFMIP_DIR``, else ``~/.woof/cache/rfmip``."""

    override = os.environ.get(CACHE_ENV)
    if override:
        return Path(override)
    return Path.home() / ".woof" / "cache" / "rfmip"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def fetch_rfmip(name: str, *, path: Path | None = None,
                download: bool = True) -> Path:
    """The verified local copy of one pinned RFMIP file.

    ``path`` names a copy the caller already has; it is verified, never
    downloaded over.  Otherwise the cache is used and, when the file is
    absent and ``download`` is true, filled from the pinned URL.  Any
    mismatch in size or SHA-256 refuses: a gate compared against bytes
    nobody pinned is not the gate.
    """

    try:
        url, size, sha256, _licence = RFMIP_FILES[name]
    except KeyError:
        raise ValueError(f"{name!r} is not a pinned RFMIP file; known: "
                         f"{sorted(RFMIP_FILES)}") from None
    target = Path(path) if path is not None else rfmip_cache_dir() / name
    if not target.is_file():
        if path is not None or not download:
            raise RfmipUnavailable(
                f"{name} is not at {target}; it is fetched from {url} "
                f"(set {CACHE_ENV} to a directory holding a verified copy "
                f"to run offline)")
        target.parent.mkdir(parents=True, exist_ok=True)
        partial = target.with_name(target.name + ".part")
        try:
            with urllib.request.urlopen(url, timeout=120) as response, \
                    partial.open("wb") as sink:
                for block in iter(lambda: response.read(1 << 20), b""):
                    sink.write(block)
        except (urllib.error.URLError, OSError) as error:
            partial.unlink(missing_ok=True)
            raise RfmipUnavailable(
                f"{name}: download from {url} failed: {error}") from error
        got = _sha256(partial)
        if partial.stat().st_size != size or got != sha256:
            partial.unlink(missing_ok=True)
            raise RfmipUnavailable(
                f"{name}: {url} served {got}, pinned {sha256}; refused")
        os.replace(partial, target)
    if target.stat().st_size != size or _sha256(target) != sha256:
        raise RfmipUnavailable(
            f"{target} is not the pinned {name} (sha256 {sha256}); refused")
    return target


__all__ = ["CACHE_ENV", "RFMIP_FILES", "RfmipUnavailable",
           "TRACE_CLIMATOLOGY_SHA256", "TRACE_CLIMATOLOGY_SOURCE",
           "UPSTREAM_COMMIT", "fetch_rfmip", "rfmip_cache_dir"]
