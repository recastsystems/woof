"""Which MPAS-route output files are written lossless-compressed, and how.

Ruling 2026-10-28: "Lossless compression is the default"; 2026-10-27: Regional and hex get features together.
WOOF's history frames are lossless NetCDF-4 since woof 2.9.10, and the ArWen-physics route here already
writes deflated frames (woof.hex.history_deflate). The MPAS-physics route (the hex default) wrote its history
and diagnostics files uncompressed through rw_mpas_stream. They are now written lossless-compressed by
rw_mpas_stream itself: NETCDF4_CLASSIC, chunks byte-shuffled and deflated on every core by the engine's
netcdf_writer (the WOOF history writer, below the forecast's priority), never by HDF5 on one thread. Every
dimension, attribute, variable and word is the one the uncompressed file holds.

Only the ``output`` (history) and ``diagnostics`` streams are compressed. Init, restart, LBC, IAU and
``da_state`` files keep their native container: native MPAS and MPAS-JEDI read those through PnetCDF,
which has no NetCDF-4 reader. A stream whose file holds several output times is not compressed either:
its later records append to the file, and a compressed file is written whole.

The knobs are WOOF's: ``GPUWM_HISTORY_COMPRESSION`` ``deflate`` (default) or ``none`` (the stream's own
``io_type`` container, as before), ``GPUWM_HISTORY_COMPRESSION_LEVEL`` (zlib 1-9, default 4) and
``GPUWM_HISTORY_COMPRESSION_THREADS`` (0 = every core). A native-identity proof (woof.hex.verify_mpas)
writes ``none``: it compares each file's container with native MPAS's.
"""
from __future__ import annotations

import os
from typing import Any, Mapping

ENVIRONMENT = "GPUWM_HISTORY_COMPRESSION"
LEVEL_ENVIRONMENT = "GPUWM_HISTORY_COMPRESSION_LEVEL"
THREADS_ENVIRONMENT = "GPUWM_HISTORY_COMPRESSION_THREADS"
#: The streams whose files nothing reads through PnetCDF.
COMPRESSED_STREAMS = frozenset({"output", "diagnostics"})


def mode() -> str:
    """``deflate`` (default) or ``none``, from GPUWM_HISTORY_COMPRESSION (WOOF's knob)."""
    chosen = (os.environ.get(ENVIRONMENT, "deflate") or "deflate").strip().lower() or "deflate"
    if chosen not in ("deflate", "none"):
        raise ValueError(f"unknown history compression {chosen!r} ({ENVIRONMENT}): expected 'deflate' "
                         "(default, lossless NetCDF-4) or 'none' (the stream's own container)")
    return chosen


#: The filename tokens that name an output time to the second (MPAS mpas_stream_manager.F:4136-4201).
TIME_TOKENS = ("$Y", "$M", "$D", "$h", "$m", "$s")


def single_frame(options: Mapping[str, Any]) -> bool:
    """Whether every output time of a stream opens a file of its own (MPAS mpas_stream_manager.F:4136-4201).

    Its file bucket is its output interval (an unset filename_interval is, as in MPAS) and its filename names the
    time to the second, so two output times never share a file; initial_only and final_only write one frame.
    A template without the full time (``history.nc``) gathers frames into one file by appending: not compressed.
    """
    output = options.get("output_interval", "none")
    template = str(options.get("filename_template") or "")
    if output in ("initial_only", "final_only"):
        return True
    if output == "none" or not all(token in template for token in TIME_TOKENS):
        return False
    return options.get("filename_interval") in (None, "output_interval", output)


def for_stream(name: str, options: Mapping[str, Any], *, append: bool, record_index: int | None) -> dict | None:
    """The ``compression`` member of an rw_mpas_stream write request, or None for the stream's own container."""
    if name not in COMPRESSED_STREAMS or append or (record_index or 0) != 0 or not single_frame(options):
        return None
    if mode() == "none":
        return None
    level = int(os.environ.get(LEVEL_ENVIRONMENT, "4") or 4)
    threads = int(os.environ.get(THREADS_ENVIRONMENT, "0") or 0)
    return {"level": max(1, min(level, 9)), "threads": max(0, threads)}
