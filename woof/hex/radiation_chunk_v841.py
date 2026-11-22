"""Radiation column-chunk width for the resident RRTMG legacy adapter.

What was measured (RTX PRO 6000 host, global 30 km on 2 cards = 331,786
columns per rank, integrate/hex-next 53559c5 + engine d547dc8c3,
2026-10-11): a radiation step costs 4.5-5.2 s against 0.66-0.68 s for an
ordinary step.  2.9 s of that is one host computation, the climatological
ozone latitude/time interpolation on its Python fallback when the engine's
CPU bridge is not built (see ``ozone_route``).  The rest is the RRTMG
kernels plus ~28,000-44,000 launches of CuPy glue per call: the engine
sizes each chunk to the card's resident threads (2,304 LW / 2,816 SW
columns), right for occupancy but ~100 launches per chunk.

A wider chunk cuts the launch count in proportion.  The adapter already
accepts an explicit ``column_chunk``; per-column results are bitwise
identical at any width (``rrtmg_lw.batch_column_chunk`` and
``rrtmg_sw.SwCuda.batch_column_chunk`` docstrings; re-proved here by every
history field, boundary fingerprint and NetCDF variable before/after).
This module picks the widest 256-column multiple whose whole-call transient,
priced by the engine's own ``legacy_radiation_vram_bytes``, fits
``RADIATION_CHUNK_BUDGET_FRACTION`` of the card or its mutex reservation,
whichever is smaller, never narrower than the
engine default.  Measured peak memory did not move (58.6 GB per card both
ways at 30 km on 2 cards).

``WOOF_HEX_RADIATION_CHUNK``: ``auto`` (default) applies the rule above,
``engine`` leaves the engine's own width, a positive integer forces a width.
"""
from __future__ import annotations

import os
from woof.hex import card_queue
from typing import Any

#: Fraction of the card's total memory the radiation transient may use.
#: 0.08 x 96 GiB = 7.7 GiB, which prices to 8,192 columns at 55 levels.
RADIATION_CHUNK_BUDGET_FRACTION = 0.08
#: Cards below this keep the engine width.  Breakage it prevents: the 30 km
#: four-rank run on 32 GiB RTX 5090s already peaks at 31.6 GiB of 32 (RTX 5090 host,
#: global30-gpu-v06 receipt, 2026-10-10), so any wider transient there could
#: run a forecast out of memory mid-run.
RADIATION_CHUNK_MIN_CARD_BYTES = 64 * 2**30
#: ``woof.core.rrtmg_lw.BATCH_CHUNK_QUANTUM``.
QUANTUM = 256


def _setting() -> str:
    text = os.environ.get("WOOF_HEX_RADIATION_CHUNK", "auto").strip().lower()
    if text in ("auto", "engine"):
        return text
    try:
        value = int(text)
    except ValueError:
        value = 0
    if value < 1:
        raise ValueError("WOOF_HEX_RADIATION_CHUNK must be auto, engine or a positive integer; "
                         "prevents an ambiguous radiation receipt")
    return str(value)


def chunk_capacity(total_bytes: int) -> int:
    """Memory this process may use when deciding whether to widen radiation.

    Pool admission reserves a share of a card. Widening against the whole
    card can exceed that reservation even on a small regional mesh.
    Exclusive execution and execution without a pool use the card capacity.
    """
    if card_queue.get(card_queue.MODE) != "pool":
        return total_bytes
    budget = int(card_queue.get(card_queue.VRAM_MIB) or 0) * 2**20
    return min(total_bytes, budget) if budget > 0 else total_bytes


def choose_width(*, ncol: int, nz: int, p_top: float, total_bytes: int, price: Any,
                 default_bytes: int | None = None) -> int | None:
    """Widest quantum multiple whose priced transient fits the budget.

    ``price(width)`` is the engine's whole-call transient for that width.
    Returns ``None`` (keep the engine width) on small cards or when no width
    wider than the engine default fits.
    """
    if total_bytes < RADIATION_CHUNK_MIN_CARD_BYTES or ncol <= QUANTUM:
        return None
    budget = int(RADIATION_CHUNK_BUDGET_FRACTION * total_bytes)
    if default_bytes is not None and default_bytes >= budget:
        return None
    top = -(-int(ncol) // QUANTUM)
    low, high = 0, top
    while low < high:
        mid = (low + high + 1) // 2
        if price(mid * QUANTUM) <= budget:
            low = mid
        else:
            high = mid - 1
    if low == 0:
        return None
    return min(low * QUANTUM, top * QUANTUM)


def ozone_route() -> dict[str, Any]:
    """Say whether the radiation ozone interpolation runs native or in Python.

    Measured (RTX PRO 6000 host, 2026-10-11): ``woof.ingest.wrf_ozone.ozn_latitude_time_int``
    takes 3.09 s for 327,680 columns on its Python fallback and 0.015 s through
    the native CPU bridge (``libgpuwm_preprocess_cpu.so``), bitwise identical.
    The fallback runs on every radiation call of every rank when the engine
    checkout has no built bridge (the RTX PRO 6000 host integration install), and it was
    2.9 of the 4.2 s a radiation step added at 327,680 columns per rank.
    """
    try:
        from woof.ingest import wrf_ozone
        native = wrf_ozone._native_ozone_entries() is not None
    except Exception as error:  # a receipt never stops a forecast
        return {"native": None, "error": f"{type(error).__name__}: {error}"}
    route: dict[str, Any] = {"native": bool(native)}
    if not native:
        route["remedy"] = ("build the engine CPU bridge (cd tools/grib1_bridge && cargo build --release "
                           "--locked --offline) or set WOOF_CPU_PREPROCESS_BRIDGE; the Python fallback "
                           "costs about 9 us per column per radiation call")
    return route


def widen_radiation_chunk(seam: Any, *, cp: Any) -> dict[str, Any]:
    """Set the seam's legacy RRTMG adapter chunk width; return a receipt."""
    setting = _setting()
    driver = getattr(seam, "_driver", None)
    adapter = getattr(driver, "radiation_callable", None)
    receipt: dict[str, Any] = {"setting": setting, "applied": False}
    if adapter is not None and type(adapter).__name__ == "RRTMGLegacyRadiation":
        receipt["ozone_interpolation"] = ozone_route()
    if adapter is None or type(adapter).__name__ != "RRTMGLegacyRadiation":
        receipt["reason"] = "no legacy RRTMG adapter on this seam"
        return receipt
    if getattr(adapter, "column_chunk", None) is not None:
        receipt["reason"] = "adapter already carries an explicit column_chunk"
        receipt["width"] = int(adapter.column_chunk)
        return receipt
    if setting == "engine":
        receipt["reason"] = "WOOF_HEX_RADIATION_CHUNK=engine"
        return receipt
    identity = getattr(seam, "_identity", {})
    ncol, nz = int(identity["n_columns"]), int(identity["n_levels"])
    p_top = getattr(adapter, "p_top", None)
    if p_top is None:
        p_top = identity["p_top_pa"]
    if setting != "auto":
        width = int(setting)
    else:
        from woof.core.rrtmg_legacy import legacy_radiation_vram_bytes

        def price(width):
            return int(legacy_radiation_vram_bytes(
                ncol=ncol, nz=nz, p_top=float(p_top), column_chunk=width,
                longwave=bool(getattr(adapter, "longwave", True)),
                shortwave=bool(getattr(adapter, "shortwave", True)),
                o3input=int(getattr(adapter, "o3input", 2)),
                aer_opt=int(getattr(adapter, "aer_opt", 0))))
        total = int(cp.cuda.Device().mem_info[1])
        default = int(legacy_radiation_vram_bytes(
            ncol=ncol, nz=nz, p_top=float(p_top),
            longwave=bool(getattr(adapter, "longwave", True)),
            shortwave=bool(getattr(adapter, "shortwave", True)),
            o3input=int(getattr(adapter, "o3input", 2)),
            aer_opt=int(getattr(adapter, "aer_opt", 0))))
        capacity = chunk_capacity(total)
        width = choose_width(ncol=ncol, nz=nz, p_top=float(p_top), total_bytes=capacity,
                             price=price, default_bytes=default)
        receipt.update(card_bytes=total, chunk_capacity_bytes=capacity,
                       engine_default_transient_bytes=default,
                       budget_fraction=RADIATION_CHUNK_BUDGET_FRACTION)
        if width is None:
            receipt["reason"] = "card or pool share below the wide-chunk floor or no wider width fits the budget"
            return receipt
        receipt["priced_transient_bytes"] = price(width)
    adapter.column_chunk = int(width)
    receipt.update(applied=True, width=int(width), columns=ncol, levels=nz)
    return receipt
