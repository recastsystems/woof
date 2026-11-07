"""New Tiedtke at the hex seam: the frozen column batch, driven at ``cu_physics = 16``.

WHY THIS ROW EXISTS.  RRFS v2 runs ``config_convection_scheme = 'cu_ntiedtke'``
on its 12 km mesh (rrfs-workflow 85ea55c0, ``scripts/exrrfs_fcst.sh:62-66``
with ``FCST_CONVECTION_SCHEME`` true in ``workflow/config_resources/
config.meshdep:31-35``).  The engine already carries the scheme as its
``cu_physics = 16`` adapter, :class:`woof.core.ntiedtke.NewTiedtke`, proven
against WRF's source in the engine's own parity gates.  The frozen column
batch cannot reach it: ``resolve_column_batch_cadences`` admits ``gf``,
``kf`` and ``None`` only, and ``mpas_column_batch.py`` is one of the sixteen
files this port pins by SHA-256, so a dispatch arm inside it would move bytes
every execution anchor is keyed to (``woof.hex.physics_backend_admission``).

WHAT THIS MODULE DOES, AND NOTHING MORE.  It owns no parameterization
arithmetic.  :class:`NewTiedtkeColumnBatchV841` is the frozen
``MpasColumnBatchPhysics`` class, unmodified and subclassed, whose
``PhysicsDriver`` is built by the engine's own ``initialize_physics`` at a
``RunConfig`` the frozen constructor produced, with ``cu_physics = 16`` and
``cudt_minutes = 0`` (the engine's own law for the scheme,
``woof/config.py`` ``validate_run_config``).  Phase one is still
``PhysicsDriver.compute`` and phase two is still ``microphysics.apply``; the
driver resolves ``cu_physics = 16`` to ``NewTiedtke`` exactly as an ARW run
does (``woof/core/physics.py``, ``initialize_physics``).

THE ONE COUPLING THIS ADDS: CUMULUS MOMENTUM.  New Tiedtke returns
``rucuten``/``rvcuten`` (``lmfdudv`` is a PARAMETER in the scheme).  The
frozen batch takes ``du``/``dv`` from the driver's retained raw YSU rates
only, because no scheme it admits produces momentum.  Native MPAS adds the
cumulus pair to the PBL pair before the cell-to-edge reconstruction
(fork 1d6e370e ``mpas_atmphys_todynamics.F``, ``case('cu_tiedtke',
'cu_ntiedtke')``: ``tend_toEdges(rucuten, rvcuten)`` then
``tend_u += rucuten_Edge * mass_edge``), so this seam returns
``du = du_pbl + rucuten`` and ``dv = dv_pbl + rvcuten``.  The raw A-grid rates
are captured from the scheme's own ``CumulusResult`` before the driver's
ARW coupling face-interpolates them (that coupled copy is never read by the
column batch).

WHAT NATIVE MPAS FEEDS THE SCHEME, AND WHAT THIS SEAM FEEDS IT.  The fork's
``convection_from_MPAS`` builds ``rqvften = rqvdynten + rqvblten`` and
``rthften = (rthdynten + rthblten + rthratenlw + rthratensw) * pi``
(``mpas_atmphys_driver_convection.F:1152-1154`` at 1d6e370e).
``NewTiedtke`` performs the same fold from the driver's lanes
(``woof/core/ntiedtke.py``, ``THE FOLD``): the dynamics lanes are the ones
hex already hands the frozen batch every step, the PBL lanes are retained by
the driver's shared PBL slot, the radiation lanes are the driver's held
rates.  ``config_gfl_sub3d = 1`` is inert under ``cu_ntiedtke``: the fork
allocates and couples the ``sub3d_*`` arrays only in its Grell-Freitas
cases (same file, ``allocate_convection`` and ``convection_to_MPAS``).

SCALE AWARENESS, PER CELL, DEFAULT ON.  cu_ntiedtke has no namelist switch
for it in MPAS-A v8.4.1: the scheme always builds two factors from the
spacing it is handed (MMM-physics 20250616-MPASv8.3 ``cu_ntiedtke.F90:
228-239``, ``dxref = 15000``) -- ``scale_fac`` multiplies the deep closure's
adjustment time (``:676``) and ``scale_fac2`` divides the shallow cloud-base
mass flux (``:715``) -- and native hands it one spacing PER CELL,
``dx_p = len_disp / meshDensity**0.25``
(``mpas_atmphys_driver_convection.F:840`` at v8.4.1, ``config_len_disp``
resolving to the mesh ``nominalMinDc``).  This seam hands the scheme the same
vector: ``dx_column_m``, built by ``woof.hex.cuda_gwdo_v841.native_cell_dx_m``
and already a required sealed static, reaches the engine's kernels through
the driver's ``gf_dx_column`` carrier.  Until 2026-10-11 the engine's adapter
filled the kernels' ``dx`` with the batch scalar ``dx_m`` and ignored the
carrier, so a variable-resolution mesh ran every cell at one nominal spacing
(on a uniform mesh the two are the same number).
``WOOF_HEX_NTIEDTKE_SCALE_AWARE=off`` is a sensitivity control, never a
default: it holds every cell finer than ``dxref`` at ``dxref``, so the
sub-15 km reduction never applies.

DECLARED DIFFERENCES (receipt-carried, not hidden).
* The scheme code is the engine's port of WRF's New Tiedtke, not the fork's
  ``physics_mmm`` submodule; both descend from the same MMM source.

RESTART.  New Tiedtke is called every model step and carries no hold, so the
captured momentum is recomputed before every use and needs no serialized
slot; the restart identity names the scheme so a payload cannot cross
schemes.
"""

from __future__ import annotations

import dataclasses
import gc
import os
import threading
from hashlib import sha256
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np


#: The engine files New Tiedtke executes beyond the frozen sixteen.  Pinned
#: here by SHA-256 because the frozen manifest does not cover them: a run of
#: this row executes these bytes, so these bytes are named.  Measured from
#: the published gpuwm 2.8.0 wheel (gpuwm-2.8.0-py3-none-manylinux_2_28_x86_64)
#: on 2026-10-05.
NTIEDTKE_ENGINE_SOURCES: Mapping[str, str] = MappingProxyType(
    {
        "woof/core/ntiedtke.py": (
            "34e03bca7f1d4a99d68378b9334d2d96c0c331f44a8ddf179e02200015b60bb9"
        ),
        "woof/core/kernels/ntiedtke.cu": (
            "06daa934a71a02797fbaaaef243b83d4c43382e2dcd1cbd83505393d9373af7a"
        ),
    }
)

#: The engine's own ``cu_physics`` code for New Tiedtke (``woof/config.py``
#: ``CU_SCHEMES``).
CU_PHYSICS_NTIEDTKE: int = 16

NTIEDTKE_ROW_SCHEMA = "gpuwm-hex.ntiedtke-column-row/v1"

#: Turns the sub-``dxref`` reduction off for a sensitivity run.  Unset, or
#: any value outside ``_OFF_WORDS``, is the scheme as MPAS-A runs it.
SCALE_AWARE_ENV = "WOOF_HEX_NTIEDTKE_SCALE_AWARE"
_OFF_WORDS = frozenset({"0", "off", "false", "no"})

DECLARED_DIFFERENCES: tuple[str, ...] = (
    "scheme code is the engine's port of WRF New Tiedtke, not the fork's physics_mmm submodule",
    "config_gfl_sub3d=1 is inert under cu_ntiedtke in the fork and has no counterpart here",
)


class NtiedtkeRowError(RuntimeError):
    """The New Tiedtke row refused, by name."""


def verify_ntiedtke_sources(root: str | Path) -> Mapping[str, str]:
    """Hash the engine files the row executes; refuse only a missing one.

    ``root`` is the engine tree whose sixteen frozen paths were already
    verified by :func:`woof.hex.cuda_arwen_physics_v841._verify_checkout_root`.

    Returns the SHA-256 of the bytes this run executes.  A missing file is
    refused: the row cannot construct the scheme without it.

    RETIRED 2026-10-10 as a gate (owner ruling; Refusals law): a hash that differs from the 2.8.0 bytes the row
    was first measured on used to raise "another engine's scheme is a
    different, unmeasured row".  That named no breakage, only missing
    evidence, and it refused New Tiedtke -- the MPAS-A mesoscale_reference
    cumulus scheme and the hex default at 20 km and coarser -- on every
    engine after 2.8.0 (found by the 2026-10-10 8-card RTX PRO 6000 host proof run on woof
    2.8.9).  Whether the executed bytes are the measured row's is now
    recorded by :func:`measured_row_match` in the run's receipt.
    """

    base = Path(root)
    found: dict[str, str] = {}
    for relative in NTIEDTKE_ENGINE_SOURCES:
        path = base / relative
        if not path.is_file():
            raise NtiedtkeRowError(
                f"the New Tiedtke row executes {relative} and the engine at "
                f"{base} does not carry it"
            )
        found[relative] = sha256(path.read_bytes()).hexdigest()
    return MappingProxyType(found)


def measured_row_match(found: Mapping[str, str]) -> dict[str, Any]:
    """Record whether executed New Tiedtke bytes are the measured 2.8.0 row's.

    Evidence for the receipt, never a permission: nothing refuses on it.
    """

    files = {
        relative: {
            "executed_sha256": found.get(relative),
            "measured_row_sha256": wanted,
            "matched": found.get(relative) == wanted,
        }
        for relative, wanted in NTIEDTKE_ENGINE_SOURCES.items()
    }
    return {
        "measured_row": "gpuwm 2.8.0 wheel, 2026-10-05",
        "matched": all(item["matched"] for item in files.values()),
        "files": files,
    }


def _imported_engine_match() -> dict[str, Any]:
    """:func:`measured_row_match` for the engine this process imported."""

    import woof

    if not getattr(woof, "__file__", None):
        return {"engine_root": None, "matched": None, "files": {}}
    root = Path(woof.__file__).resolve().parents[1]
    found = {
        relative: sha256((root / relative).read_bytes()).hexdigest()
        for relative in NTIEDTKE_ENGINE_SOURCES
        if (root / relative).is_file()
    }
    return {"engine_root": str(root), **measured_row_match(found)}


def scale_aware_requested(environ: Mapping[str, str] | None = None) -> bool:
    """True unless ``WOOF_HEX_NTIEDTKE_SCALE_AWARE`` turns it off."""

    values = os.environ if environ is None else environ
    return str(values.get(SCALE_AWARE_ENV, "on")).strip().lower() not in _OFF_WORDS


def engine_takes_column_dx() -> bool:
    """Whether the imported engine's New Tiedtke reads the per-cell dx."""

    from woof.core import ntiedtke as engine_ntiedtke

    return hasattr(engine_ntiedtke.NewTiedtke, "_column_dx")


def scale_awareness_receipt(
    dx_column_m: Any, *, scale_aware: bool, dxref_m: float
) -> dict[str, Any]:
    """What the scheme's two scale factors are on this mesh, cell by cell.

    The arithmetic is cu_ntiedtke.F90:230-238 on the host, for the receipt;
    the device kernel computes the values the forecast uses.
    """

    dx = np.asarray(dx_column_m, dtype=np.float32).reshape(-1)
    seen = dx if scale_aware else np.maximum(dx, np.float32(dxref_m))
    fine = seen < np.float32(dxref_m)
    with np.errstate(divide="ignore", invalid="ignore"):
        base = np.float32(1.06133) + np.log(np.float32(dxref_m) / seen)
    factor = np.where(
        fine, base * base * base, np.float32(1.0) + np.float32(1.33e-5) * seen
    ).astype(np.float32)
    factor2 = np.where(fine, np.sqrt(factor), np.float32(1.0)).astype(np.float32)
    return {
        "scale_aware": bool(scale_aware),
        "switch": SCALE_AWARE_ENV,
        "dx_source": "per cell, len_disp/meshDensity**0.25 (native_cell_dx_m)",
        "dxref_m": float(dxref_m),
        "cells": int(dx.size),
        "cells_finer_than_dxref": int(np.count_nonzero(dx < np.float32(dxref_m))),
        "mesh_dx_m": {"min": float(dx.min()), "max": float(dx.max())},
        "scheme_dx_m": {"min": float(seen.min()), "max": float(seen.max())},
        "deep_adjustment_time_factor": {
            "min": float(factor.min()),
            "max": float(factor.max()),
        },
        "shallow_mass_flux_divisor": {
            "min": float(factor2.min()),
            "max": float(factor2.max()),
        },
    }


#: New Tiedtke chunk cap per multiprocessor on this seam: 128 against the
#: engine's 512 (21,760 columns on a 170-SM card), a quarter of the held
#: workspace.  See MomentumCapturingNewTiedtke.__init__.
HEX_NT_TILE_COLUMNS_PER_SM = 128


def _capturing_new_tiedtke() -> Any:
    """The engine's ``NewTiedtke``, recording the raw momentum it returns."""

    from woof.core.ntiedtke import NewTiedtke

    class MomentumCapturingNewTiedtke(NewTiedtke):
        """``NewTiedtke`` that keeps the RAW A-grid ``rucuten``/``rvcuten``.

        The driver couples the result for an ARW grid (a C-grid face roll),
        which the column batch never reads; the raw pair is what MPAS adds to
        its PBL momentum before ``tend_toEdges``.  The arrays are the
        scheme's own freshly allocated outputs, valid until the next call.
        """

        def __init__(self) -> None:
            super().__init__()
            self.last_momentum: tuple[Any, Any] | None = None
            self.calls = 0
            # Hex memory census item 3: the engine holds the scheme's
            # workspace for the run at its chunk width (89 level arrays plus
            # cutypen scratch, 22.3 KB per column).  Its default cap, 512
            # columns per multiprocessor, made that the whole mesh up to
            # 87,040 columns on a 170-SM card: 891 MiB held all run at
            # x1.40962, about 1.85 GiB at 30 km.  This seam walks it at
            # HEX_NT_TILE_COLUMNS_PER_SM.  Every stage indexes the global
            # column and llo3 is reduced and checked per chunk, exactly as on
            # any mesh already wider than the cap, so the width moves memory
            # and launch counts only.  An engine without the per-instance cap
            # (before development branch) ignores the attribute.
            self.tile_columns_per_sm = HEX_NT_TILE_COLUMNS_PER_SM

        def __call__(self, *, atmosphere, fields, state, cfg):
            result = super().__call__(
                atmosphere=atmosphere, fields=fields, state=state, cfg=cfg
            )
            if result.rucuten is None or result.rvcuten is None:
                raise NtiedtkeRowError(
                    "New Tiedtke returned no momentum pair; the row couples "
                    "rucuten/rvcuten as native MPAS does and refuses to drop "
                    "them"
                )
            self.last_momentum = (result.rucuten, result.rvcuten)
            self.calls += 1
            return result

    return MomentumCapturingNewTiedtke()


def _column_batch_class() -> type:
    from woof.core.mpas_column_batch import MpasColumnBatchPhysics

    return MpasColumnBatchPhysics


def _make_seam_class() -> type:
    base = _column_batch_class()

    class NewTiedtkeColumnBatchV841(base):  # type: ignore[misc, valid-type]
        """The frozen column batch with its driver at ``cu_physics = 16``."""

        def __init__(
            self,
            *,
            cumulus_scheme: str,
            cumulus_seconds: float,
            gf_ishallow: int | None = None,
            **values: Any,
        ) -> None:
            if cumulus_scheme != "ntiedtke":
                raise NtiedtkeRowError(
                    f"the New Tiedtke seam runs cumulus_scheme='ntiedtke', "
                    f"got {cumulus_scheme!r}"
                )
            dt = float(values["dt"])
            if cumulus_seconds is None or float(cumulus_seconds) != dt:
                raise NtiedtkeRowError(
                    "New Tiedtke is called every model step (WRF pins "
                    "cudt=0 for cu_physics=16 and the RRFS v2 namelists set "
                    f"no config_conv_interval): cumulus_seconds must equal "
                    f"dt={dt} s, got {cumulus_seconds!r}"
                )
            if gf_ishallow not in (None, 0):
                raise NtiedtkeRowError(
                    "gf_ishallow is Grell-Freitas's shallow switch and means "
                    "nothing under New Tiedtke"
                )
            # The frozen constructor, unmodified, at cumulus OFF: every
            # validation, buffer, state and seam carrier it builds is the
            # frozen class's own.
            super().__init__(
                cumulus_scheme=None,
                cumulus_seconds=None,
                gf_ishallow=0,
                **values,
            )
            self._rebuild_driver_at_ntiedtke(values)

        # --------------------------------------------------------------
        def _rebuild_driver_at_ntiedtke(self, values: Mapping[str, Any]) -> None:
            """Rebuild the driver with the engine's own ``cu_physics = 16``.

            The cumulus-off driver the frozen constructor built is released
            first, so its device memory returns to the pool before the
            replacement allocates; peak memory is one driver, not two.
            """

            import cupy as cp
            from woof.config import validate_run_config
            from woof.core.physics import initialize_physics

            cfg16 = dataclasses.replace(
                self._cfg, cu_physics=CU_PHYSICS_NTIEDTKE, cudt_minutes=0.0
            )
            validate_run_config(cfg16)

            ncol = int(self._ncol)
            state = self._state
            # Release the cumulus-off driver before building its successor.
            self._driver = None
            if getattr(state, "physics", None) is not None:
                state.physics = None
            gc.collect()
            cp.get_default_memory_pool().free_all_blocks()

            def surface(item: Any) -> Any:
                if np.ndim(item) == 0:
                    return item
                if hasattr(item, "__cuda_array_interface__"):
                    return item.reshape(1, ncol)
                return np.asarray(item).reshape(1, ncol)

            def soil(item: Any) -> Any:
                if np.ndim(item) == 0:
                    return item
                if hasattr(item, "__cuda_array_interface__"):
                    return item.reshape(item.shape[0], 1, ncol)
                item = np.asarray(item)
                return item.reshape(item.shape[0], 1, ncol)

            # The SAME arguments, with the SAME defaults, the frozen
            # constructor hands initialize_physics (mpas_column_batch.py,
            # MpasColumnBatchPhysics.__init__), plus the scheme callable.
            get = values.get
            xland = get("xland")
            lat = np.asarray(values["latitude_deg"], dtype=np.float32).reshape(-1)
            lon = np.asarray(values["longitude_deg"], dtype=np.float32).reshape(-1)
            self._ntiedtke = _capturing_new_tiedtke()
            self._driver = initialize_physics(
                state,
                cfg16,
                landmask=surface(get("landmask", 1.0)),
                xland=None if xland is None else surface(xland),
                xice_threshold=float(get("xice_threshold", 0.5)),
                tsk=surface(get("tsk", 300.0)),
                soil_temperature=soil(get("soil_temperature", 285.0)),
                soil_moisture=soil(get("soil_moisture", 0.30)),
                ivgtyp=surface(get("ivgtyp", 10)),
                isltyp=surface(get("isltyp", 6)),
                vegfra=surface(get("vegfra", 50.0)),
                tmn=surface(get("tmn", 285.0)),
                xice=surface(get("xice", 0.0)),
                snow=surface(get("snow", 0.0)),
                snow_depth=surface(get("snow_depth", 0.0)),
                glw=None,
                radiation_start_time=values["start_time"],
                radiation_latitude=lat.reshape(1, ncol),
                radiation_longitude=lon.reshape(1, ncol),
                cumulus=self._ntiedtke,
            )
            # The seam-owned dynamics forcing lanes, rebound to the new
            # driver exactly as the frozen constructor bound them.
            self._driver.gf_rthdynten = self._gf_rthdynten
            self._driver.gf_rqvdynten = self._gf_rqvdynten
            # The per-cell spacing native hands the scheme.  The engine's
            # adapter reads it from the driver's gf_dx_column carrier.
            dx_column = get("dx_column_m")
            if dx_column is None:
                raise NtiedtkeRowError(
                    "New Tiedtke builds its scale factors from each cell's "
                    "own spacing (cu_ntiedtke.F90:230-238) and the seam was "
                    "handed no dx_column_m; one nominal spacing would give "
                    "every cell of a variable-resolution mesh the same "
                    "convective adjustment time"
                )
            dx_host = np.asarray(dx_column, dtype=np.float32).reshape(ncol)
            takes_column_dx = engine_takes_column_dx()
            if not takes_column_dx and float(dx_host.min()) != float(dx_host.max()):
                raise NtiedtkeRowError(
                    "this mesh's cells span "
                    f"{float(dx_host.min()):.1f} m to {float(dx_host.max()):.1f} m "
                    "and the imported engine's New Tiedtke runs every column "
                    f"at the one nominal spacing {float(cfg16.dx):.1f} m "
                    "(its adapter has no per-column dx), so the fine cells "
                    "would get the coarse cells' convective adjustment time; "
                    "use an engine whose woof/core/ntiedtke.py reads "
                    "gf_dx_column"
                )
            scale_aware = scale_aware_requested()
            if not scale_aware and not takes_column_dx:
                raise NtiedtkeRowError(
                    f"{SCALE_AWARE_ENV}=off needs an engine whose New Tiedtke "
                    "carries the switch; on this engine the request would be "
                    "ignored and the run would be labelled as something it is "
                    "not"
                )
            self._driver.gf_dx_column = cp.asarray(dx_host.reshape(1, ncol))
            if takes_column_dx:
                from woof.core.ntiedtke import NT_DXREF_M

                self._ntiedtke.scale_aware = scale_aware
                dxref_m = float(NT_DXREF_M)
            else:
                dxref_m = 15000.0
            self._cfg = cfg16
            self._cadences = dict(
                self._cadences,
                cu_physics=CU_PHYSICS_NTIEDTKE,
                cudt_minutes=0.0,
                stepcu=1,
            )
            self._identity = dict(
                self._identity,
                cumulus_scheme="ntiedtke",
                stepcu=1,
                gf_ishallow=0,
            )
            nz = int(self._nz)
            self._cu_du = cp.zeros((nz, ncol), dtype=cp.float32)
            self._cu_dv = cp.zeros((nz, ncol), dtype=cp.float32)
            self._du_total = cp.zeros((nz, ncol), dtype=cp.float32)
            self._dv_total = cp.zeros((nz, ncol), dtype=cp.float32)
            self.ntiedtke_receipt = MappingProxyType(
                {
                    "schema": NTIEDTKE_ROW_SCHEMA,
                    "cu_physics": CU_PHYSICS_NTIEDTKE,
                    "cudt_minutes": 0.0,
                    "stepcu": 1,
                    "batch_dx_m": float(cfg16.dx),
                    "scale_awareness": {
                        "engine_reads_per_cell_dx": bool(takes_column_dx),
                        **scale_awareness_receipt(
                            dx_host, scale_aware=scale_aware, dxref_m=dxref_m
                        ),
                    },
                    "engine_sources_measured_row": dict(NTIEDTKE_ENGINE_SOURCES),
                    "measured_row_match": _imported_engine_match(),
                    "momentum": "du/dv = PBL + raw rucuten/rvcuten (native todynamics)",
                    "declared_differences": list(DECLARED_DIFFERENCES),
                }
            )

        # --------------------------------------------------------------
        def run_phase1(self, **arguments: Any):
            self._ntiedtke.last_momentum = None
            result = super().run_phase1(**arguments)
            if not result.cumulus_ran:
                raise NtiedtkeRowError(
                    "New Tiedtke did not run on a model step; the row is an "
                    "every-step scheme and a skipped call would hold stale "
                    "momentum"
                )
            momentum = self._ntiedtke.last_momentum
            if momentum is None:
                raise NtiedtkeRowError(
                    "the cumulus call ran and published no momentum pair"
                )
            nz, ncol = int(self._nz), int(self._ncol)
            self._cu_du[...] = momentum[0].reshape(nz, ncol)
            self._cu_dv[...] = momentum[1].reshape(nz, ncol)
            cp = self._cp
            cp.add(result.du, self._cu_du, out=self._du_total)
            cp.add(result.dv, self._cu_dv, out=self._dv_total)
            return dataclasses.replace(
                result, du=self._du_total, dv=self._dv_total
            )

    NewTiedtkeColumnBatchV841.__module__ = __name__
    return NewTiedtkeColumnBatchV841


_SEAM_CLASS: type | None = None
_SEAM_CLASS_LOCK = threading.Lock()


def seam_class() -> type:
    """The row's seam class, built once against the imported engine.

    THE BREAKAGE THE LOCK PREVENTS, MEASURED (2026-10-10, 8-card RTX PRO 6000 host, 8 same-host
    ranks constructed on a thread pool under free-threaded CPython 3.14t):
    without it two ranks could each build a class and the later one replaced
    the global, so a rank's seam was an instance of a class that
    ``seam_class()`` no longer returned.  The identity check in
    ``PersistentTwoPhaseCudaPhysicsBackendV841._private_binding_guard`` then
    refused rank 1 with "private radius binding requires the exact frozen
    Arwen v2 seam class" (30 km, New Tiedtke, run global30-g30-old); the
    same configuration family had constructed cleanly minutes earlier, so
    it was a race, not a configuration fault.
    """

    global _SEAM_CLASS
    if _SEAM_CLASS is None:
        with _SEAM_CLASS_LOCK:
            if _SEAM_CLASS is None:
                _SEAM_CLASS = _make_seam_class()
    return _SEAM_CLASS


def run_ntiedtke_column_batch(**values: Any) -> Any:
    """Construct the New Tiedtke column-batch seam (keyword-only)."""

    return seam_class()(**values)


__all__ = [
    "CU_PHYSICS_NTIEDTKE",
    "DECLARED_DIFFERENCES",
    "NTIEDTKE_ENGINE_SOURCES",
    "NTIEDTKE_ROW_SCHEMA",
    "NtiedtkeRowError",
    "measured_row_match",
    "SCALE_AWARE_ENV",
    "engine_takes_column_dx",
    "run_ntiedtke_column_batch",
    "scale_aware_requested",
    "scale_awareness_receipt",
    "seam_class",
    "verify_ntiedtke_sources",
]
