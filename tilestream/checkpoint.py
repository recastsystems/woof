"""Writing and READING BACK a wrfrst restart when the domain is host-resident.

``woof/io/restart.py`` serializes a checkpoint by walking a live
``DomainState`` and a live ``PhysicsDriver``.  A streamed run has neither at
full domain shape: the authoritative state is a pinned
:class:`tilestream.hoststore.HostDomainStore`, and the only ``DomainState``
that exists is one halo tile.  This module closes that gap in both
directions -- a checkpoint written FROM the store, and a checkpoint read back
INTO one -- and it does so without a second copy of restart.py's
classification tables.

THE FACT THAT MAKES THIS CHEAP
------------------------------
``tilestream.physics_inventory.carrier_manifest`` is built by CALLING
``restart.state_manifest`` / ``restart._scratch_manifest`` /
``restart._driver_manifest``.  So the streaming carrier set and the restart
archive's member set are not merely compatible, they are THE SAME SET,
computed by the same enforced code.  Measured at ``full(real74) +KF``,
48x40x49: 155 carriers, 155 archive members, symmetric difference EMPTY.

One qualification, added with ``restart.CARRIED_SCRATCH_SLOTS``: the store is
the CROSS-STEP set and the archive is the SERIALIZED one, and those differ by
exactly that class -- the two ``nwp_diagnostics`` tracker windows, two
``(ny, nx)`` planes, present only under ``nwp_diagnostics = 1``.  A window
means "max since that consumer last looked" and no file knows when the
consumer will next look, so it is carried and never written.  The write
filters through ``physics_inventory.checkpointed_carriers`` and the read
zeroes the complement.

The store therefore already holds every array a restart needs, keyed by the
archive's own member names (``state/thp``, ``scratch/mp_rainnc``,
``driver/pbl_tendencies/ru``, ``fields/ust``, ``cumulus/w0avg``).  Nothing
has to be collected, converted or renamed at checkpoint time; the write is
the store's own bytes plus a header.

WHAT IS NOT IN THE STORE, AND WHY IT CANNOT BE FAKED FROM A TILE
----------------------------------------------------------------
The header is.  Four of its fields are functions of the DOMAIN and a tile
cannot supply them:

``config``
    Trivially the domain's -- but ``_configuration_fingerprint`` folds
    ``nx``/``ny`` in, so a tile's config hashes differently.

``setup_fingerprint``
    SHA-256 over ``STATE_SETUP_ARRAYS`` -- the base state, the vertical
    coordinate, the map factors, the Coriolis parameter, the terrain -- plus
    ``STATE_SETUP_SCALARS`` and the whole lateral-boundary forcing table.
    Domain-shaped by construction.  MEASURED at 48x40 vs a 32x24 tile:
    ``37e63e71...`` vs ``9a2abf9c...``.

``physics_setup`` / ``physics_setup_fingerprint``
    Resolved scheme identity.  Almost all of it is config- and
    asset-derived and a tile computes it correctly, but TWO members are
    domain-shaped geography: the radiation callable's ``latitude_deg`` /
    ``longitude_deg`` grid, and -- under ``sf_surface_physics=4`` -- Noah-MP's
    ``NoahmpSolarGeometry`` lat/lon.  MEASURED: computing the identity from a
    tile driver leaves exactly those three digests differing at
    ``full+MYNN+Noah-MP`` and two shape entries at every radiation rung; with
    both grids substituted the identity is IDENTICAL at every rung tested
    (mp10, RRTMGP, full+KF, full+MYNN+Noah-MP).

``elapsed_seconds`` and the driver call counts
    The scalar carriers.  ``physics_inventory.carrier_scalars`` already
    carries them through a tiled sweep because every physics cadence test
    reads them; the header is the same three numbers.

This is why the header cannot be built from the tile that happens to be in
the buffer.  It is ALSO why building it from a tile is a genuinely dangerous
shortcut rather than a merely wrong one: a checkpoint whose setup fingerprint
was computed from a tile would be REJECTED by a monolithic resume (fail
closed, fine) but ACCEPTED by another streamed resume that made the same
mistake -- so the fingerprint would silently stop protecting against resuming
onto different terrain, a different projection or a different base state.
:func:`tilestream.test_io.negative_tile_fingerprint` is that control.

:class:`DomainSetup` is the answer: capture the domain's setup ONCE, at
preparation, when it is known, and carry it beside the store.  It is INPUT,
exactly like the geography the tile gather needs, and it is small -- MEASURED
0.780 B/cell at 48x40x49, 0.747 at 96x80x49 and 0.735 at 512x496x49, against
233.6 B/cell of carriers at ``full(real74) +KF``, i.e. **0.31% of the store**.
The asymptote is exact and worth knowing: nine of the twenty-nine setup
arrays are horizontal (``mub2d ht msft msfu msfv f e sina cosa``) and the
other twenty are 1-D columns, so the cost is ``9 * 4 / nz`` B/cell = 0.735 at
nz=49 and everything above that is the vertical arrays amortising away.  A
run that streams a domain too large for VRAM can hold it three hundred times
over.

THE ROUND TRIP IS THE POINT
---------------------------
A checkpoint that only its own writer can read is not a checkpoint.  The
files this module writes are ordinary v5 woof restarts: byte-for-byte the
same members in the same order with the same header keys as
``restart.write_restart`` produces from a resident state, and
``restart.restore_restart`` reads them with no changes at all.
:mod:`tilestream.test_io` proves all four legs -- streamed->streamed,
streamed->monolithic, monolithic->streamed, and the uninterrupted control --
bit-exact.

WHAT THIS MODULE DELIBERATELY DOES NOT DO
-----------------------------------------
It does not re-derive the header layout.  :func:`store_restart_header`
composes the same dict ``restart.write_restart`` composes, from the same
functions, and :func:`tilestream.test_io.case_header_equivalence` compares
the result key-for-key and member-for-member against a real
``write_restart`` at a size where both are possible.  If restart.py grows a
header key or bumps ``RESTART_FORMAT_VERSION``, that comparison fails --
which is the only drift guard worth having, because a guard that reads the
other module's source would pass while the semantics moved underneath it.
"""

from __future__ import annotations

import contextlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from woof.io import restart as _restart
from woof.state_serialization_contract import (
    STATE_SETUP_ARRAYS, STATE_SETUP_SCALARS,
)


__all__ = [
    "CheckpointError",
    "DomainSetup",
    "domain_physics_setup",
    "driver_restart_header",
    "read_store_restart",
    "restart_scalars",
    "store_restart_header",
    "write_store_restart",
]


class CheckpointError(RuntimeError):
    """A store-side checkpoint could not be written or read."""


# --------------------------------------------------------------------------
# the domain's setup, captured once
# --------------------------------------------------------------------------

#: Non-array attributes ``restart`` reads off a state while composing or
#: validating a header.  ``lateral_boundaries`` drives the setup digest's LBC
#: tail and the ``interval_at`` admissibility check;
#: ``_lateral_boundary_device`` decides ``root_external_lbc_clock_identity``;
#: ``_nest_restart_classification`` switches the setup digest to the nest
#: form.  A :class:`DomainSetup` that omitted any of them would compute a
#: DIFFERENT fingerprint from the state it was captured from, which is the
#: one thing it must never do.
_SETUP_PASSTHROUGH = ("lateral_boundaries", "_lateral_boundary_device",
                      "_nest_restart_classification")


class DomainSetup:
    """The domain's setup and resolved physics identity, as host data.

    Everything a restart header needs and the carrier store does not hold.
    Captured once, at preparation, from whatever built the domain -- a
    resident ``DomainState`` at a size that fits, or the preprocessor's
    geography at a size that does not.  It is INPUT: nothing in a forecast
    ever writes it, so a streamed run holds one copy for the whole run and
    never gathers, scatters or re-derives it.

    The object deliberately duck-types as a ``DomainState`` for exactly the
    three functions that need it -- ``restart.setup_fingerprint``,
    ``restart.setup_core_fingerprint`` and
    ``restart.root_external_lbc_clock_identity`` -- by carrying their inputs
    under their own attribute names.  It is not a state and does not pretend
    to be one anywhere else.
    """

    def __init__(self, cfg, arrays: Mapping[str, np.ndarray],
                 scalars: Mapping[str, Any], physics_setup: dict | None,
                 passthrough: Mapping[str, Any]):
        self.cfg = cfg
        missing = [name for name in STATE_SETUP_ARRAYS if name not in arrays]
        if missing:
            raise CheckpointError(
                f"DomainSetup is missing setup arrays {missing}; "
                "restart.setup_fingerprint reads every name in "
                "STATE_SETUP_ARRAYS unconditionally and would raise")
        for name in STATE_SETUP_ARRAYS:
            setattr(self, name, np.ascontiguousarray(arrays[name]))
        for name in STATE_SETUP_SCALARS:
            setattr(self, name, scalars[name])
        for name in _SETUP_PASSTHROUGH:
            setattr(self, name, passthrough.get(name))
        self.physics_setup = physics_setup

    @classmethod
    def capture(cls, state, cfg, *, physics_setup: dict | None = None,
                latitude=None, longitude=None) -> "DomainSetup":
        """Capture ``state``'s setup at the shape ``cfg`` describes.

        ``state`` must be at the DOMAIN's extents.  ``physics_setup``
        overrides the identity (pass one from :func:`domain_physics_setup`
        when the state is a tile-shaped stand-in); by default it is read from
        ``state`` directly, which is correct when ``state`` really is the
        domain.  ``latitude``/``longitude`` are the domain's radiation grid
        and are only consulted when the identity is derived from a
        tile-shaped ``state``.
        """
        arrays = {name: _restart._host(getattr(state, name))
                  for name in STATE_SETUP_ARRAYS}
        scalars = {name: getattr(state, name) for name in STATE_SETUP_SCALARS}
        passthrough = {name: getattr(state, name, None)
                       for name in _SETUP_PASSTHROUGH}
        if physics_setup is None:
            physics_setup = domain_physics_setup(
                state, cfg, latitude=latitude, longitude=longitude)
        return cls(cfg, arrays, scalars, physics_setup, passthrough)

    # -- the three things restart asks a state for -----------------------

    @property
    def setup_fingerprint(self) -> str:
        return _restart.setup_fingerprint(self)

    @property
    def setup_core_fingerprint(self) -> str:
        return _restart.setup_core_fingerprint(self)

    def lbc_clock_identity(self, cfg=None) -> str | None:
        return _restart.root_external_lbc_clock_identity(self, cfg or self.cfg)

    # -- cost, so the claim "it is small" is a number --------------------

    @property
    def nbytes(self) -> int:
        return sum(int(getattr(self, name).nbytes)
                   for name in STATE_SETUP_ARRAYS)

    def bytes_per_cell(self, cfg=None) -> float:
        cfg = cfg or self.cfg
        return self.nbytes / float(cfg.nx * cfg.ny * cfg.nz)

    def output_statics(self, nz: int, ny: int, nx: int) -> dict:
        """The setup fields a wrfout frame carries, at output shape.

        ``PB`` and ``PHB`` are the 1-D base state broadcast to a full column
        field -- the device frame materialises them the same way, so the
        on-disk layout is identical with and without terrain -- and the rest
        are the setup arrays under their WRF names.  Materialised ONCE per
        run: none of them is a carrier, none changes between frames, and
        re-broadcasting them per frame is 1.3 GB of pointless host traffic at
        1024^2 (MEASURED as part of the 29.6 ms host derive).
        """
        pb = np.asarray(self.pb)
        phb = np.asarray(self.phb)
        pb3 = pb if pb.ndim == 3 else pb[:, None, None]
        phb3 = phb if phb.ndim == 3 else phb[:, None, None]
        statics = {
            "PB": np.ascontiguousarray(np.broadcast_to(pb3, (nz, ny, nx))),
            "PHB": np.ascontiguousarray(
                np.broadcast_to(phb3, (nz + 1, ny, nx))),
            "MUB": np.ascontiguousarray(np.asarray(self.mub2d)),
            "HGT": np.ascontiguousarray(np.asarray(self.ht)),
            "ZNU": np.ascontiguousarray(np.asarray(self.znu)),
            "ZNW": np.ascontiguousarray(np.asarray(self.znw)),
            "P_TOP": np.asarray(self.p_top, dtype=np.float32),
        }
        if self.cfg.mp_physics == 55:
            statics.update(C1H=self.c1h, C2H=self.c2h)
        return statics


def _like(reference, value):
    """``value`` as an array of ``reference``'s module and dtype."""
    dtype = getattr(reference, "dtype", None)
    if type(reference).__module__.startswith("cupy"):
        import cupy as cp
        return cp.asarray(_restart._host(value), dtype=dtype)
    return np.asarray(_restart._host(value), dtype=dtype)


@contextlib.contextmanager
def _substituted_geography(driver, latitude, longitude):
    """Temporarily point a tile driver's lat/lon grids at the DOMAIN's.

    The two sites are the radiation callable's ``latitude_deg``/
    ``longitude_deg`` and Noah-MP's ``NoahmpSolarGeometry``.  Both are pure
    SETUP -- read by the scheme, never written by it -- so swapping them for
    the duration of an identity computation changes no state and touches no
    trajectory.  It is the same substitution the tile gather performs per
    tile for every other piece of horizontally varying geography; the
    difference is only that these two grids live on scheme objects rather
    than on the ``DomainState``, which is exactly why
    ``physics_inventory.assert_tileable_geography`` names them separately.

    Restored in a ``finally`` because leaving a tile driver pointing at
    domain-shaped grids would make its next radiation call index out of
    bounds.
    """
    if driver is None or (latitude is None and longitude is None):
        yield
        return
    saved: list[tuple[Any, str, Any]] = []
    for holder in (getattr(driver, "radiation_callable", None),
                   getattr(driver, "noahmp_geometry", None)):
        if holder is None or getattr(holder, "latitude_deg", None) is None:
            continue
        for attr, value in (("latitude_deg", latitude),
                            ("longitude_deg", longitude)):
            if value is None:
                continue
            current = getattr(holder, attr)
            saved.append((holder, attr, current))
            # Match the array module as well as the dtype: the radiation
            # adapters keep their grids on the device and ``np.asarray`` on a
            # cupy array raises rather than transferring.
            setattr(holder, attr, _like(current, value))
    try:
        yield
    finally:
        for holder, attr, value in reversed(saved):
            setattr(holder, attr, value)


def domain_physics_setup(state, cfg, *, latitude=None, longitude=None) -> dict:
    """The resolved physics identity AT DOMAIN SHAPE.

    ``state`` supplies the scheme objects -- their classes, their resolved
    cadences, their packaged-asset digests -- and may be a tile-shaped
    stand-in, which is what a streamed run has.  ``cfg`` must be the
    DOMAIN's config, and ``latitude``/``longitude`` the domain's radiation
    grid; without them a tile-derived identity differs from the domain's in
    the radiation grid shape (every radiation rung) and in Noah-MP's solar
    geometry digests (``sf_surface_physics=4``), and a checkpoint carrying it
    is refused by a monolithic resume.
    """
    with _substituted_geography(getattr(state, "physics", None),
                                latitude, longitude):
        return _restart.physics_setup_identity(state, cfg)


# --------------------------------------------------------------------------
# the header
# --------------------------------------------------------------------------

def restart_scalars(header: Mapping[str, Any]) -> dict:
    """``physics_inventory.carrier_scalars`` shape, read out of a header.

    The clock is a carrier and so are the three driver counters: a resumed
    sweep that did not restore ``microphysics_updates`` would re-run NSSL
    mp18's first-call branch and would let RRTMGP distrust effective radii it
    should trust.  Returning them in ``set_carrier_scalars``'s own shape is
    what makes ``set_carrier_scalars(tile_state, restart_scalars(header))``
    the whole of a resume's scalar restore.
    """
    scalars: dict[str, Any] = {
        "elapsed_seconds": float(header["elapsed_seconds"])}
    if "wrf_mass_carriers_cold" in header:
        scalars["wrf_mass_carriers_cold"] = _restart.validate_strict_mass_cold(header)
    if "scratch/wrf_exact_muts" in header.get("array_manifest", {}):
        mass = header.get("strict_mass_carriers", {})
        scalars["wrf_mass_carriers_ready"] = bool(mass.get("ready", True))
        scalars["wrf_start_diagnostics_carried"] = bool(
            mass.get("start_diagnostics_carried", False))
    driver = header.get("driver")
    if driver is not None:
        if 'radiation_owned' in driver:
            scalars['radiation_owned']=driver['radiation_owned']
        if "wrf_rthraten_coupled" in driver:
            scalars["wrf_rthraten_coupled"] = bool(driver["wrf_rthraten_coupled"])
        scalars["call_counts"] = {str(key): int(value)
                                  for key, value in
                                  driver["call_counts"].items()}
        scalars["ysu_nan_guard_fires"] = int(driver["ysu_nan_guard_fires"])
        scalars["microphysics_updates"] = int(driver["microphysics_updates"])
        scalars["swdd_from_scheme"] = bool(driver.get("swdd_from_scheme", False))
        if "fire" in driver:
            from tilestream.sfire import GRID_CLOCKS
            scalars["fire_clocks"] = {key: driver["fire"]["grid"][key] for key in GRID_CLOCKS}
            scalars["fire_header"] = {key: driver[key] for key in
                                     ("fire", "fire_setup_identity", "fire_setup_fingerprint")}
        # The carrier contract's records travel with the clock for the same
        # reason the counters do: the freshness law
        # (woof/core/radiation_carriers.py) reads produced-at stamps
        # immediately before every land-surface call, and a resumed sweep
        # whose buffers keep their build-time stamps refuses the run as
        # stale on the first consuming step.  A header WITHOUT them is a
        # pre-contract checkpoint; the marker asks set_carrier_scalars to
        # apply restart.py's own legacy remedy (_restore_carriers): rerun
        # the producers once before anything consumes them.
        carriers = driver.get("carriers")
        if carriers is not None:
            scalars["carriers"] = {
                str(name): {
                    "source": row["source"],
                    "last_update_model_time":
                        (None if row["last_update_model_time"] is None
                         else float(row["last_update_model_time"]))}
                for name, row in carriers.items()}
            policy = driver.get("surface_radiation_policy")
            if policy is not None:
                scalars["surface_radiation_policy"] = str(policy)
        else:
            scalars["carriers_need_producer_refresh"] = True
    return scalars


def driver_restart_header(scalars: Mapping[str, Any]) -> dict | None:
    """Encode every driver scalar carrier for either streamed writer."""
    driver_header = None
    if "call_counts" in scalars:
        driver_header = {
            "call_counts": {str(key): int(value)
                            for key, value in scalars["call_counts"].items()},
            "ysu_nan_guard_fires": int(scalars["ysu_nan_guard_fires"]),
            "microphysics_updates": int(scalars["microphysics_updates"]),
            "swdd_from_scheme": bool(scalars.get("swdd_from_scheme", False)),
        }
        if 'radiation_owned' in scalars:
            driver_header['radiation_owned']=scalars['radiation_owned']
        if "wrf_rthraten_coupled" in scalars:
            driver_header["wrf_rthraten_coupled"] = bool(scalars["wrf_rthraten_coupled"])
        # Key-for-key what restart.write_restart's driver header carries
        # (woof/io/restart.py, "carriers"/"surface_radiation_policy"):
        # the carrier contract is the provenance a checkpoint's arrays
        # cannot carry, and test_io.case_header_equivalence compares the
        # two headers as values, not as key subsets.
        if "carriers" in scalars:
            driver_header["carriers"] = {
                str(name): {
                    "source": row["source"],
                    "last_update_model_time":
                        (None if row["last_update_model_time"] is None
                         else float(row["last_update_model_time"]))}
                for name, row in scalars["carriers"].items()}
        if "surface_radiation_policy" in scalars:
            driver_header["surface_radiation_policy"] = str(
                scalars["surface_radiation_policy"])
    if driver_header is not None and "fire_header" in scalars:
        driver_header.update(scalars["fire_header"])
    return driver_header


def store_restart_header(arrays: Mapping[str, np.ndarray],
                         scalars: Mapping[str, Any],
                         setup: DomainSetup, cfg, *,
                         run_trackers=None) -> dict:
    """Compose the v5 restart header for an out-of-core domain.

    Field for field what ``restart.write_restart`` composes, from the same
    functions, with the two domain-shaped inputs taken from ``setup``
    instead of from a state that does not exist.  ``arrays`` is the store's
    carrier map; only its shapes and dtypes are read here.
    """
    driver_header = driver_restart_header(scalars)
    physics_setup = setup.physics_setup
    if physics_setup is None:
        raise CheckpointError(
            "DomainSetup carries no resolved physics identity; capture it "
            "with domain_physics_setup(tile_state, domain_cfg, "
            "latitude=..., longitude=...) so the header binds the DOMAIN's "
            "radiation grid rather than a tile's")
    header = {
        "format_version": _restart.RESTART_FORMAT_VERSION,
        "case": cfg.case,
        "created": datetime.now(timezone.utc).isoformat(),
        "producer": _restart.producer_identity(),
        "elapsed_seconds": _restart._admissible_elapsed_seconds(
            scalars["elapsed_seconds"], "streamed restart write"),
        "config": _restart.configuration_echo(cfg),
        "setup_fingerprint": setup.setup_fingerprint,
        "physics_setup": physics_setup,
        "physics_setup_fingerprint": _restart._json_sha256(physics_setup),
        "driver": driver_header,
        "run_trackers": (None if run_trackers is None else dict(run_trackers)),
        "array_manifest": {
            key: {"shape": list(np.asarray(arrays[key]).shape),
                  "dtype": str(np.asarray(arrays[key]).dtype)}
            for key in sorted(arrays)},
    }
    if scalars.get("wrf_mass_carriers_cold", False):
        header["wrf_mass_carriers_cold"] = True
        _restart.validate_strict_mass_cold(header)
    lbc_clock = setup.lbc_clock_identity(cfg)
    if lbc_clock is not None:
        header["root_external_lbc_clock"] = lbc_clock
    return header


# --------------------------------------------------------------------------
# write
# --------------------------------------------------------------------------

def _store_arrays(store) -> dict[str, np.ndarray]:
    inner = getattr(store, "arrays", None)
    source = inner if isinstance(inner, dict) else store
    if not isinstance(source, Mapping):
        raise CheckpointError(
            f"expected a HostDomainStore or a {{key: array}} mapping, got "
            f"{type(store).__name__}")
    return {str(key): np.asarray(value) for key, value in source.items()
            if value is not None}


def write_store_restart(path, store, scalars, setup: DomainSetup, cfg, *,
                        run_trackers=None, verify: bool = True) -> Path:
    """Write a v5 woof restart straight out of the pinned host store.

    No device traffic at all: the carriers are already host arrays and
    ``numpy.savez`` reads them where they lie.  The archive is published the
    way ``restart.write_restart`` publishes -- write to ``.tmp``, ``fsync``,
    ``os.replace`` -- because the durability discipline is the point of a
    checkpoint, and a checkpoint that is only in the page cache when the
    machine dies is the file that was not written.

    ``verify=True`` re-reads the header off the published file and checks it
    round-tripped through JSON with the member set intact.  It costs one
    zip-directory read (MEASURED 1.8 ms at 155 members) and it is the
    difference between "the write returned" and "the file is a restart".
    """
    from tilestream import physics_inventory as _physinv

    path = Path(path)
    arrays = _store_arrays(store)
    if not arrays:
        raise CheckpointError("refusing to write an empty checkpoint")
    # The store is the CROSS-STEP set; an archive holds the SERIALIZED one,
    # and since restart.CARRIED_SCRATCH_SLOTS those are no longer the same
    # set.  Writing the difference would produce a file woof's own reader
    # refuses ("restart carries non-serializable scratch slot"), and would
    # restore a tracker window measured against a boundary that no longer
    # exists.  See physics_inventory.is_checkpointed.
    arrays = _physinv.checkpointed_carriers(arrays)
    header = store_restart_header(arrays, scalars, setup, cfg,
                                  run_trackers=run_trackers)
    # Member order is the archive's own: header first, then the carriers in
    # sorted key order.  It costs nothing to match and it makes a streamed
    # checkpoint and a monolithic one comparable member by member in file
    # order, which is how test_io proves they are the same archive.
    payload = {_restart._HEADER_KEY: np.frombuffer(
        json.dumps(header, allow_nan=False).encode("utf-8"), dtype=np.uint8)}
    for key in sorted(arrays):
        payload[key] = arrays[key]
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    try:
        with temp.open("wb") as stream:
            np.savez(stream, **payload)
        _restart.fsync_file(temp)
        os.replace(temp, path)
    except BaseException:
        temp.unlink(missing_ok=True)
        raise
    if verify:
        published = _restart.read_restart_header(path)
        if set(published["array_manifest"]) != set(arrays):
            raise CheckpointError(
                f"published checkpoint {path} lost members on the way to "
                "disk")
    return path


# --------------------------------------------------------------------------
# read back
# --------------------------------------------------------------------------

def read_store_restart(path, store, setup: DomainSetup, cfg, *,
                       fill: bool = True) -> dict:
    """Validate a restart against this domain and load it into ``store``.

    Every refusal ``restart._validate_restart`` performs that a host store
    can perform is performed here, against restart.py's own functions:
    format version, the required header keys, the config echo, the external
    LBC clock semantic, the NSSL2 contract, the physics setup identity, the
    setup fingerprint, the archive's self-consistency, and -- the streaming
    analogue of restart's per-attribute shape checks -- the archive's member
    set and shapes against the STORE's.

    Returns the scalar carriers in ``set_carrier_scalars`` shape.  With
    ``fill=False`` nothing is copied and only the validation runs, which is
    what a resume wants when it is choosing between checkpoints.
    """
    path = Path(path)
    header, stored = _restart._load_restart(path, with_arrays=True)
    _restart.validate_strict_mass_cold(header)

    missing_header = sorted({
        "format_version", "config", "setup_fingerprint", "physics_setup",
        "physics_setup_fingerprint", "array_manifest", "elapsed_seconds",
    } - set(header))
    if missing_header:
        raise _restart.RestartMismatchError(
            f"restart file {path} header is missing {missing_header}")
    version = header.get("format_version")
    _restart.require_readable_format_version(version, path)
    _restart._require_current_algorithm_identities(header, cfg, path)
    _restart._require_config_match(header["config"], cfg, path)

    live_lbc_clock = setup.lbc_clock_identity(cfg)
    if live_lbc_clock is not None:
        stored_clock = header.get("root_external_lbc_clock",
                                  _restart.ROOT_EXTERNAL_LBC_CLOCK_LEGACY)
        if stored_clock != live_lbc_clock:
            raise _restart.RestartMismatchError(
                f"restart file {path} was integrated under the "
                f"root_external_lbc_clock semantic {stored_clock!r} but this "
                f"run streams {live_lbc_clock!r}: the root external Davies "
                "dtbc consumption differs, so resuming would splice two "
                "different trajectories")
    _restart._require_nssl2_restart_contract(header, cfg, path)

    stored_identity = header["physics_setup"]
    if _restart._json_sha256(stored_identity) != \
            header["physics_setup_fingerprint"]:
        raise _restart.RestartMismatchError(
            f"restart file {path} physics setup fingerprint does not match "
            "its own stored identity")
    if setup.physics_setup is not None and \
            _restart._json_sha256(setup.physics_setup) != \
            header["physics_setup_fingerprint"]:
        raise _restart.RestartMismatchError(
            f"restart file {path} was written under a different resolved "
            "physics setup (scheme callables, radiation grid/calendar, "
            "land-surface tables or packaged assets); rebuild the identical "
            "preparation before resuming")
    elapsed = _restart._admissible_elapsed_seconds(
        header["elapsed_seconds"], f"restart file {path}")
    if header["setup_fingerprint"] != setup.setup_fingerprint:
        raise _restart.RestartMismatchError(
            f"restart file {path} was written on a different model setup "
            "(base state / coordinates / map factors / terrain fingerprint "
            "mismatch); rebuild the identical preparation before restoring")

    manifest = header["array_manifest"]
    if not isinstance(manifest, dict):
        raise _restart.RestartMismatchError(
            f"restart file {path} has a malformed array manifest")
    if set(stored) != set(manifest):
        raise _restart.RestartMismatchError(
            f"restart file {path} arrays disagree with its own manifest "
            f"(missing {sorted(set(manifest) - set(stored))}, extra "
            f"{sorted(set(stored) - set(manifest))})")

    from tilestream import physics_inventory as _physinv

    whole_store = _store_arrays(store)
    # Compared against the CHECKPOINTED subset for the same reason the
    # writer emits only that subset; the rest is zeroed below.
    arrays = _physinv.checkpointed_carriers(whole_store)
    # The checkpoint-only driver carriers are tolerant both ways, as the
    # resident reader is (physics_inventory.is_absent_tolerant).
    stored = {key: value for key, value in stored.items()
              if key in arrays or not _physinv.is_absent_tolerant(key)}
    arrays = {key: value for key, value in arrays.items()
              if key in stored or not _physinv.is_absent_tolerant(key)}
    if set(stored) != set(arrays):
        missing = sorted(set(arrays) - set(stored))
        extra = sorted(set(stored) - set(arrays))
        raise _restart.RestartMismatchError(
            f"restart file {path} carrier inventory does not match this "
            f"streamed run's store (the file is missing {missing[:8]}, and "
            f"carries {extra[:8]} the store has no home for).  A partial "
            "restore would resume with dirty carriers, which is the failure "
            "the streaming inventory check exists to prevent")
    for key, target in arrays.items():
        host = stored[key]
        if tuple(host.shape) != tuple(target.shape) or \
                host.dtype != target.dtype:
            raise _restart.RestartMismatchError(
                f"restart member {key} is {tuple(host.shape)} "
                f"{host.dtype} but this domain's store slot is "
                f"{tuple(target.shape)} {target.dtype}")

    if fill:
        for key, target in arrays.items():
            target[...] = stored[key]
        # A restart starts the tracker windows EMPTY (restart.py:
        # CARRIED_SCRATCH_SLOTS).  A resident restore gets that for free --
        # a fresh state allocates them zeroed and nothing writes them -- but
        # a store is restored IN PLACE, so it has to be said out loud or the
        # resumed run keeps a window measured before the checkpoint.
        for target in _physinv.carried_only_carriers(whole_store).values():
            target[...] = 0.0
    scalars = restart_scalars(header)
    if scalars["elapsed_seconds"] != elapsed:
        raise CheckpointError(
            "restart clock disagreed with itself between the header and the "
            "admissibility check")
    return scalars
