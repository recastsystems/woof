"""Versioned, single-domain orchestration. Importing this module needs no CUDA.

Arrays are live device arrays in (z, y, x) order; see docs/domain-api.md.
The numerical work and restart contract belong to the existing engine.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
import math
import operator
from pathlib import Path
import tempfile
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from woof.config import RunConfig
    from woof.core.state import DomainState
    from woof.core.physics import PhysicsDriver
    from woof.static.projection import ProjectedGrid

API_VERSION = "1.0"


@dataclass(frozen=True)
class WarmBubble:
    """Cosine-squared theta perturbation: K, radii/centre in metres.

    x/y are measured from the domain centre; z is height above flat ground.
    Applied during balanced construction, before physics initialization.
    """

    amplitude: float = 3.0
    radii: tuple[float, float, float] = (10000.0, 10000.0, 1500.0)
    centre: tuple[float, float, float] = (0.0, 0.0, 1500.0)

    def __post_init__(self) -> None:
        if (len(self.radii) != 3 or len(self.centre) != 3
                or not all(math.isfinite(v) for v in
                           (self.amplitude, *self.radii, *self.centre))
                or min(self.radii) <= 0):
            raise ValueError("bubble needs finite amplitude/centre and three positive radii")


@dataclass(frozen=True)
class DomainSnapshot:
    """Owned host restart bytes and the RunConfig used to write them.

    This is an in-process checkpoint, governed by the engine restart format,
    not a cross-version pickle protocol. Setup must match at restoration.
    """

    payload: bytes
    config: RunConfig
    api_version: str = API_VERSION
    radiation_next_step: bool = False


def _array(value: Any) -> bool:
    return hasattr(value, "shape") and hasattr(value, "dtype")


def _start_time(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    if not isinstance(value, datetime):
        raise TypeError("start_time must be a datetime or None")
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _placement_grid(nx: int, ny: int, dx: float, dy: float,
                    center_lat: float | None, center_lon: float | None
                    ) -> ProjectedGrid | None:
    if center_lat is None and center_lon is None:
        return None
    if center_lat is None or center_lon is None:
        raise ValueError("placement needs both center_lat and center_lon to locate the atmosphere")
    if (not math.isfinite(center_lat) or not -90 < center_lat < 90
            or not math.isfinite(center_lon) or not -180 <= center_lon <= 180):
        raise ValueError("placement needs finite -90 < center_lat < 90 and -180 <= center_lon <= 180; Mercator cannot place a pole")
    from woof.static.projection import MercatorGrid
    return MercatorGrid(ref_lat=center_lat, ref_lon=center_lon,
                        truelat1=center_lat, truelat2=center_lat,
                        stand_lon=center_lon, dx=dx, dy=dy,
                        e_we=nx + 1, e_sn=ny + 1)


def _sounding(sounding: Any, z: Any, p: Any, theta: Any, qv: Any,
              u: Any, v: Any, p_surf: float | None, ztop: float) -> tuple:
    """Normalize explicit SI columns; input_sounding alone uses hPa/g kg-1."""
    import numpy as np
    from woof.core import constants as c

    if sounding is not None:
        if any(a is not None for a in (z, p, theta, qv, u, v)):
            raise ValueError("pass a sounding or separate columns, not both")
        if isinstance(sounding, (str, Path)):
            rows = [line.split("#", 1)[0].split() for line in
                    Path(sounding).read_text(encoding="utf-8").splitlines()]
            rows = [list(map(float, row)) for row in rows if row]
            if not rows or len(rows[0]) != 3 or any(len(row) != 5 for row in rows[1:]):
                raise ValueError("input_sounding needs p(hPa),theta(K),qv(g/kg), then z,theta,qv,u,v rows")
            ps, th0, q0 = rows[0]
            data = np.asarray(rows[1:], dtype=float)
            if data.shape[0] < 2:
                raise ValueError("input_sounding needs at least two height rows")
            z, theta, qv, u, v = data.T
            qv = qv / 1000.0
            if z[0] > 0:
                z = np.r_[0.0, z]
                theta, qv = np.r_[th0, theta], np.r_[q0 / 1000.0, qv]
                u, v = np.r_[u[0], u], np.r_[v[0], v]
            if p_surf is None:
                p_surf = ps * 100.0
        elif isinstance(sounding, Mapping):
            unknown = set(sounding) - {"z", "p", "theta", "qv", "u", "v", "p_surf"}
            if unknown:
                raise ValueError(f"unknown sounding columns: {sorted(unknown)}")
            z, p, theta, qv, u, v = (sounding.get(k) for k in
                                        ("z", "p", "theta", "qv", "u", "v"))
            if p_surf is None:
                p_surf = sounding.get("p_surf")
        else:
            raise TypeError("sounding must be a column mapping or input_sounding path")
    if (z is None) == (p is None) or theta is None:
        raise ValueError("sounding needs theta and exactly one of z(m) or p(Pa)")
    axis = np.asarray(z if z is not None else p, dtype=float)
    th = np.asarray(theta, dtype=float)
    if axis.ndim != 1 or axis.size < 2 or th.shape != axis.shape:
        raise ValueError("sounding columns must be equal one-dimensional arrays with at least two rows")
    cols = [th] + [np.zeros_like(th) if a is None else np.asarray(a, dtype=float)
                   for a in (qv, u, v)]
    if any(a.shape != axis.shape or not np.isfinite(a).all() for a in [axis, *cols]):
        raise ValueError("sounding columns must have equal shapes and finite values")
    if (th <= 0).any() or (cols[1] < 0).any():
        raise ValueError("theta must be positive and qv nonnegative")
    if p is not None:
        if (axis <= 0).any() or not (np.diff(axis) < 0).all():
            raise ValueError("pressure must be positive and strictly decrease from the surface")
        if p_surf is None:
            p_surf = float(axis[0])
        if float(p_surf) != float(axis[0]):
            raise ValueError("pressure sounding must start at p_surf to anchor its surface height")
        exner = (axis / c.P0) ** c.RCP
        thv = th * (1 + c.RVOVRD * cols[1]) / (1 + cols[1])
        axis = np.r_[0.0, np.cumsum(-c.CP / c.G * 0.5 *
                                   (thv[1:] + thv[:-1]) * np.diff(exner))]
    if (axis < 0).any() or not (np.diff(axis) > 0).all():
        raise ValueError("heights must be nonnegative and strictly increase")
    if axis[0] != 0 or axis[-1] < ztop:
        raise ValueError("sounding must cover the surface through ztop; extrapolation changes the requested atmosphere")
    ps = 1.0e5 if p_surf is None else float(p_surf)
    if not math.isfinite(ps) or ps <= 0:
        raise ValueError("p_surf must be finite and positive")
    return axis, *cols, ps


class Domain:
    """One state, immutable configuration and optional attached physics driver.

    Use the factories below. ``state`` and ``cfg`` remain accessible for engine
    integration; normal construction, stepping and field access need no internals.
    """

    def __init__(self, state: DomainState, cfg: RunConfig, *,
                 grid: ProjectedGrid | None = None,
                 start_time: datetime | None = None) -> None:
        if grid is not None and (grid.e_we - 1, grid.e_sn - 1, grid.dx, grid.dy) != (
                cfg.nx, cfg.ny, cfg.dx, cfg.dy):
            raise ValueError("grid dimensions/spacing must match cfg or radar would place the state on different cells")
        self.state = state
        self.cfg = cfg
        self.grid = grid
        self.start_time = _start_time(start_time)
        self._radiation_next_step = False

    @classmethod
    def from_state(cls, state: DomainState, cfg: RunConfig,
                   physics: PhysicsDriver | None = None, *,
                   grid: ProjectedGrid | None = None,
                   start_time: datetime | None = None) -> Domain:
        """Wrap an already prepared state without changing its arrays or clock.

        An explicit driver is attached; None preserves the existing attachment.
        The caller owns base-state, boundary and physics preparation. Optional
        grid/start_time provide location and a UTC epoch without modifying state.
        """
        from woof.config import validate_run_config
        validate_run_config(cfg)
        if (tuple(state.u.shape) != (cfg.nz, cfg.ny, cfg.nx + 1)
                or tuple(state.v.shape) != (cfg.nz, cfg.ny + 1, cfg.nx)
                or tuple(state.w.shape) != (cfg.nz + 1, cfg.ny, cfg.nx)):
            raise ValueError("state staggering does not match cfg dimensions")
        if physics is not None:
            state.physics = physics
        return cls(state, cfg, grid=grid, start_time=start_time)

    @classmethod
    def from_sounding(
            cls, sounding: Any = None, *, nx: int, ny: int, nz: int,
            dx: float, dy: float, ztop: float, dt: float,
            z: Any = None, p: Any = None, theta: Any = None, qv: Any = None,
            u: Any = None, v: Any = None, p_surf: float | None = None,
            microphysics: int | str | None = None, suite: str | None = None,
            boundary: str = "periodic", overrides: Mapping[str, Any] | None = None,
            physics_options: Mapping[str, Any] | None = None,
            bubble: WarmBubble | None = None,
            center_lat: float | None = None, center_lon: float | None = None,
            start_time: datetime | None = None) -> Domain:
        """Build a flat uniform atmosphere from SI arrays or input_sounding.

        Explicit arrays: z(m) or decreasing p(Pa), theta(K), qv(kg/kg), u/v(m/s).
        Missing qv/winds are zero. File format uses hPa and g/kg. Profiles must
        span 0..ztop. ``suite`` is a single_domain_runtime_switches profile.
        Overrides are RunConfig keywords; physics_options go to initialize_physics.
        Missing scheme tables raise the engine's error; no scheme is substituted.
        Optional centre coordinates attach the engine's MercatorGrid (dx==dy),
        true at center_lat, as metadata only: dynamics remain Cartesian.
        valid_time is start_time (naive means UTC) plus elapsed model seconds.
        """
        import numpy as np
        from woof.config import RunConfig, validate_run_config
        from woof.core.grid import make_vertical_coord, make_base_state
        from woof.core.state import init_at_rest
        from woof.core.moist import init_moist_balanced
        from woof.core.physics_inventory import physics_driver_required

        grid = _placement_grid(nx, ny, dx, dy, center_lat, center_lon)
        start_time = _start_time(start_time)
        axis, th, vapour, uw, vw, ps = _sounding(
            sounding, z, p, theta, qv, u, v, p_surf, ztop)
        if boundary not in ("periodic", "open", "specified"):
            raise ValueError("boundary must be periodic, open or specified")
        options = {}
        if suite is not None:
            from woof.physics_compat import single_domain_runtime_switches
            options.update(single_domain_runtime_switches(suite))
        if microphysics is not None:
            if isinstance(microphysics, str):
                from woof.microphysics_schemes import resolve_mp_physics
                microphysics = resolve_mp_physics(microphysics)
            if suite is not None and microphysics != options.get("mp_physics"):
                raise ValueError("microphysics conflicts with the selected suite")
            options["mp_physics"] = microphysics
        options.setdefault("mp_physics", 0)
        options["moist"] = bool(np.any(vapour) or options["mp_physics"])
        options.update(nx=nx, ny=ny, nz=nz, dx=dx, dy=dy, ztop=ztop, dt=dt,
                       p_surf=ps, run_seconds=1.0e9, open_x=boundary == "open",
                       open_y=boundary == "open", specified=boundary == "specified")
        options.update(overrides or {})
        # These determine the requested atmosphere/geometry, not a tunable switch.
        for key, expected in dict(nx=nx, ny=ny, nz=nz, dx=dx, dy=dy,
                                  ztop=ztop, dt=dt, p_surf=ps,
                                  open_x=boundary == "open", open_y=boundary == "open",
                                  specified=boundary == "specified").items():
            if options[key] != expected:
                raise ValueError(f"override {key} conflicts with the explicit domain argument")
        cfg = RunConfig(**options)
        validate_run_config(cfg)
        if np.any(vapour) and not cfg.moist:
            raise ValueError("moist=False would discard the sounding's water vapour")
        if cfg.terrain_opt or cfg.map_proj:
            raise ValueError("from_sounding builds flat unprojected columns; use from_state for terrain/projection setup")
        coord = make_vertical_coord(nz, hybrid_opt=cfg.hybrid_opt, etac=cfg.etac)
        base = make_base_state(coord, lambda heights: np.interp(heights, axis, th), ps, ztop)
        perturb = None
        if bubble is not None:
            def perturb(x: Any, heights: Any) -> Any:
                y = (np.arange(ny) + 0.5) * dy - 0.5 * ny * dy
                rx, ry, rz = bubble.radii
                cx, cy, cz = bubble.centre
                rad = np.sqrt(((x[None, None, :] - cx) / rx) ** 2
                              + ((y[None, :, None] - cy) / ry) ** 2
                              + ((heights[:, None, None] - cz) / rz) ** 2)
                return np.where(rad <= 1, bubble.amplitude * np.cos(np.pi * rad / 2) ** 2, 0)
        if cfg.moist:
            state = init_moist_balanced(cfg, coord, base,
                        lambda heights: np.interp(heights, axis, vapour), perturb)
        elif perturb is not None:
            from woof.core.state import init_theta_perturbation
            state = init_theta_perturbation(cfg, coord, base, perturb)
        else:
            state = init_at_rest(cfg, coord, base)
        import cupy as cp
        heights = state.height_half()
        state.u[...] = cp.asarray(np.interp(heights, axis, uw), dtype=state.u.dtype)[:, None, None]
        state.v[...] = cp.asarray(np.interp(heights, axis, vw), dtype=state.v.dtype)[:, None, None]
        domain = cls(state, cfg, grid=grid, start_time=start_time)
        domain.diagnostics()
        if physics_driver_required(cfg):
            from woof.core.physics import initialize_physics
            initialize_physics(state, cfg, **dict(physics_options or {}))
        if cfg.specified:
            sides = {side: domain.extract_side(side) for side in
                     ("west", "east", "south", "north")}
            domain.set_lateral_boundaries(first=sides, second=sides,
                                         start_seconds=0, end_seconds=cfg.run_seconds)
        return domain

    @property
    def time(self) -> float:
        """Elapsed model seconds, owned by dycore.step."""
        return float(self.state.elapsed_seconds)

    @property
    def valid_time(self) -> datetime | None:
        """UTC start_time plus elapsed model seconds, or None without an epoch.

        Location/time metadata stay with this wrapper during restore/copy_overlap;
        they are independent of the engine restart's numerical state.
        """
        return None if self.start_time is None else self.start_time + timedelta(seconds=self.time)

    @property
    def dt(self) -> float:
        """Current dynamics step in seconds."""
        return float(self.cfg.dt)

    @dt.setter
    def dt(self, value: float) -> None:
        from woof.config import validate_run_config
        if not math.isfinite(value) or value <= 0:
            raise ValueError("dt must be finite and positive")
        cfg = replace(self.cfg, dt=float(value))
        validate_run_config(cfg)
        self.cfg = cfg

    def step(self, n: int = 1) -> None:
        """Call the existing dycore exactly n times with its default arguments."""
        from woof.core.dycore import step
        n = operator.index(n)
        if n < 0:
            raise ValueError("step count must be nonnegative")
        for _ in range(n):
            if not self._radiation_next_step:
                step(self.state, self.cfg)
                continue
            driver = self.state.physics
            previous = driver.radiation_due_override
            driver.radiation_due_override = True
            try:
                step(self.state, self.cfg)
                self._radiation_next_step = False
            finally:
                driver.radiation_due_override = previous

    def advance(self, seconds: float) -> None:
        """Advance a duration, shortening only the last step; restore dt afterwards."""
        if not math.isfinite(seconds) or seconds < 0:
            raise ValueError("duration must be finite and nonnegative")
        target, original = self.time + seconds, self.dt
        try:
            while self.time < target:
                remaining = target - self.time
                if remaining <= max(1.0, abs(target)) * 1.e-14:
                    break
                self.dt = min(original, remaining)
                self.step()
        finally:
            self.dt = original

    def prognostic_names(self) -> tuple[str, ...]:
        """State arrays carried by a checkpoint (including auxiliary/diagnostic carriers)."""
        from woof.io.restart import classify_state_attr
        return tuple(sorted(name for name, value in vars(self.state).items()
                            if _array(value) and classify_state_attr(name)
                            in ("serialize", "checkpoint_only")))

    def moist_species(self) -> tuple[str, ...]:
        """Active qv/qc/qr and extra water mass/number/aerosol fields; excludes PBL scalars."""
        from woof.core.moist import _extra_water_species
        return tuple(n for n in ("qv", "qc", "qr", *_extra_water_species(self.state))
                     if getattr(self.state, n, None) is not None)

    def get(self, name: str) -> Any:
        """Return a live state array. Scalar/setup object access is excluded.

        Scalar cells (nz,ny,nx); u (nz,ny,nx+1); v (nz,ny+1,nx);
        w/php (nz+1,ny,nx); mup (ny,nx). Units are native engine SI units.
        """
        if name.startswith("_") or not _array(value := getattr(self.state, name, None)):
            raise KeyError(f"no state array {name!r}")
        return value

    def set(self, name: str, array: Any) -> Any:
        """Copy an exactly shaped array in place and return its live storage."""
        import cupy as cp
        target = self.get(name)
        source = cp.asarray(array, dtype=target.dtype)
        if source.shape != target.shape:
            raise ValueError(f"{name} needs shape {target.shape}, got {source.shape}")
        target[...] = source
        return target

    def to_host(self, names: Iterable[str]) -> dict[str, Any]:
        """Return owned NumPy copies, synchronized by the device readback."""
        import cupy as cp
        if isinstance(names, str):
            names = (names,)
        return {name: cp.asnumpy(self.get(name)).copy() for name in names}

    def diagnostics(self) -> dict[str, Any]:
        """Refresh EOS diagnostics and return live p/al/alt arrays."""
        from woof.core.diagnostics import update_diagnostics
        update_diagnostics(self.state, self.cfg.hypsometric_opt)
        return {name: self.get(name) for name in ("p", "al", "alt")}

    def carried_arrays(self) -> dict[str, dict[str, str]]:
        """Actual state/scratch array names mapped to unchanged restart verdicts.

        serialize/checkpoint_only are checkpointed; scratch carry crosses steps
        but is ordinarily absent from checkpoints. setup/derived_setup/infra
        and rebuild retain their engine meanings. Unknown arrays raise.
        """
        from woof.io.restart import classify_state_attr, classify_scratch_slot
        return {"state": {n: classify_state_attr(n) for n, a in vars(self.state).items() if _array(a)},
                "scratch": {n: classify_scratch_slot(n) for n, a in self.state._scratch.items() if _array(a)}}

    def snapshot(self) -> DomainSnapshot:
        """Capture a complete host checkpoint, including physics clocks and carry slots.

        The existing restart writer validates completeness. A temporary NPZ is
        removed before returning; only owned bytes remain in host memory.
        """
        from woof.io.restart import write_restart
        extra = tuple(n for n, verdict in self.carried_arrays()["scratch"].items()
                      if verdict == "carry")
        with tempfile.TemporaryDirectory(prefix="gpuwm-domain-") as directory:
            path = Path(directory) / "snapshot.npz"
            write_restart(path, self.state, self.cfg, extra_scratch_slots=extra)
            return DomainSnapshot(path.read_bytes(), self.cfg,
                                  radiation_next_step=self._radiation_next_step)

    def restore(self, snapshot: DomainSnapshot) -> None:
        """Restore a snapshot into identical prepared setup; also restore its dt.

        The restart reader validates the configuration/setup before mutation.
        Different grids/physics/forcing raise the existing restart mismatch.
        """
        from woof.io.restart import restore_restart
        if snapshot.api_version != API_VERSION:
            raise ValueError(f"unsupported snapshot API version {snapshot.api_version!r}")
        cfg = replace(self.cfg, dt=snapshot.config.dt)
        with tempfile.TemporaryDirectory(prefix="gpuwm-domain-") as directory:
            path = Path(directory) / "snapshot.npz"
            path.write_bytes(snapshot.payload)
            restore_restart(path, self.state, cfg)
        self.cfg = cfg
        self._radiation_next_step = snapshot.radiation_next_step

    def copy_overlap(self, src_domain: Domain, offset_i: int, offset_j: int) -> dict[str, int]:
        """Copy carried state/scratch/physics arrays into the prepared overlap.

        Source cell (j,i) maps to destination (j+offset_j,i+offset_i), including
        u/v/w faces. Exposed cells keep their initialized values. Clocks,
        physics bookkeeping and setup are NOT relocated: this is array transfer,
        not a restart. Use snapshot/restore for complete continuation.
        """
        di, dj = operator.index(offset_i), operator.index(offset_j)
        src, dst = src_domain.cfg, self.cfg
        # Different cell metrics/species cannot represent the same moving grid.
        for key in ("nz", "dx", "dy", "ztop", "mp_physics", "moist",
                    "hybrid_opt", "etac", "p_surf", "chem_sets"):
            if getattr(src, key) != getattr(dst, key):
                raise ValueError(f"copy_overlap requires matching {key}; copying would change the represented cells/species")
        source_inventory = src_domain.carried_arrays()
        targets, result = [], {}
        for group, carried in (("state", ("serialize", "checkpoint_only")),
                               ("scratch", ("serialize", "carry"))):
            for name, verdict in source_inventory[group].items():
                if verdict not in carried:
                    continue
                a = (getattr(src_domain.state, name) if group == "state"
                     else src_domain.state._scratch[name])
                if a.ndim < 2 or a.shape[-2:] not in (
                        (src.ny, src.nx), (src.ny, src.nx + 1), (src.ny + 1, src.nx)):
                    continue
                b = (getattr(self.state, name, None) if group == "state"
                     else self.state._scratch.get(name))
                shape = (*a.shape[:-2], dst.ny + a.shape[-2] - src.ny,
                         dst.nx + a.shape[-1] - src.nx)
                if b is None and group == "scratch":
                    targets.append((group, name, a, None, shape))
                elif b is None or b.shape != shape or b.dtype != a.dtype:
                    raise ValueError(f"copy_overlap destination {group}/{name} lacks matching storage {shape}")
                else:
                    targets.append((group, name, a, b, shape))
        if (src_domain.state.physics is None) != (self.state.physics is None):
            raise ValueError("copy_overlap requires matching physics attachments to avoid dropping carried fields")
        if src_domain.state.physics is not None:
            from woof.io.restart import _driver_manifest
            # Use the restart writer's inventory, including held tendencies,
            # surface fields and nested scheme state, rather than guessing
            # which direct driver attributes happen to be arrays today.
            source_physics = _driver_manifest(src_domain.state.physics)
            target_physics = _driver_manifest(self.state.physics)
            for name, a in source_physics.items():
                if not _array(a) or a.ndim < 2 or a.shape[-2:] not in (
                        (src.ny, src.nx), (src.ny, src.nx + 1), (src.ny + 1, src.nx)):
                    continue
                shape = (*a.shape[:-2], dst.ny + a.shape[-2] - src.ny,
                         dst.nx + a.shape[-1] - src.nx)
                b = target_physics.get(name)
                if b is None or b.shape != shape or b.dtype != a.dtype:
                    raise ValueError(f"copy_overlap destination physics/{name} lacks matching prepared storage {shape}")
                targets.append(("physics", name, a, b, shape))
        import cupy as cp
        # Validate every target before changing any array.
        for group, name, a, b, shape in targets:
            if b is None:
                b = cp.zeros(shape, dtype=a.dtype)
                self.state._scratch[name] = b
            i0, j0 = max(0, -di), max(0, -dj)
            i1, j1 = min(a.shape[-1], b.shape[-1] - di), min(a.shape[-2], b.shape[-2] - dj)
            if i1 > i0 and j1 > j0:
                b[..., j0 + dj:j1 + dj, i0 + di:i1 + di] = a[..., j0:j1, i0:i1].copy()
                result[f"{group}/{name}"] = math.prod(a.shape[:-2]) * (i1 - i0) * (j1 - j0)
        self.diagnostics()
        return result

    def force_radiation_next_step(self) -> None:
        """Request exactly one due call, preserving the driver's previous override."""
        from woof.config import radiation_enabled
        if self.state.physics is None or not radiation_enabled(self.cfg):
            raise ValueError("no radiation scheme is configured; a due override cannot produce radiation")
        self._radiation_next_step = True

    def boundary_snapshot(self) -> Mapping[str, Any]:
        """Host float64 mass-coupled LBC fields (not primitive get() units)."""
        from woof.ingest.lateral_bc import domain_boundary_snapshot
        return domain_boundary_snapshot(self.state)

    def extract_side(self, side: str, width: int | None = None) -> Mapping[str, Any]:
        """Host WRF side table; east/north run outermost-first."""
        from woof.ingest.lateral_bc import extract_lateral_side
        return extract_lateral_side(self.boundary_snapshot(), side,
                                    self.cfg.spec_bdy_width if width is None else width)

    def set_lateral_boundaries(self, boundaries: Any = None, *,
                               intervals: Iterable[Any] | None = None,
                               first: Mapping | None = None, second: Mapping | None = None,
                               start_seconds: float | None = None,
                               end_seconds: float | None = None) -> None:
        """Attach LateralBoundaries, intervals, or paired W/E/S/N tables.

        Uses the engine attachment, including its clock-reset semantics.
        Supply exactly one form; cfg must select specified boundaries.
        """
        from woof.ingest.lateral_bc import (LateralBoundaries,
            build_lateral_interval_from_sides, attach_lateral_boundaries)
        if sum((boundaries is not None, intervals is not None, first is not None)) != 1:
            raise ValueError("supply boundaries, intervals, or first/second side tables")
        if first is None and any(v is not None for v in (second, start_seconds, end_seconds)):
            raise ValueError("side-table times/second require first; they cannot modify an existing boundary object")
        if not self.cfg.specified:
            raise ValueError("specified=False would ignore the attached forcing; build with boundary='specified'")
        if first is not None:
            if second is None or start_seconds is None or end_seconds is None:
                raise ValueError("side tables need first, second, start_seconds and end_seconds")
            intervals = (build_lateral_interval_from_sides(first, second,
                            start_seconds=start_seconds, end_seconds=end_seconds),)
        if intervals is not None:
            boundaries = LateralBoundaries(tuple(intervals), self.cfg.spec_bdy_width,
                                          self.cfg.spec_zone, self.cfg.relax_zone)
        attach_lateral_boundaries(self.state, boundaries)


def device_free_bytes() -> int | None:
    """Reusable device bytes, or None when no device can be read."""
    from woof.core.resident_admission import device_free_bytes as free
    return free()


def estimate_device_bytes(nx: int, ny: int, nz: int, physics: RunConfig, *,
                          p_top: float = 5000.0, n_lbc_intervals: int = 0) -> int:
    """Existing single-domain resident + transient estimate in bytes.

    ``physics`` is the full RunConfig; no selector-only heuristic. This is the
    engine's itemized domain estimate, excluding CUDA context, allocator margin
    and other domains. It is not an allocation guarantee or the forecast envelope.
    """
    from types import SimpleNamespace
    from woof.core.preflight import estimate_domain
    cfg = replace(physics, nx=nx, ny=ny, nz=nz)
    estimate = estimate_domain(SimpleNamespace(run=cfg, grid_id=cfg.grid_id, parent_id=0),
                               p_top=p_top, n_lbc_intervals=n_lbc_intervals)
    return int(estimate.resident_bytes + estimate.transient_bytes)


__all__ = ["API_VERSION", "Domain", "DomainSnapshot", "WarmBubble",
           "device_free_bytes", "estimate_device_bytes"]
