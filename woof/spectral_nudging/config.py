"""Opt-in controls for large-scale downscaling; no device import at load time."""
from dataclasses import asdict, dataclass
import math
import tomllib
from pathlib import Path


@dataclass(frozen=True)
class NudgingConfig:
    enabled: bool = False
    mode: str = "spectral"
    convention: str = "regional"
    cutoff_x: int = 3
    cutoff_y: int = 2
    relaxation_seconds: float = 5000.0
    minimum_height_m: float = 1500.0
    ramp_depth_m: float = 1500.0
    boundary_cells: int = 1
    guv: float | None = None
    gt: float | None = None
    gph: float | None = None
    gq: float = 0.0
    if_no_pbl_nudging_uv: int = 1
    if_no_pbl_nudging_t: int = 1
    if_no_pbl_nudging_ph: int = 1
    if_no_pbl_nudging_q: int = 1
    if_zfac_uv: int = 0
    if_zfac_t: int = 0
    if_zfac_ph: int = 0
    if_zfac_q: int = 0
    k_zfac_uv: int = 10
    k_zfac_t: int = 10
    k_zfac_ph: int = 10
    k_zfac_q: int = 10
    dk_zfac_uv: int = 1
    dk_zfac_t: int = 1
    dk_zfac_ph: int = 1
    dk_zfac_q: int = 1
    fgdt: float = 0.0
    fgdtzero: int = 0
    gfdda_interval_m: float | None = None
    gfdda_end_h: float | None = None
    if_ramping: int = 0
    dtramp_min: float = 0.0
    grid_sfdda: int = 0
    sgfdda_inname: str = "sgfdda_d<domain>"
    io_form_sgfdda: int = 2
    sgfdda_interval_m: float = 0.0
    sgfdda_end_h: float = 0.0
    guv_sfc: float = 0.0
    gt_sfc: float = 0.0
    gq_sfc: float = 0.0
    rinblw: float = 0.0
    gfdda_inname: str = "wrffdda_d<domain>"
    io_form_gfdda: int = 2

    def __post_init__(self):
        if type(self.enabled) is not bool:
            raise ValueError("enabled must be a boolean")
        if type(self.grid_sfdda) is not int or self.grid_sfdda not in (0, 1, 2):
            raise ValueError("grid_sfdda must be 0, 1 or 2")
        if self.grid_sfdda:
            if self.mode != "grid" or self.convention != "wrf":
                raise ValueError("WRF surface FDDA runs inside grid_fdda=1")
            if self.io_form_sgfdda != 2:
                raise ValueError("surface FDDA requires NetCDF auxinput9 (io_form_sgfdda=2)")
            if self.io_form_gfdda != 2:
                raise ValueError("surface FDDA's native grid driver requires NetCDF auxinput10 (io_form_gfdda=2)")
            if self.sgfdda_interval_m <= 0 or self.rinblw < .001:
                raise ValueError("surface FDDA requires sgfdda_interval_m > 0 and rinblw >= .001 km")
            if not isinstance(self.sgfdda_inname, str) or not self.sgfdda_inname:
                raise ValueError("surface FDDA needs its native sgfdda_inname")
            if any(self.coefficient(name)>0 for name in ("u","v","temperature","qv")):
                if self.gfdda_interval_m is None or self.gfdda_interval_m <= 0:
                    raise ValueError("surface FDDA with upper nudging requires gfdda_interval_m > 0")
                if not isinstance(self.gfdda_inname,str) or not self.gfdda_inname:
                    raise ValueError("surface FDDA with upper nudging needs gfdda_inname")
        if self.mode not in ("spectral", "grid"):
            raise ValueError("nudging mode must be spectral or grid")
        if self.convention not in ("regional", "wrf"):
            raise ValueError("nudging convention must be regional or wrf")
        if self.mode == "grid" and self.convention != "wrf":
            raise ValueError("grid nudging uses WRF potential-temperature relaxation")
        for name in self.__dataclass_fields__:
            value = getattr(self, name)
            if name.startswith("if_") or name == "fgdtzero":
                if type(value) is not int or value not in (0, 1):
                    raise ValueError(f"{name} must be 0 or 1")
            if name.startswith(("k_zfac_", "dk_zfac_")):
                if type(value) is not int or value < int(name.startswith("dk_")):
                    raise ValueError(f"{name} has an invalid model-level count")
        for name in ("guv", "gt", "gph", "gq", "fgdt", "gfdda_interval_m", "gfdda_end_h", "dtramp_min",
                     "sgfdda_interval_m", "sgfdda_end_h", "guv_sfc", "gt_sfc", "gq_sfc", "rinblw"):
            value = getattr(self, name)
            if value is None:
                continue
            if isinstance(value, bool) or not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
            if name != "dtramp_min" and value < 0:
                raise ValueError(f"{name} must be nonnegative")
        if self.gfdda_interval_m == 0:
            raise ValueError("gfdda_interval_m must be positive to bracket analysis times")
        for name in ("cutoff_x", "cutoff_y", "boundary_cells"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        for name in ("relaxation_seconds", "ramp_depth_m"):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite")
        if (isinstance(self.minimum_height_m, bool)
                or not math.isfinite(self.minimum_height_m)
                or self.minimum_height_m < 0):
            raise ValueError("minimum_height_m must be nonnegative and finite")

    def validate_grid(self, shape):
        if len(shape) != 3 or min(shape) < 1:
            raise ValueError("fields must have (level, y, x) shape")
        _, ny, nx = shape
        if self.cutoff_x > nx // 2 or self.cutoff_y > ny // 2:
            raise ValueError("cutoff exceeds the grid Nyquist wavenumber")
        if 2 * self.boundary_cells >= min(nx, ny):
            raise ValueError("the protected boundary consumes the entire grid")

    def to_dict(self):
        values = asdict(self)
        if not self.grid_sfdda:
            for name in SURFACE_KEYS:
                values.pop(name)
        return values

    def coefficient(self, name):
        key = {"u": "guv", "v": "guv", "temperature": "gt",
               "geopotential": "gph", "qv": "gq"}[name]
        if name == "geopotential" and self.mode == "grid":
            return 0.0
        value = getattr(self, key)
        return 1.0 / self.relaxation_seconds if value is None else value

    def cadence_steps(self, dt):
        # WRF module_physics_init: nearest integer, at least one step.
        return max(1, int(self.fgdt * 60.0 / dt + 0.5))

    def held_steps(self, dt):
        """Most steps one refresh's tendency is applied (kernels.cu relax).

        Without fgdtzero a refresh is held until the next one: N steps. With
        fgdtzero=1 the tendency is cleared on the step after each regular
        refresh, so only the first call (step 1, held until step N) is applied
        for more than one step: N - 1 steps.
        """
        steps = self.cadence_steps(dt)
        return max(1, steps - 1) if self.fgdtzero and steps > 1 else steps

    def step_strength(self, dt):
        """Largest share of the analysis mismatch one refresh applies.

        The relaxation is an explicit update: each step adds coefficient*dt of
        the mismatch, and a tendency refreshed every N steps (fgdt) is held for
        N steps, so one refresh moves coefficient*dt*N of it. Masks and time
        ramps only scale this down. Returns (strength, key, coefficient).
        """
        steps = self.held_steps(dt)
        rows = [(key, self.coefficient(name)) for key, name in
                (("guv", "u"), ("gt", "temperature"), ("gph", "geopotential"), ("gq", "qv"))]
        if self.grid_sfdda:
            rows += [(key, getattr(self, key)) for key in ("guv_sfc", "gt_sfc", "gq_sfc")]
        key, value = max(rows, key=lambda row: row[1])
        return value * dt * steps, key, value

    def check_step(self, dt, where="nudging"):
        """NUDGING_STEP_STRENGTH: refuse a strength that blows the run up.

        One refresh multiplies the analysis mismatch by (1 - strength).
        Measured on the 9 km ERA5 example (nudging-blowups note, 2026-11-10):
        - applied every step (fgdt=0), above 1 the state is pushed past the
          analysis on every step and the model amplifies that two-step
          oscillation: 1.35 went non-finite at step 572, 2.25 at step 13
          (grid and spectral); 1.0 and 0.45 ran 12 h;
        - held by fgdt, every refresh overshoots by (strength - 1): 1.8 ran
          12 h but its 10 m wind map is shredded into grid-scale noise, and
          2.16 went non-finite at step 956. 1.08 (8 % overshoot) ran and
          looked normal; it is refused too, with the admissible coefficient
          named, because the line that never overshoots is 1.
        Native WRF has no guard and diverges the same way.
        """
        if not self.enabled or not math.isfinite(dt) or dt <= 0:
            return
        strength, key, value = self.step_strength(dt)
        if strength <= STEP_STRENGTH_SAFE:
            return
        steps = self.held_steps(dt)
        name ="1/relaxation_seconds" if key in ("guv", "gt", "gph") and getattr(self, key) is None else key
        held = f" x {steps} steps held by fgdt" if steps > 1 else ""
        raise ValueError(
            f"NUDGING_STEP_STRENGTH: {where} {name}={value:.4g} s^-1 x time_step {dt:g} s{held} = "
            f"{strength:.3g} of the analysis mismatch per refresh, so each refresh overshoots the analysis "
            f"and the run blows up; the safe range is coefficient x time_step x steps <= 1, here "
            f"{name} <= {STEP_STRENGTH_SAFE / (dt * steps):.3g} s^-1 (WRF's usual value is 3e-4)")


#: Share of the mismatch per refresh that never overshoots the analysis.
STEP_STRENGTH_SAFE = 1.0


SURFACE_KEYS = ("grid_sfdda", "sgfdda_inname", "io_form_sgfdda",
                "sgfdda_interval_m", "sgfdda_end_h", "guv_sfc", "gt_sfc", "gq_sfc", "rinblw",
                "gfdda_inname", "io_form_gfdda")


@dataclass(frozen=True)
class NudgingOptions:
    """Per-domain selection. Unselected nests retain their unnudged trajectory."""
    domains: tuple[tuple[int, NudgingConfig], ...]

    def for_domain(self, grid_id):
        return dict(self.domains).get(int(grid_id))

    def to_dict(self):
        return {str(key): value.to_dict() for key, value in self.domains}


def from_mapping(row, domain_ids=(1,)):
    """Parse a root request and optional [nudging.domain.N] overrides."""
    row = dict(row)
    selected = row.pop("domain", {})
    result = {}
    if row:
        row.setdefault("enabled", True)
        if row.get("mode") == "grid":
            row.setdefault("convention", "wrf")
        cfg = NudgingConfig(**row)
        if cfg.enabled:
            result[domain_ids[0]] = cfg
    for key, value in selected.items():
        grid_id = int(key)
        if grid_id not in domain_ids:
            raise ValueError(f"nudging domain {grid_id} does not exist")
        settings = {**row, **value}
        settings.setdefault("enabled", True)
        if settings.get("mode") == "grid":
            settings.setdefault("convention", "wrf")
        cfg = NudgingConfig(**settings)
        if cfg.enabled:
            result[grid_id] = cfg
        else:
            result.pop(grid_id, None)
    return NudgingOptions(tuple(sorted(result.items()))) if result else None


def load_config(path):
    with Path(path).open("rb") as stream:
        document = tomllib.load(stream)
    row = document.get("downscale", {})
    unknown = set(row) - set(NudgingConfig.__dataclass_fields__)
    if unknown:
        raise ValueError(f"unknown downscale controls: {sorted(unknown)}")
    return NudgingConfig(**row)
