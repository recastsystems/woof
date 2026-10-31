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

    def __post_init__(self):
        if type(self.enabled) is not bool:
            raise ValueError("enabled must be a boolean")
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
        for name in ("guv", "gt", "gph", "gq", "fgdt", "gfdda_interval_m", "gfdda_end_h", "dtramp_min"):
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
        return asdict(self)

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
