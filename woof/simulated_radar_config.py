"""Metadata, validation and forecast-door admission for the optional simulated radar products."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import re
from typing import Mapping


SCAN_STRATEGIES = {
    "vcp212": (0.5, 0.9, 1.3, 1.8, 2.4, 3.1, 4.0, 5.1, 6.4, 8.0,
               10.0, 12.5, 15.6, 19.5),
    "low_tilts": (0.5, 0.9, 1.3),
}
FORMATS = ("level2", "cfradial1", "cfradial2", "odim")
FIELDS = ("reflectivity", "velocity", "zdr", "rhohv", "phidp", "kdp", "spectrum_width")
#: The colour tables the reflectivity and velocity PPI images draw with:
#: ``standard`` is the renderer's radar tables, ``classic`` the reflectivity
#: ladder and blue-red velocity scale they replaced in 2.8.5. The same two
#: names the render door takes as ``--radar-colors``.
COLOR_TABLES = ("standard", "classic")
REQUEST_SCHEMA = "simulated-radar.request/v1"
MANIFEST_SCHEMA = "simulated-radar.manifest/v1"
BAND_WAVELENGTHS_MM = {"S": 107.068735, "C": 53.5, "X": 33.3}


def radar_band(band=None, wavelength_mm=None):
    """Resolve legacy S or a C/X table without relabeling its wavelength."""
    if band is not None:
        if not isinstance(band, str) or band.upper() not in BAND_WAVELENGTHS_MM:
            raise ValueError("radar_band_unsupported: band must be S, C or X")
        band = band.upper()
    if wavelength_mm is not None:
        if isinstance(wavelength_mm, bool) or not isinstance(wavelength_mm, (int, float)):
            raise ValueError("radar_wavelength_unsupported: wavelength_mm must be a finite number")
        matched = next((key for key, value in BAND_WAVELENGTHS_MM.items()
                        if wavelength_mm == value), None)
        if matched is None:
            raise ValueError("radar_wavelength_unsupported: legacy S is 107.068735 mm (2.8 GHz), "
                             "and C/X tables are 53.5 and 33.3 mm; substituting another wavelength would change "
                             "the requested scattering and attenuation")
        if band is not None and band != matched:
            raise ValueError("radar_band_wavelength_mismatch: band and wavelength select different tables")
        band = matched
    return band

#: Upper limits on the scan geometry, each with the breakage it prevents.
#: The same rows are ``GEOMETRY_LIMITS`` in the native
#: ``tools/rustwx/crates/rw-simradar/src/request.rs``, so this door refuses
#: with the native's sentence before anything runs.  ``gate_spacing_m`` has
#: no universal ceiling: the scan always budgets one gate, and only Level II
#: stores the spacing in a bounded field (:data:`LEVEL2_MAX_GATE_SPACING_M`).
GEOMETRY_LIMITS = {
    "range_km": (460.0, (
        "460 km is the WSR-88D's longest unambiguous range (its long-PRT "
        "surveillance cut), so a simulated S-band volume reaching farther would "
        "hold echo where that radar's own data range-folds")),
    "azimuth_step_deg": (720.0, (
        "the ray count is 360 / azimuth_step_deg rounded, which is zero above 720 "
        "degrees; the simulator would scan one ray while the memory and work "
        "estimates priced none")),
    "volume_duration_s": (3600.0, (
        "a custom ladder turns at 360 x elevations / volume_duration_s deg/s and "
        "the simulator holds that rate at no less than 0.1 deg/s, so a longer "
        "one-elevation scan would run 3600 s per sweep instead of the duration "
        "asked for")),
}

#: Message 31 carries gate spacing in a field holding 1 to 32767 m.
LEVEL2_MAX_GATE_SPACING_M = 32767.0


def _number(value, key, low, high, *, open_low=False):
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or value > high
            or (value <= low if open_low else value < low)):
        raise ValueError(f"[simulated_radar] {key} must be finite and "
                         f"{'greater than' if open_low else 'at least'} {low}, "
                         f"and at most {high}; got {value!r}")
    return float(value)


@dataclass(frozen=True)
class RadarSite:
    id: str
    lat: float
    lon: float
    height_m: float
    band: str | None = None
    beamwidth_deg: float = 0.95
    wavelength_mm: float | None = None

    def __post_init__(self):
        band = radar_band(self.band, self.wavelength_mm) or "S"
        object.__setattr__(self, "band", band)
        object.__setattr__(self, "wavelength_mm", BAND_WAVELENGTHS_MM[band])

    def to_mapping(self):
        """Keep historical site requests byte-identical at default geometry."""
        band = radar_band(self.band, self.wavelength_mm) or "S"
        result = {key: getattr(self, key) for key in ("id", "lat", "lon", "height_m")}
        if band != "S":
            result["band"] = band
        if self.beamwidth_deg != 0.95:
            result["beamwidth_deg"] = self.beamwidth_deg
        return result


@dataclass(frozen=True)
class SimulatedRadarOptions:
    enabled: bool = False
    sites: str | tuple = "auto"
    scan_strategy: str = "vcp212"
    elevations_deg: tuple[float, ...] = SCAN_STRATEGIES["vcp212"]
    formats: tuple[str, ...] = ("level2", "cfradial1")
    timing: str = "history"
    fields: str | tuple[str, ...] = "auto"
    range_km: float = 230.0
    gate_spacing_m: float = 250.0
    azimuth_step_deg: float = 1.0
    volume_duration_s: float = 300.0
    color_tables: str = "standard"
    band: str | None = None
    wavelength_mm: float | None = None

    @classmethod
    def from_mapping(cls, value, *, source="configuration"):
        if value is None:
            return OFF
        if not isinstance(value, Mapping):
            raise ValueError(f"[simulated_radar] in {source} must be a table")
        unknown = sorted(set(value) - set(cls.__dataclass_fields__))
        if unknown:
            raise ValueError(f"unknown keys in [simulated_radar] of {source}: {unknown}")
        enabled = value.get("enabled", True)
        if not isinstance(enabled, bool):
            raise ValueError("[simulated_radar] enabled must be true or false")
        sites = value.get("sites", "auto")
        if sites != "auto":
            if not isinstance(sites, (list, tuple)) or not sites:
                raise ValueError("[simulated_radar] sites must be 'auto' or a nonempty list")
            normalized = []
            for site in sites:
                if isinstance(site, str) and re.fullmatch(r"[A-Za-z0-9]{4}", site):
                    normalized.append(site.upper())
                elif isinstance(site, Mapping):
                    if (not {"id", "lat", "lon", "height_m"} <= set(site)
                            or set(site) - {"id", "lat", "lon", "height_m", "band", "beamwidth_deg", "wavelength_mm"}):
                        raise ValueError("custom radar sites require id, lat, lon and height_m")
                    name = site["id"]
                    if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9]{4}", name):
                        raise ValueError("radar site id must have four letters or digits for Level II")
                    normalized.append(RadarSite(
                        name.upper(), _number(site["lat"], "site latitude", -90, 90),
                        _number(site["lon"], "site longitude", -180, 180),
                        _number(site["height_m"], "site height_m", -500, 10000),
                        band=site.get("band"), wavelength_mm=site.get("wavelength_mm"),
                        beamwidth_deg=_number(site.get("beamwidth_deg", 0.95),
                                              "site beamwidth_deg", 0.05, math.inf)))
                    normalized[-1].to_mapping()
                else:
                    raise ValueError("radar sites must be four-character IDs or coordinate tables")
            ids = [s if isinstance(s, str) else s.id for s in normalized]
            if len(ids) != len(set(ids)):
                raise ValueError("duplicate radar site IDs would overwrite volume files")
            sites = tuple(normalized)
        strategy = value.get("scan_strategy", "custom" if "elevations_deg" in value else "vcp212")
        if not isinstance(strategy, str) or strategy not in (*SCAN_STRATEGIES, "custom"):
            raise ValueError("scan_strategy must be 'vcp212', 'low_tilts' or 'custom'")
        elevations = value.get("elevations_deg", SCAN_STRATEGIES.get(strategy, SCAN_STRATEGIES["vcp212"]))
        if not isinstance(elevations, (list, tuple)) or not 1 <= len(elevations) <= 255:
            raise ValueError("elevations_deg must contain 1 to 255 sweep angles")
        elevations = tuple(_number(e, "elevations_deg", -1, 90) for e in elevations)
        if any(b <= a for a, b in zip(elevations, elevations[1:])):
            raise ValueError("elevations_deg must increase strictly to keep sweep indices unique")
        if strategy != "custom" and elevations != SCAN_STRATEGIES[strategy]:
            raise ValueError("custom elevations_deg require scan_strategy='custom' to avoid mislabeling the volume")
        formats = value.get("formats", ("level2", "cfradial1"))
        if (not isinstance(formats, (list, tuple)) or not formats
                or any(f not in FORMATS for f in formats)
                or len(formats) != len(set(formats))):
            raise ValueError(f"formats must be a nonempty list of unique values from {FORMATS}")
        fields = value.get("fields", "auto")
        if fields != "auto":
            if (not isinstance(fields, (list, tuple)) or not fields
                    or any(f not in FIELDS for f in fields)
                    or len(fields) != len(set(fields))):
                raise ValueError(f"fields must be 'auto' or a nonempty unique list from {FIELDS}")
            fields = tuple(fields)
        timing = value.get("timing", "history")
        if timing not in ("history", "scan"):
            raise ValueError("timing must be 'history' or 'scan'")
        color_tables = value.get("color_tables", "standard")
        if color_tables not in COLOR_TABLES:
            raise ValueError(f"color_tables must be one of {COLOR_TABLES}; got {color_tables!r}")
        defaults = {"range_km": 230.0, "gate_spacing_m": 250.0,
                    "azimuth_step_deg": 1.0, "volume_duration_s": 300.0}
        checked = {}
        for key, default in defaults.items():
            number = value.get(key, default)
            if (isinstance(number, bool) or not isinstance(number, (int, float))
                    or not math.isfinite(number) or number <= 0):
                raise ValueError(
                    f"[simulated_radar] {key} must be finite and positive: a zero, "
                    "negative or nonfinite value leaves no gate, ray or scan time to "
                    f"sample; got {number!r}")
            limit, reason = GEOMETRY_LIMITS.get(key, (math.inf, ""))
            if number > limit:
                raise ValueError(f"[simulated_radar] {key} must be at most {limit:g}: "
                                 f"{reason}; got {number!r}")
            checked[key] = float(number)
        if "level2" in formats and checked["gate_spacing_m"] > LEVEL2_MAX_GATE_SPACING_M:
            raise ValueError(
                f"[simulated_radar] gate_spacing_m must be at most {LEVEL2_MAX_GATE_SPACING_M:g} "
                "with level2: Message 31 stores gate spacing in a field holding 1 to "
                "32767 m; omit level2 for coarser gates")
        if not checked["gate_spacing_m"].is_integer():
            raise ValueError("gate_spacing_m must use whole metres: the shared radar geometry stores integer gate spacing")
        gates = 1000.0 * checked["range_km"] / checked["gate_spacing_m"]
        rays = 360.0 / checked["azimuth_step_deg"]
        if rays > 65535:
            raise ValueError("radial count exceeds the radar writer's 16-bit geometry fields")
        if gates > 16384:
            raise ValueError("radar writers support at most 16384 gates per radial; increase gate_spacing_m or reduce range_km")
        if "level2" in formats:
            if strategy != "vcp212" and len(elevations) > 32:
                raise ValueError("Level II supports at most 32 cuts; shorten the ladder or omit level2")
            if not isinstance(sites, str) and any(
                    isinstance(site, RadarSite) and site.id.startswith("T") and site.id != "TJUA"
                    for site in sites):
                raise ValueError("custom IDs beginning T trigger a fixed station lookup in Py-ART; use another ID or omit level2")
        band = radar_band(value.get("band"), value.get("wavelength_mm"))
        return cls(enabled=enabled, sites=sites, scan_strategy=strategy, band=band,
                   elevations_deg=elevations, formats=tuple(formats), timing=timing,
                   fields=fields, color_tables=color_tables, **checked)

    def to_mapping(self):
        result = asdict(self)
        result["band"] = radar_band(self.band, self.wavelength_mm)
        result["sites"] = self.sites if isinstance(self.sites, str) else [
            site.to_mapping() if isinstance(site, RadarSite) else site for site in self.sites]
        result["elevations_deg"] = list(self.elevations_deg)
        result["formats"] = list(self.formats)
        result["fields"] = self.fields if isinstance(self.fields, str) else list(self.fields)
        # The default is left out, so a table that does not name the key
        # writes the request, and gets the config identity, it always did.
        if result["color_tables"] == "standard":
            del result["color_tables"]
        if result["band"] is None:
            del result["band"]
        # The band is the canonical identity of these exact wavelength tables.
        del result["wavelength_mm"]
        return result


OFF = SimulatedRadarOptions()


def declared_key_rows():
    """Metadata for configuration editors, using the parsed option defaults."""
    from woof.config_keys import KeyRow
    defaults = SimulatedRadarOptions(enabled=True).to_mapping()
    defaults["color_tables"] = "standard"
    defaults["band"] = None
    defaults["wavelength_mm"] = None
    types = {"enabled": "boolean", "sites": ("string", "array"),
             "scan_strategy": "string", "elevations_deg": "array", "formats": "array",
             "timing": "string", "fields": ("string", "array"),
             "color_tables": "string", "band": "string",
             "wavelength_mm": "number"}
    descriptions = {
        "enabled": "Write simulated radar volumes and PPI images from saved history.",
        "sites": "Automatic coverage selection, site IDs or custom coordinates and antenna height.",
        "scan_strategy": "Native VCP 212 timing or a custom elevation ladder.",
        "elevations_deg": "Strictly increasing custom beam elevation angles in degrees.",
        "formats": "Radar file formats: level2, cfradial1, cfradial2 and odim.",
        "timing": "History snapshots or scan timing using neighboring outputs.",
        "fields": "Automatic available radar moments or an explicit list.",
        "range_km": "Maximum slant range in kilometres.",
        "gate_spacing_m": "Range gate spacing in metres.",
        "azimuth_step_deg": "Ray spacing in degrees.",
        "volume_duration_s": "Custom volume duration in seconds.",
        "color_tables": "PPI colour tables: standard, or classic for the reflectivity and velocity colours before 2.8.5.",
        "band": "Override the site band with S, C or X; omitted uses the site metadata.",
        "wavelength_mm": "Select legacy S at 107.068735 mm or a C/X table at 53.5 or 33.3 mm.",
    }
    return {name: KeyRow(name, types.get(name, "number"), default, descriptions[name])
            for name, default in defaults.items()}


def load_options(path):
    """Read only the same authorized TOML bytes the run consumes."""
    import tomllib
    from woof.config_authority import read_config_authority
    document = tomllib.loads(read_config_authority(path).payload.decode("utf-8-sig"))
    return SimulatedRadarOptions.from_mapping(document.get("simulated_radar"), source=str(path))


def execution_argument(value):
    """Parse a runtime output override without changing prepared authorities."""
    if value is None or isinstance(value, SimulatedRadarOptions):
        return value
    if isinstance(value, str):
        import json
        try:
            value = json.loads(value)
        except ValueError as error:
            raise ValueError(f"--simulated-radar-table must contain a JSON object: {error}") from None
    return SimulatedRadarOptions.from_mapping(value, source="--simulated-radar-table")


def execution_flags(options):
    """The same table crosses every prepared-run subprocess boundary."""
    import json
    options = execution_argument(options)
    return ([] if options is None else
            ["--simulated-radar-table", json.dumps(options.to_mapping(), sort_keys=True)])


def add_execution_argument(parser):
    import argparse

    def parse(value):
        try:
            return execution_argument(value)
        except (ValueError, TypeError) as error:
            raise argparse.ArgumentTypeError(str(error)) from None

    parser.add_argument(
        "--simulated-radar-table", default=None, metavar="JSON", type=parse,
        help="[simulated_radar] options as JSON; overrides output options without editing a prepared configuration")


def apply_execution_options(experiment, options):
    """Overlay output metadata after preparation identity has been checked."""
    if options is None:
        return experiment
    from dataclasses import replace
    from woof.io.history_selection import resolve
    options = execution_argument(options)
    for domain in experiment.domains:
        validate_history_selection(options, resolve(experiment.output, domain.output),
                                   where=f"d{domain.grid_id:02d}")
    return replace(experiment, simulated_radar=options)


#: The label a forecast door gives a radar refusal beside its other
#: machine preconditions.
REFUSAL_LABEL = "[simulated_radar]"


def scene_shapes(experiment) -> list[tuple[int, int, int]]:
    """Every grid's ``(nx, ny, nz)`` mass shape, as its history will carry it."""
    return [(int(domain.run.nx), int(domain.run.ny), int(domain.run.nz))
            for domain in experiment.domains]


def door_refusals(experiment) -> tuple[tuple[str, str], ...]:
    """``(label, sentence)`` for radar this machine cannot serve, or ``()``.

    Asked by every forecast door before the fetch, preparation or GPU
    allocation, through :func:`woof.config.experiment_preparation_refusals`
    and the prepared runners' preflight. Before this, a missing or stale
    ``rw_simradar`` and a scan the host memory cannot hold were found only
    when the first history landed, after the download, the preparation and
    a whole history interval of forecast.
    """
    options = getattr(experiment, "simulated_radar", None)
    if options is None or not options.enabled:
        return ()
    from pathlib import Path

    from woof import rustwx

    try:
        binary = rustwx.require_simulated_radar_binary()
        estimate = rustwx.estimate_simulated_radar(
            (), outdir=Path.cwd(), config=options.to_mapping(),
            scene_shapes=scene_shapes(experiment), binary=binary)
    except rustwx.SimulatedRadarRefusal as refusal:
        return ((REFUSAL_LABEL, str(refusal)),)
    if not estimate.get("memory_admitted_now"):
        return ((REFUSAL_LABEL, (
            "the radar scan runs beside the forecast and would be refused at the "
            f"first history: {estimate.get('memory_refusal')}. Next: widen "
            "gate_spacing_m or azimuth_step_deg, shorten the elevation ladder, "
            "or turn [simulated_radar] off.")),)
    return ()


def require_admitted(experiment) -> None:
    """Raise the first :func:`door_refusals` sentence, for a door that raises.

    A ``ValueError``, as :func:`woof.config.validate_experiment_preparation`
    raises the same inventory: every door prints it as one refusal at exit 2.
    """
    for label, sentence in door_refusals(experiment):
        raise ValueError(f"{label}: {sentence}")


def validate_history_selection(options, selection, *, where):
    """Reject a tape that drops the atmospheric columns beams need."""
    if not options.enabled:
        return
    required = ("XLAT", "XLONG", "HGT", "PH", "PHB", "P", "PB", "T",
                "U", "V", "W", "QVAPOR", "QRAIN")
    missing = [name for name in required if not selection.keeps(name)]
    if missing:
        raise ValueError(f"simulated radar on {where} requires history variables {missing}; "
                         "[output] drops the atmospheric columns used by the virtual beam")
