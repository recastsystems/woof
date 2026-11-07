"""Optional high-resolution water-surface-temperature overlay (task #71).

ERA5's water-surface temperature is coarse over lakes and coasts: SST is
an ocean analysis painted at 0.25 degrees and SKINTEMP over lakes is the
0.25-degree FLake state, so a convection-permitting nest inherits hard
rectangular quantization over water (the documented stock-WRF + ERA5
limitation; WRF users mitigate it with ``avg_tsfc.exe`` or a
high-resolution SST product fed to metgrid).  This module is the second
community remedy implemented natively: a user-supplied high-resolution
gridded water-temperature analysis (netCDF or GRIB2; OSTIA-class global
or ESA-CCI-class products are the reference shapes) replaces ERA5
SST/SKINTEMP over WATER source cells before horizontal interpolation.

The overlay is applied at the snapshot level, upstream of every
interpolation chain, so ``woof/ingest/horiz.py`` needs no edit and the
absent-configuration path is the identity: no overlay means the exact
same snapshot objects flow onward, byte for byte.

Documented v1 seams:

* Coverage ends at the overlay's bounding box and at cells whose four
  bilinear corners are all invalid; those water cells keep their ERA5
  values (ERA5-cell-granularity edge between overlay and fallback).
* A global overlay's longitude ring is treated as a finite ascending
  axis with one artificial cut; there is no periodic wrap across the
  cut, matching the stance ``orient_global_source_longitudes`` takes for
  the ERA5 ring.  Water cells in that one-cell gap fall back to ERA5.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from functools import lru_cache
from collections.abc import Sequence
import hashlib
import os
from pathlib import Path

import numpy as np


class WaterOverlayError(ValueError):
    """A named refusal from the water-temperature overlay."""


#: CF standard names accepted as an unambiguous water-temperature field.
_STANDARD_NAMES = frozenset({
    "sea_surface_temperature",
    "sea_surface_skin_temperature",
    "sea_surface_subskin_temperature",
    "sea_surface_foundation_temperature",
    "lake_surface_water_temperature",
    "sea_water_temperature",
    "surface_temperature",
})
#: Variable names accepted when no standard_name identifies the field.
_VARIABLE_NAMES = frozenset({
    "analysed_sst", "sst", "water_temp", "wtmp", "lswt",
})
_KELVIN_UNITS = frozenset({
    "k", "kelvin", "kelvins", "degree_kelvin", "degrees_kelvin",
    "deg_k", "degk",
})
_CELSIUS_UNITS = frozenset({
    "c", "celsius", "degc", "deg_c", "degree_c", "degree_celsius",
    "degrees_celsius", "degrees_c",
})
#: Median plausibility window for an analyzed water-temperature field, K.
_MEDIAN_WINDOW_K = (250.0, 340.0)
#: Per-cell hard window on valid cells, K (catches undeclared sentinels).
_CELL_WINDOW_K = (230.0, 350.0)
#: WMO GRIB2 (discipline, category, parameter) for water temperature, K.
_GRIB2_WATER_TEMPERATURE = (10, 3, 0)


@dataclass(frozen=True)
class WaterTemperatureOverlay:
    """One loaded high-resolution water-temperature analysis.

    ``latitude``/``longitude`` are 1-D float64 strictly ascending axes;
    ``temperature_k`` is 2-D float64 kelvin on (latitude, longitude);
    ``valid`` folds declared-missing and finiteness into one 2-D bool.
    """

    path: Path
    source_format: str
    variable: str
    declared_units: str
    latitude: np.ndarray
    longitude: np.ndarray
    temperature_k: np.ndarray
    valid: np.ndarray

    def __post_init__(self) -> None:
        for name in ("latitude", "longitude"):
            axis = np.asarray(getattr(self, name), dtype=np.float64)
            if axis.ndim != 1 or axis.size < 2:
                raise WaterOverlayError(
                    f"overlay {name} must be a 1-D axis of at least two "
                    f"points, got shape {axis.shape}")
            if not np.isfinite(axis).all():
                raise WaterOverlayError(
                    f"overlay {name} contains non-finite values")
            if not np.all(np.diff(axis) > 0.0):
                raise WaterOverlayError(
                    f"overlay {name} axis must be strictly ascending")
            copied = axis.copy()
            copied.setflags(write=False)
            object.__setattr__(self, name, copied)
        shape = (self.latitude.size, self.longitude.size)
        temperature = np.asarray(self.temperature_k, dtype=np.float64)
        valid = np.asarray(self.valid, dtype=bool)
        if temperature.shape != shape or valid.shape != shape:
            raise WaterOverlayError(
                f"overlay grids must have shape {shape}; got temperature "
                f"{temperature.shape} and valid {valid.shape}")
        valid = valid & np.isfinite(temperature)
        temperature = temperature.copy()
        temperature.setflags(write=False)
        valid.setflags(write=False)
        object.__setattr__(self, "temperature_k", temperature)
        object.__setattr__(self, "valid", valid)


def _normalize_units(declared: str, path: Path) -> str:
    token = str(declared).strip().lower().replace(" ", "_")
    token = token.replace("°", "deg")
    if token in _KELVIN_UNITS:
        return "K"
    if token in _CELSIUS_UNITS:
        return "C"
    raise WaterOverlayError(
        f"water-temperature overlay {path} declares units {declared!r}; "
        "only kelvin and Celsius spellings are accepted")


def _plausibility_guard(
        temperature_k: np.ndarray, valid: np.ndarray,
        declared: str, path: Path) -> None:
    """Named K-vs-C refusals, both directions, then the per-cell window."""

    if not np.any(valid):
        raise WaterOverlayError(
            f"water-temperature overlay {path} has no valid cells")
    low, high = _MEDIAN_WINDOW_K
    median = float(np.median(temperature_k[valid]))
    if not low <= median <= high:
        if declared == "K" and low <= median + 273.15 <= high:
            raise WaterOverlayError(
                f"water-temperature overlay {path} has Celsius-shaped "
                f"values (median {median:.2f}) under a kelvin declaration")
        if declared == "C" and low <= median - 273.15 <= high:
            raise WaterOverlayError(
                f"water-temperature overlay {path} has kelvin-shaped "
                f"values (median {median - 273.15:.2f} before conversion) "
                "under a Celsius declaration")
        raise WaterOverlayError(
            f"water-temperature overlay {path} is implausible as a water "
            f"temperature analysis: valid-cell median {median:.2f} K is "
            f"outside [{low:g}, {high:g}] K")
    cell_low, cell_high = _CELL_WINDOW_K
    outside = valid & ((temperature_k < cell_low)
                       | (temperature_k > cell_high))
    count = int(np.count_nonzero(outside))
    if count:
        raise WaterOverlayError(
            f"water-temperature overlay {path} has {count} valid cell(s) "
            f"outside [{cell_low:g}, {cell_high:g}] K -- an undeclared "
            "missing-value sentinel is the usual cause; declare it in the "
            "file's mask/attributes")


def _oriented(latitude, longitude, temperature, valid, path):
    """Ascending axes; a wrapped longitude ring is cut and re-sorted."""

    latitude = np.asarray(latitude, dtype=np.float64)
    longitude = np.asarray(longitude, dtype=np.float64)
    if latitude.ndim != 1 or longitude.ndim != 1:
        raise WaterOverlayError(
            f"water-temperature overlay {path} coordinates must be 1-D "
            "axes")
    if np.any(np.abs(latitude) > 90.0 + 1.0e-9):
        raise WaterOverlayError(
            f"water-temperature overlay {path} latitude axis leaves "
            "[-90, 90]; the trailing dimensions must be (latitude, "
            "longitude)")
    if latitude.size >= 2 and np.all(np.diff(latitude) < 0.0):
        latitude = latitude[::-1]
        temperature = temperature[::-1, :]
        valid = valid[::-1, :]
    if longitude.size >= 2 and np.all(np.diff(longitude) < 0.0):
        longitude = longitude[::-1]
        temperature = temperature[:, ::-1]
        valid = valid[:, ::-1]
    if not np.all(np.diff(longitude) > 0.0):
        # The mapped_source GRIB2 recipe: wrap into [-180, 180) and
        # re-sort the columns once.
        wrapped = (longitude + 180.0) % 360.0 - 180.0
        order = np.argsort(wrapped)
        longitude = wrapped[order]
        temperature = temperature[:, order]
        valid = valid[:, order]
    for name, axis in (("latitude", latitude), ("longitude", longitude)):
        steps = np.diff(axis)
        if axis.size < 2 or not np.all(steps > 0.0):
            raise WaterOverlayError(
                f"water-temperature overlay {path} {name} axis is not "
                "strictly monotonic")
        if float(steps.max()) > 1.5 * float(steps.min()):
            raise WaterOverlayError(
                f"water-temperature overlay {path} {name} axis spacing "
                "varies by more than 1.5x -- a crop rotated across the "
                "antimeridian seam is the usual cause; keep the crop's "
                "longitudes continuous")
    return latitude, longitude, temperature, valid


def _load_netcdf(path: Path, variable: str | None) -> WaterTemperatureOverlay:
    # A user-supplied SST/water-temperature field is third-party
    # meteorological data, so it is decoded by the Rust bridge like every
    # other NetCDF source.  Function-local import for the same reason the
    # netCDF4 one was: this module's other formats must load without it.
    from woof import netcdf_bridge

    with netcdf_bridge.open_dataset(path) as dataset:
        names = list(dataset.variables)
        if variable is not None:
            if variable not in dataset.variables:
                raise WaterOverlayError(
                    f"water-temperature overlay {path} has no variable "
                    f"{variable!r}; file variables: {sorted(names)}")
            picked = variable
        else:
            candidates = []
            for name in names:
                var = dataset.variables[name]
                if var.ndim < 2:
                    continue
                standard = str(getattr(var, "standard_name", "")).lower()
                if standard in _STANDARD_NAMES or name.lower() in _VARIABLE_NAMES:
                    candidates.append(name)
            if len(candidates) != 1:
                described = candidates if candidates else sorted(names)
                raise WaterOverlayError(
                    f"water-temperature overlay {path} must identify "
                    "exactly one water-temperature variable by "
                    "standard_name or name; found "
                    f"{len(candidates)} candidate(s) among {described}; "
                    "declare the variable explicitly")
            picked = candidates[0]
        var = dataset.variables[picked]
        if var.ndim < 2:
            raise WaterOverlayError(
                f"water-temperature overlay variable {picked!r} in {path} "
                f"must be at least 2-D, got {var.ndim}-D")
        dims = var.dimensions
        for dim in dims[:-2]:
            if dataset.dimensions[dim].size != 1:
                raise WaterOverlayError(
                    f"water-temperature overlay variable {picked!r} in "
                    f"{path} has leading dimension {dim!r} of size "
                    f"{dataset.dimensions[dim].size}; a single analysis "
                    "time/level is required")
        axes = []
        for dim in dims[-2:]:
            if dim not in dataset.variables:
                raise WaterOverlayError(
                    f"water-temperature overlay {path} lacks a 1-D "
                    f"coordinate variable for dimension {dim!r}")
            coordinate = dataset.variables[dim]
            if coordinate.ndim != 1:
                raise WaterOverlayError(
                    f"water-temperature overlay coordinate {dim!r} in "
                    f"{path} must be 1-D, got {coordinate.ndim}-D")
            axes.append(np.asarray(coordinate[:], dtype=np.float64))
        declared_units = getattr(var, "units", None)
        if declared_units is None:
            raise WaterOverlayError(
                f"water-temperature overlay variable {picked!r} in {path} "
                "declares no units attribute; kelvin or Celsius must be "
                "declared")
        units = _normalize_units(declared_units, path)
        raw = var[...]
        data = np.ma.masked_invalid(np.ma.asarray(raw))
        data = data.reshape(data.shape[-2:] if data.ndim == 2
                            else (-1, *data.shape[-2:]))
        if data.ndim == 3:
            data = data[0]
        valid = ~np.ma.getmaskarray(data)
        temperature = np.asarray(
            np.ma.filled(data, np.nan), dtype=np.float64)
    if units == "C":
        temperature = np.where(valid, temperature + 273.15, temperature)
    latitude, longitude, temperature, valid = _oriented(
        axes[0], axes[1], temperature, valid, path)
    _plausibility_guard(temperature, valid, units, path)
    return WaterTemperatureOverlay(
        path=path, source_format="netcdf", variable=picked,
        declared_units=str(declared_units), latitude=latitude,
        longitude=longitude, temperature_k=temperature, valid=valid)


def _grib2_tool(name: str, explicit) -> Path:
    if explicit is not None:
        return Path(explicit)
    from_env = os.environ.get(f"GPUWM_{name.upper()}")
    if from_env:
        return Path(from_env)
    root = Path(__file__).resolve().parents[2]
    release = root / "tools" / "grib1_bridge" / "target" / "release"
    for candidate in (release / name, release / f"{name}.exe"):
        if candidate.is_file():
            return candidate
    raise WaterOverlayError(
        f"GRIB2 water-temperature overlays need the Rust {name} tool; "
        f"set GPUWM_{name.upper()} or build it with `cargo build "
        "--locked --release --bin grib2_inventory --bin grib2_dump` in "
        "tools/grib1_bridge")


def _load_grib2(path: Path, grib2_inventory, grib2_dump
                ) -> WaterTemperatureOverlay:
    from woof.mapped_source import _grib2_inventory as run_inventory
    from woof.mapped_source import _grib2_records as run_records

    inventory_tool = _grib2_tool("grib2_inventory", grib2_inventory)
    dump_tool = _grib2_tool("grib2_dump", grib2_dump)
    rows = run_inventory(path, inventory_tool)
    matches = [
        row for row in rows
        if (int(row["discipline"]), int(row["category"]),
            int(row["parameter"])) == _GRIB2_WATER_TEMPERATURE
    ]
    if len(matches) != 1:
        found = sorted({
            (int(row["discipline"]), int(row["category"]),
             int(row["parameter"]))
            for row in rows
        })
        raise WaterOverlayError(
            f"water-temperature overlay {path} must contain exactly one "
            f"GRIB2 record with (discipline, category, parameter) == "
            f"{_GRIB2_WATER_TEMPERATURE}; found {len(matches)} among "
            f"{found}")
    index = int(matches[0]["index"])
    records = run_records(path, inventory_tool, dump_tool, {index})
    record = records[0]
    values = np.asarray(record.values, dtype=np.float64)
    valid = np.isfinite(values)
    latitude, longitude, values, valid = _oriented(
        record.latitude, record.longitude, values, valid, path)
    # WMO 4.2-10-3-0 declares kelvin; no per-file units attribute exists.
    _plausibility_guard(values, valid, "K", path)
    return WaterTemperatureOverlay(
        path=path, source_format="grib2",
        variable="discipline=10,category=3,parameter=0",
        declared_units="K", latitude=latitude, longitude=longitude,
        temperature_k=values, valid=valid)


def load_water_temperature_overlay(
        path: str | Path, *, variable: str | None = None,
        grib2_inventory: str | Path | None = None,
        grib2_dump: str | Path | None = None) -> WaterTemperatureOverlay:
    """Load one overlay file, container sniffed by magic, fail-loud."""

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"water-temperature overlay file does not exist: {path}")
    with path.open("rb") as stream:
        magic = stream.read(8)
    if magic.startswith(b"\x89HDF") or magic.startswith(b"CDF"):
        return _load_netcdf(path, variable)
    if magic.startswith(b"GRIB"):
        edition = magic[7] if len(magic) >= 8 else None
        if edition == 2:
            return _load_grib2(path, grib2_inventory, grib2_dump)
        if edition == 1:
            raise WaterOverlayError(
                f"water-temperature overlay {path} is GRIB edition 1; "
                "supply the analysis as netCDF or GRIB2")
        raise WaterOverlayError(
            f"water-temperature overlay {path} declares unsupported GRIB "
            f"edition {edition!r}")
    raise WaterOverlayError(
        f"water-temperature overlay {path} is neither netCDF (HDF5/CDF) "
        f"nor GRIB2 (magic {magic[:4]!r})")


@lru_cache(maxsize=4)
def _cached_overlay_resolved(
        path_key: str, content_key: str,
        variable: str | None) -> WaterTemperatureOverlay:
    del content_key
    return load_water_temperature_overlay(path_key, variable=variable)


def cached_water_temperature_overlay(
        path: str | Path, *,
        variable: str | None = None) -> WaterTemperatureOverlay:
    """Process-lifetime cache keyed by resolved path + size/mtime."""

    resolved = Path(path).resolve()
    stat = resolved.stat()
    content_key = f"size={stat.st_size};mtime_ns={stat.st_mtime_ns}"
    return _cached_overlay_resolved(
        os.fspath(resolved), content_key, variable)


def masked_bilinear_sample(
        overlay: WaterTemperatureOverlay, target_lat, target_lon, *,
        workers: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Sample the overlay at target points; invalid corners are excluded.

    Returns ``(values, covered)``: bilinear temperatures with weights
    renormalized over VALID corners only, and a bool coverage mask.  A
    point outside the overlay bounding box, or whose four corners are
    all invalid, is not covered (``values`` is NaN there).
    """

    latitude = overlay.latitude
    longitude = overlay.longitude
    target_lat = np.asarray(target_lat, dtype=np.float64)
    target_lon = np.asarray(target_lon, dtype=np.float64)
    # The horiz.py midpoint convention: unwrap targets into the frame
    # the overlay's finite axis occupies.
    lon_mid = 0.5 * (longitude[0] + longitude[-1])
    unwrapped = target_lon + 360.0 * np.round(
        (lon_mid - target_lon) / 360.0)
    inside = ((target_lat >= latitude[0]) & (target_lat <= latitude[-1])
              & (unwrapped >= longitude[0]) & (unwrapped <= longitude[-1]))
    y = np.interp(target_lat, latitude, np.arange(latitude.size))
    x = np.interp(unwrapped, longitude, np.arange(longitude.size))
    y0 = np.clip(np.floor(y).astype(np.intp), 0, latitude.size - 2)
    x0 = np.clip(np.floor(x).astype(np.intp), 0, longitude.size - 2)
    fy = np.clip(y - y0, 0.0, 1.0)
    fx = np.clip(x - x0, 0.0, 1.0)
    # The corner blend runs in the Rust preprocessing library
    # (gpuwm_overlay_bilinear_sample_f64), byte-identical to the NumPy
    # loop kept as its test oracle (woof/verify/water_blend_oracle.py).
    from woof.ingest.cpu_backend import water_blend_backend

    return water_blend_backend().overlay_bilinear_sample(
        overlay.temperature_k, overlay.valid, y0, x0, fy, fx, inside,
        workers=workers)


def _source_cell_coordinates(snapshot, rows, cols):
    """Geographic lat/lon of the source cells at ``(rows, cols)``.

    The overlay is a geographic dataset, so the cells it is sampled at
    have to be in degrees.  A geographic source's axes already ARE
    degrees; a projected source's axes are its projection's own, and
    indexing them for a lat/lon reads a plane coordinate as a degree --
    the same pairing defect the coverage receipt carried, one route over.
    The descriptor says which case this is, so nothing here branches on a
    source name and a source without one is unchanged.
    """

    explicit = getattr(snapshot, "source_cell_latlon", None)
    if explicit is not None:
        return explicit(rows, cols)
    from woof.ingest.horiz import declared_source_projection

    projection = declared_source_projection(snapshot)
    if projection is None:
        return snapshot.latitude[rows], snapshot.longitude[cols]
    from woof.mapped_source import declared_projected_source_grid

    parameters = projection["parameters"]
    unit = float(parameters["axis_unit_m"])
    source = declared_projected_source_grid(projection)
    # The inverse of source_coordinate_transform's forward step:
    # LambertGrid coordinates are one-based and axis zero sits on the
    # first grid point, so axis value v is grid coordinate v/d + 1.
    i = (np.asarray(snapshot.longitude[cols], dtype=np.float64)
         * unit / float(parameters["dx_m"])) + 1.0
    j = (np.asarray(snapshot.latitude[rows], dtype=np.float64)
         * unit / float(parameters["dy_m"])) + 1.0
    return source.ij_to_latlon(i, j)


def apply_water_temperature_overlay(snapshot, overlay, *, workers=None):
    """Replace SST/SKINTEMP over covered WATER source cells.

    Returns ``(new_snapshot, receipt)``.  Land cells, uncovered water
    cells, and every other field are value-identical.  The snapshot must
    carry LANDSEA (the water mask authority) and at least one of
    SST/SKINTEMP.
    """

    fields = snapshot.fields
    if "LANDSEA" not in fields:
        raise WaterOverlayError(
            "the water-temperature overlay requires LANDSEA in the source "
            "snapshot to identify water cells")
    replaced_names = [name for name in ("SST", "SKINTEMP")
                      if name in fields]
    if not replaced_names:
        raise WaterOverlayError(
            "the water-temperature overlay found neither SST nor SKINTEMP "
            "in the source snapshot")
    water = np.asarray(fields["LANDSEA"], dtype=np.float64) < 0.5
    rows, cols = np.nonzero(water)
    values, covered = masked_bilinear_sample(
        overlay, *_source_cell_coordinates(snapshot, rows, cols),
        workers=workers)
    replacements = {}
    for name in replaced_names:
        updated = np.array(fields[name], dtype=np.float64)
        updated[rows[covered], cols[covered]] = values[covered]
        replacements[name] = updated
    receipt = {
        "path": str(overlay.path),
        "source_format": overlay.source_format,
        "variable": overlay.variable,
        "declared_units": overlay.declared_units,
        "fields": list(replaced_names),
        "water_cells": int(water.sum()),
        "replaced_cells": int(np.count_nonzero(covered)),
        "fallback_cells": int(np.count_nonzero(~covered)),
    }
    # Replace only the surface fields. Coordinates, projection, vertical
    # metadata, source-window offsets and native adapter type all survive.
    with_fields = getattr(snapshot, "with_fields", None)
    rebuilt = (with_fields(replacements) if with_fields is not None else
               replace(snapshot, fields={**fields, **replacements}))
    return rebuilt, receipt


def overlay_snapshots(snapshots, overlay, *, workers=None):
    """Apply the overlay to a snapshot sequence; None is the identity.

    Returns ``(snapshots, receipt_or_None)``.  With ``overlay=None`` the
    EXACT input object is returned: the absent-configuration path never
    constructs anything.  A configured overlay that intersects none of
    the crop's water is a misconfiguration and refuses by name.
    """

    if overlay is None:
        return snapshots, None
    rebuilt = []
    receipts = []
    for snapshot in snapshots:
        replaced, receipt = apply_water_temperature_overlay(
            snapshot, overlay, workers=workers)
        rebuilt.append(replaced)
        receipts.append(receipt)
    if not receipts:
        raise WaterOverlayError(
            "a water-temperature overlay is configured but there are no "
            "source snapshots to apply it to")
    total_replaced = sum(item["replaced_cells"] for item in receipts)
    if total_replaced == 0:
        first = receipts[0]
        raise WaterOverlayError(
            f"water-temperature overlay {first['path']} does not "
            "intersect any water cell of the source crop "
            f"({first['water_cells']} water cell(s) per snapshot, 0 "
            "covered); the overlay and the forcing describe different "
            "places")
    aggregate = dict(receipts[0])
    aggregate["snapshots"] = len(receipts)
    return tuple(rebuilt), aggregate


def overlay_file_identity(path):
    """Bind the actual overlay bytes without retaining another file copy."""
    path = Path(path).resolve()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


class OverlaySnapshotSequence(Sequence):
    """Apply an analysis on access, retaining metadata and no weather copies."""

    def __init__(self, snapshots, overlay, *, binding=None, workers=None):
        if not len(snapshots):
            raise WaterOverlayError("a water overlay has no source snapshots to apply to")
        self._snapshots = snapshots
        self.overlay = overlay
        #: The preparation's host-step threads for the corner blend.
        self.workers = workers
        self.binding = overlay_file_identity(overlay.path)
        if binding is not None and self.binding != binding:
            raise WaterOverlayError("water-temperature overlay changed after it was loaded")
        self._receipts = {}
        self._verified = False

    def __len__(self):
        return len(self._snapshots)

    @property
    def valid_times(self):
        declared = getattr(self._snapshots, "valid_times", None)
        return (declared if declared is not None else
                tuple(snapshot.valid_time for snapshot in self._snapshots))

    def snapshot_metadata(self, index):
        from woof.ingest.source_metadata import snapshot_metadata
        return snapshot_metadata(self._snapshots, index)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return tuple(self[i] for i in range(*index.indices(len(self))))
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        result, receipt = apply_water_temperature_overlay(
            self._snapshots[index], self.overlay, workers=self.workers)
        self._receipts[index] = receipt
        if len(self._receipts) == len(self) and not self._verified:
            self.verify_complete()
        return result

    def verify_complete(self):
        if len(self._receipts) != len(self):
            raise WaterOverlayError("water overlay was not applied to every prepared forcing time")
        if not sum(row["replaced_cells"] for row in self._receipts.values()):
            raise WaterOverlayError(f"water-temperature overlay {self.overlay.path} does not "
                                    "intersect any water cell of the source crop")
        if overlay_file_identity(self.overlay.path) != self.binding:
            raise WaterOverlayError("water-temperature overlay bytes changed during preparation")
        self._verified = True
        return self.receipt

    @property
    def receipt(self):
        rows = [self._receipts[index] for index in sorted(self._receipts)]
        return {**self.binding, "snapshots": len(rows),
                "per_snapshot": rows}


def load_bound_water_overlay(path):
    """Load one analysis and prove its bytes stayed unchanged during decode."""
    if path is None:
        return None, None
    binding = overlay_file_identity(path)
    overlay = load_water_temperature_overlay(path)
    if overlay_file_identity(path) != binding:
        raise WaterOverlayError("water-temperature overlay bytes changed while being loaded")
    return overlay, binding


def overlay_snapshot_sequence(snapshots, overlay, *, binding=None,
                              workers=None):
    """The absent declaration is identity; an active one stays lazy."""
    return (snapshots if overlay is None else
            OverlaySnapshotSequence(snapshots, overlay, binding=binding,
                                    workers=workers))


def verify_overlay_sequence(snapshots):
    """Finish the declared overlay proof after every forcing time was consumed."""
    return snapshots.verify_complete() if isinstance(snapshots, OverlaySnapshotSequence) else None


def overlay_snapshots_by_time(by_time, overlay):
    """Dict-of-snapshots variant used by the runtime route."""

    if overlay is None:
        return by_time, None
    rebuilt, receipt = overlay_snapshots(tuple(by_time.values()), overlay)
    return dict(zip(by_time.keys(), rebuilt)), receipt


__all__ = [
    "WaterOverlayError", "WaterTemperatureOverlay",
    "apply_water_temperature_overlay", "cached_water_temperature_overlay",
    "load_water_temperature_overlay", "masked_bilinear_sample",
    "overlay_snapshots", "overlay_snapshots_by_time",
    "load_bound_water_overlay", "overlay_snapshot_sequence",
    "verify_overlay_sequence", "overlay_file_identity",
]
