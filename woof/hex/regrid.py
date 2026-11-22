"""Reusable MPAS cell-to-regular-lat/lon regridding.

Algorithm evidence
------------------
This is an explicitly labelled ``implemented-unverified`` visualization
bridge, not a transcription of an MPAS dynamical operator.  Source and target
locations are mapped to Cartesian points on the unit sphere, a SciPy
``cKDTree`` finds nearest cells, and inverse-distance weights use great-circle
angular distance.  The representation is dateline safe because longitude is
never compared as a scalar.  Tests cover constants, smooth analytic fields,
the +/-180-degree seam, saved-weight round trips, and mutation controls on the
published x1.2562 mesh.  No conservative-remapping or frozen-Fortran oracle
claim is made.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from netCDF4 import Dataset
from numpy.typing import NDArray
from scipy.spatial import cKDTree

from .errors import ConfigurationRefusal


REGRID_EVIDENCE = (
    "implemented-unverified visualization interpolation: unit-sphere cKDTree "
    "neighbors plus great-circle inverse-distance weights; analytic x1.2562 tests; "
    "not conservative and not an MPAS-Fortran operator oracle"
)
REGRID_WEIGHT_SCHEMA = 2
_WEIGHT_SCHEMA = REGRID_WEIGHT_SCHEMA
_SOURCE_FINGERPRINT_SCHEMA = b"mpas-port.regrid-source-coordinates.v2\0"


def _normalize_units(units: str, knob: str) -> str:
    normalized = str(units).strip().lower()
    aliases = {
        "degree": "degrees",
        "deg": "degrees",
        "degrees_east": "degrees",
        "degrees_north": "degrees",
        "radian": "radians",
        "rad": "radians",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in {"degrees", "radians"}:
        raise ConfigurationRefusal(
            knob,
            units,
            "coordinate units must be explicit angular degrees or radians",
            f"{knob}='degrees' or {knob}='radians'",
        )
    return normalized


#: How far past a pole a source latitude may sit and still be read as the pole (about 11 m).
POLE_ROUNDING_TOLERANCE_DEGREES = 1.0e-4


def _angles_to_radians(
    latitude: Any,
    longitude: Any,
    units: str,
    *,
    require_same_shape: bool = True,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    lat = np.asarray(latitude, dtype=np.float64)
    lon = np.asarray(longitude, dtype=np.float64)
    if require_same_shape and lat.shape != lon.shape:
        raise ValueError(f"latitude shape {lat.shape} != longitude shape {lon.shape}")
    if not np.all(np.isfinite(lat)) or not np.all(np.isfinite(lon)):
        raise ValueError("latitude and longitude must be finite")
    normalized = _normalize_units(units, "coordinate_units")
    if normalized == "degrees":
        lat = np.deg2rad(lat)
        lon = np.deg2rad(lon)
    # MPAS stores latCell as float32, and float32(pi/2) is 4.37e-8 rad past the pole: every global mesh with
    # a pole cell carries one. Within POLE_ROUNDING_TOLERANCE_DEGREES of a pole is rounding and is clamped to
    # the pole (as rw_mpas::window does); further out is a broken file and is refused.
    limit = np.pi / 2.0 + np.deg2rad(POLE_ROUNDING_TOLERANCE_DEGREES)
    if np.any(np.abs(lat) > limit):
        raise ValueError("latitude lies outside [-90, 90] degrees by more than the "
                         f"{POLE_ROUNDING_TOLERANCE_DEGREES} degree pole-rounding tolerance")
    lat = np.clip(lat, -np.pi / 2.0, np.pi / 2.0)
    return lat, lon


def _unit_sphere(latitude_radians: NDArray[Any], longitude_radians: NDArray[Any]) -> NDArray[np.float64]:
    cos_lat = np.cos(latitude_radians)
    return np.stack(
        (
            cos_lat * np.cos(longitude_radians),
            cos_lat * np.sin(longitude_radians),
            np.sin(latitude_radians),
        ),
        axis=-1,
    ).astype(np.float64, copy=False)


def _source_fingerprint(latitude: Any, longitude: Any, units: str) -> str:
    """Hash exact input coordinates, not platform-libm-derived Cartesian values."""

    normalized_units = _normalize_units(units, "source_units")
    lat = np.ascontiguousarray(np.asarray(latitude, dtype="<f8").ravel())
    lon = np.ascontiguousarray(np.asarray(longitude, dtype="<f8").ravel())
    if lat.shape != lon.shape:
        raise ValueError("source latitude and longitude shapes differ")
    digest = hashlib.sha256()
    digest.update(_SOURCE_FINGERPRINT_SCHEMA)
    digest.update(normalized_units.encode("ascii") + b"\0")
    digest.update(np.asarray(lat.size, dtype="<u8").tobytes())
    digest.update(lat.tobytes())
    digest.update(lon.tobytes())
    return digest.hexdigest()


def _mesh_coordinates(mesh: object) -> tuple[NDArray[Any], NDArray[Any]]:
    try:
        return np.asarray(getattr(mesh, "latCell")), np.asarray(getattr(mesh, "lonCell"))
    except AttributeError:
        arrays = getattr(mesh, "arrays", None)
        if arrays is None or "latCell" not in arrays or "lonCell" not in arrays:
            raise AttributeError("mesh must provide MPAS latCell and lonCell") from None
        return np.asarray(arrays["latCell"]), np.asarray(arrays["lonCell"])


@dataclass(frozen=True, slots=True)
class RegridWeights:
    """Precomputed sparse cell-to-regular-grid interpolation weights."""

    target_latitude: NDArray[np.float64]
    target_longitude: NDArray[np.float64]
    source_indices: NDArray[np.int64]
    weights: NDArray[np.float64]
    source_count: int
    source_fingerprint: str
    method: str
    power: float = 2.0
    evidence: str = REGRID_EVIDENCE

    def __post_init__(self) -> None:
        object.__setattr__(self, "target_latitude", np.asarray(self.target_latitude, dtype=np.float64))
        object.__setattr__(self, "target_longitude", np.asarray(self.target_longitude, dtype=np.float64))
        object.__setattr__(self, "source_indices", np.asarray(self.source_indices, dtype=np.int64))
        object.__setattr__(self, "weights", np.asarray(self.weights, dtype=np.float64))
        self.validate()

    @property
    def shape(self) -> tuple[int, int]:
        return (self.target_latitude.size, self.target_longitude.size)

    @property
    def n_neighbors(self) -> int:
        return int(self.source_indices.shape[-1])

    @property
    def indices(self) -> NDArray[np.int64]:
        """Short compatibility alias for :attr:`source_indices`."""

        return self.source_indices

    @property
    def latitude(self) -> NDArray[np.float64]:
        return self.target_latitude

    @property
    def longitude(self) -> NDArray[np.float64]:
        return self.target_longitude

    def validate(self) -> "RegridWeights":
        if self.target_latitude.ndim != 1 or self.target_longitude.ndim != 1:
            raise ValueError("target latitude and longitude must be one-dimensional regular-grid axes")
        if not self.target_latitude.size or not self.target_longitude.size:
            raise ValueError("target latitude and longitude axes must be non-empty")
        if np.any(np.abs(self.target_latitude) > 90.0 + 1.0e-10):
            raise ValueError("stored target latitude lies outside [-90, 90] degrees")
        expected_prefix = (*self.shape,)
        if self.source_indices.ndim != 3 or self.source_indices.shape[:2] != expected_prefix:
            raise ValueError(
                f"source_indices shape {self.source_indices.shape} must begin with {expected_prefix}"
            )
        if self.weights.shape != self.source_indices.shape:
            raise ValueError("source_indices and weights shapes differ")
        if self.source_indices.shape[-1] < 1:
            raise ValueError("at least one source neighbor is required")
        if self.source_count < 1:
            raise ValueError("source_count must be positive")
        if np.any(self.source_indices < 0) or np.any(self.source_indices >= self.source_count):
            raise ValueError("source_indices contain an out-of-range cell")
        if not np.all(np.isfinite(self.weights)) or np.any(self.weights < 0.0):
            raise ValueError("regrid weights must be finite and nonnegative")
        sums = np.sum(self.weights, axis=-1)
        if not np.allclose(sums, 1.0, rtol=0.0, atol=2.0e-13):
            raise ValueError("regrid weights do not sum to one at every target")
        if self.method not in {"nearest", "inverse_distance"}:
            raise ValueError(f"unsupported stored regrid method {self.method!r}")
        if not np.isfinite(self.power) or self.power <= 0.0:
            raise ValueError("inverse-distance power must be finite and positive")
        if len(self.source_fingerprint) != 64:
            raise ValueError("source_fingerprint is not a SHA-256 digest")
        return self

    def validate_source(
        self,
        source_or_mesh: object,
        source_longitude: Any | None = None,
        *,
        source_units: str = "radians",
    ) -> "RegridWeights":
        """Refuse reuse against a reordered or different source mesh."""

        if source_longitude is None and (
            hasattr(source_or_mesh, "latCell") or hasattr(source_or_mesh, "arrays")
        ):
            source_latitude, source_longitude = _mesh_coordinates(source_or_mesh)
        elif source_longitude is None:
            raise ValueError("source_longitude is required when validating raw coordinates")
        else:
            source_latitude = source_or_mesh
        lat = np.asarray(source_latitude)
        lon = np.asarray(source_longitude)
        if lat.size != self.source_count or lon.size != self.source_count:
            raise ValueError(
                f"weight source_count={self.source_count}, coordinates have "
                f"{lat.size}/{lon.size} entries"
            )
        actual = _source_fingerprint(lat, lon, source_units)
        if actual != self.source_fingerprint:
            raise ValueError("regrid weights were built for a different source mesh or cell ordering")
        return self

    def apply(
        self,
        values: Any,
        *,
        cell_axis: int = -1,
        nan_policy: str = "propagate",
        fill_value: float = np.nan,
    ) -> NDArray[Any]:
        """Apply weights to ``(..., nCells)`` with arbitrary leading axes.

        ``nan_policy='propagate'`` emits NaN if a contributing neighbor is
        NaN.  ``'omit'`` renormalizes the remaining finite weights per target
        and per leading-index slice, using ``fill_value`` if none remain.
        ``'raise'`` refuses sampled NaNs.  Zero-weight exact-match neighbors
        never contaminate a result.
        """

        self.validate()
        if nan_policy not in {"propagate", "omit", "raise"}:
            raise ConfigurationRefusal(
                "nan_policy",
                nan_policy,
                "only propagate, omit-with-renormalization, and raise semantics are defined",
                "nan_policy='propagate', 'omit', or 'raise'",
            )
        source = np.asarray(values)
        if source.ndim < 1:
            raise ValueError("regridded values must have a cell axis")
        axis = np.lib.array_utils.normalize_axis_index(cell_axis, source.ndim)
        if source.shape[axis] != self.source_count:
            raise ValueError(
                f"cell axis has {source.shape[axis]} entries, weights require {self.source_count}"
            )
        if source.dtype.kind not in "fiu":
            raise TypeError(f"regridding requires numeric values, got {source.dtype}")
        output_dtype = source.dtype if source.dtype.kind == "f" else np.dtype("float64")
        moved = np.moveaxis(source, axis, -1).astype(output_dtype, copy=False)
        sampled = moved[..., self.source_indices]
        weight_dtype = np.dtype(output_dtype)
        local_weights = self.weights.astype(weight_dtype, copy=False)
        contributing = local_weights > 0.0
        isnan = np.isnan(sampled)

        if nan_policy == "raise" and np.any(isnan & contributing):
            raise ValueError("sampled source values contain NaN under nan_policy='raise'")
        if nan_policy in {"propagate", "raise"}:
            clean_samples = np.where(contributing, sampled, np.asarray(0.0, dtype=output_dtype))
            result = np.sum(clean_samples * local_weights, axis=-1, dtype=output_dtype)
            if nan_policy == "propagate":
                result = np.where(np.any(isnan & contributing, axis=-1), np.nan, result)
        else:
            valid = (~isnan) & contributing
            effective = np.where(valid, local_weights, np.asarray(0.0, dtype=output_dtype))
            denominator = np.sum(effective, axis=-1, dtype=output_dtype)
            numerator = np.sum(
                np.where(valid, sampled, np.asarray(0.0, dtype=output_dtype)) * effective,
                axis=-1,
                dtype=output_dtype,
            )
            with np.errstate(divide="ignore", invalid="ignore"):
                result = numerator / denominator
            result = np.where(denominator > 0.0, result, np.asarray(fill_value, dtype=output_dtype))

        # Neighbor indexing placed the two target axes at the end.  Replace
        # the original cell axis in-place so explicit cell_axis remains useful.
        n_other = source.ndim - 1
        permutation = (
            *range(axis),
            n_other,
            n_other + 1,
            *range(axis, n_other),
        )
        if permutation != tuple(range(result.ndim)):
            result = np.transpose(result, permutation)
        return np.asarray(result, dtype=output_dtype)

    def save(self, path: str | Path, *, overwrite: bool = False) -> Path:
        """Save deterministic weights as ``.npz`` or inspectable NetCDF."""

        self.validate()
        destination = Path(path).expanduser().resolve()
        if destination.exists() and not overwrite:
            raise FileExistsError(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.suffix.lower() in {".nc", ".nc4", ".netcdf"}:
            with Dataset(destination, "w", format="NETCDF4") as dataset:
                dataset.createDimension("lat", self.shape[0])
                dataset.createDimension("lon", self.shape[1])
                dataset.createDimension("nNeighbors", self.n_neighbors)
                dataset.schema_version = _WEIGHT_SCHEMA
                dataset.method = self.method
                dataset.power = self.power
                dataset.source_count = self.source_count
                dataset.source_fingerprint = self.source_fingerprint
                dataset.evidence = self.evidence
                lat = dataset.createVariable("lat", "f8", ("lat",))
                lon = dataset.createVariable("lon", "f8", ("lon",))
                indices = dataset.createVariable("source_index", "i8", ("lat", "lon", "nNeighbors"))
                weights = dataset.createVariable("weight", "f8", ("lat", "lon", "nNeighbors"))
                lat.units = "degrees_north"
                lon.units = "degrees_east"
                indices.start_index = 0
                weights.normalization = "sum(weight, nNeighbors) = 1"
                lat[:] = self.target_latitude
                lon[:] = self.target_longitude
                indices[:] = self.source_indices
                weights[:] = self.weights
        else:
            with destination.open("wb") as handle:
                np.savez_compressed(
                    handle,
                    schema_version=np.asarray(_WEIGHT_SCHEMA, dtype=np.int64),
                    target_latitude=self.target_latitude,
                    target_longitude=self.target_longitude,
                    source_indices=self.source_indices,
                    weights=self.weights,
                    source_count=np.asarray(self.source_count, dtype=np.int64),
                    source_fingerprint=np.asarray(self.source_fingerprint),
                    method=np.asarray(self.method),
                    power=np.asarray(self.power, dtype=np.float64),
                    evidence=np.asarray(self.evidence),
                )
        return destination

    @classmethod
    def load(cls, path: str | Path) -> "RegridWeights":
        source = Path(path).expanduser().resolve(strict=True)
        if source.suffix.lower() in {".nc", ".nc4", ".netcdf"}:
            with Dataset(source) as dataset:
                version = int(getattr(dataset, "schema_version", -1))
                if version != _WEIGHT_SCHEMA:
                    raise ValueError(f"weight schema {version} != supported {_WEIGHT_SCHEMA}")
                return cls(
                    target_latitude=np.asarray(dataset.variables["lat"][:]),
                    target_longitude=np.asarray(dataset.variables["lon"][:]),
                    source_indices=np.asarray(dataset.variables["source_index"][:]),
                    weights=np.asarray(dataset.variables["weight"][:]),
                    source_count=int(dataset.source_count),
                    source_fingerprint=str(dataset.source_fingerprint),
                    method=str(dataset.method),
                    power=float(dataset.power),
                    evidence=str(dataset.evidence),
                )
        with np.load(source, allow_pickle=False) as archive:
            version = int(np.asarray(archive["schema_version"]).item())
            if version != _WEIGHT_SCHEMA:
                raise ValueError(f"weight schema {version} != supported {_WEIGHT_SCHEMA}")
            return cls(
                target_latitude=archive["target_latitude"],
                target_longitude=archive["target_longitude"],
                source_indices=archive["source_indices"],
                weights=archive["weights"],
                source_count=int(np.asarray(archive["source_count"]).item()),
                source_fingerprint=str(np.asarray(archive["source_fingerprint"]).item()),
                method=str(np.asarray(archive["method"]).item()),
                power=float(np.asarray(archive["power"]).item()),
                evidence=str(np.asarray(archive["evidence"]).item()),
            )


@dataclass(frozen=True, slots=True)
class LatLonRegridder:
    """Small object wrapper around reusable :class:`RegridWeights`."""

    weights: RegridWeights

    @classmethod
    def from_mesh(
        cls,
        mesh: object,
        target_latitude: Any,
        target_longitude: Any,
        *,
        method: str = "inverse_distance",
        neighbors: int = 4,
        power: float = 2.0,
        target_units: str = "degrees",
    ) -> "LatLonRegridder":
        return cls(
            build_regrid_weights(
                mesh,
                target_latitude=target_latitude,
                target_longitude=target_longitude,
                method=method,
                neighbors=neighbors,
                power=power,
                target_units=target_units,
            )
        )

    @classmethod
    def load(cls, path: str | Path) -> "LatLonRegridder":
        return cls(RegridWeights.load(path))

    def save(self, path: str | Path, *, overwrite: bool = False) -> Path:
        return self.weights.save(path, overwrite=overwrite)

    def __call__(self, values: Any, **kwargs: Any) -> NDArray[Any]:
        return self.weights.apply(values, **kwargs)

    def regrid(self, values: Any, **kwargs: Any) -> NDArray[Any]:
        return self.weights.apply(values, **kwargs)


MeshRegridder = LatLonRegridder


def build_regrid_weights(
    source_latitude_or_mesh: Any,
    source_longitude: Any | None = None,
    target_latitude: Any | None = None,
    target_longitude: Any | None = None,
    *,
    method: str = "inverse_distance",
    neighbors: int = 4,
    power: float = 2.0,
    source_units: str = "radians",
    target_units: str = "degrees",
) -> RegridWeights:
    """Precompute spherical nearest or inverse-distance interpolation.

    Raw-coordinate form is ``(source_lat, source_lon, target_lat, target_lon)``.
    Mesh form accepts either ``(mesh, target_lat, target_lon)`` positionally or
    ``(mesh, target_latitude=..., target_longitude=...)``.  Target coordinates
    must be one-dimensional axes; the Cartesian KD tree makes longitude wrap
    and the dateline seam exact rather than special-cased.
    """

    is_mesh = hasattr(source_latitude_or_mesh, "latCell") or hasattr(
        source_latitude_or_mesh, "arrays"
    )
    if is_mesh:
        cell_latitude, cell_longitude = _mesh_coordinates(source_latitude_or_mesh)
        # Three-positional mesh spelling: (mesh, target_lat, target_lon).
        if target_longitude is None and source_longitude is not None and target_latitude is not None:
            target_longitude = target_latitude
            target_latitude = source_longitude
        elif source_longitude is not None:
            raise TypeError(
                "source_longitude is not used for mesh input; pass target_latitude and target_longitude"
            )
        source_latitude = cell_latitude
        source_longitude = cell_longitude
        source_units = "radians"
    else:
        source_latitude = source_latitude_or_mesh
        if source_longitude is None:
            raise TypeError("source_longitude is required for raw source coordinates")
    if target_latitude is None or target_longitude is None:
        raise TypeError("target_latitude and target_longitude are required")

    normalized_method = str(method).strip().lower().replace("-", "_")
    aliases = {"idw": "inverse_distance", "nearest_neighbor": "nearest"}
    normalized_method = aliases.get(normalized_method, normalized_method)
    if normalized_method not in {"nearest", "inverse_distance"}:
        raise ConfigurationRefusal(
            "method",
            method,
            "only spherical nearest-neighbor and inverse-distance interpolation are implemented",
            "method='nearest' or method='inverse_distance'",
        )
    if not isinstance(neighbors, (int, np.integer)) or int(neighbors) < 1:
        raise ValueError("neighbors must be a positive integer")
    if not np.isfinite(power) or power <= 0.0:
        raise ValueError("power must be finite and positive")

    source_lat_rad, source_lon_rad = _angles_to_radians(
        source_latitude, source_longitude, source_units
    )
    if source_lat_rad.ndim != 1:
        source_lat_rad = source_lat_rad.ravel()
        source_lon_rad = source_lon_rad.ravel()
    source_xyz = _unit_sphere(source_lat_rad, source_lon_rad)
    if source_xyz.shape[0] == 0:
        raise ValueError("source coordinates must be non-empty")
    if np.unique(source_xyz, axis=0).shape[0] != source_xyz.shape[0]:
        # Duplicate points are mathematically supportable, but their arbitrary
        # KD-tree ordering makes saved weights needlessly non-reproducible.
        raise ValueError("source coordinates contain duplicate unit-sphere locations")

    target_lat = np.asarray(target_latitude, dtype=np.float64)
    target_lon = np.asarray(target_longitude, dtype=np.float64)
    if target_lat.ndim != 1 or target_lon.ndim != 1:
        raise ValueError("target latitude and longitude must be one-dimensional axes")
    target_lat_rad, _ = _angles_to_radians(
        target_lat,
        np.zeros_like(target_lat),
        target_units,
    )
    _, target_lon_rad = _angles_to_radians(
        np.zeros_like(target_lon),
        target_lon,
        target_units,
    )
    units = _normalize_units(target_units, "target_units")
    target_lat_degrees = np.rad2deg(target_lat_rad) if units == "radians" else target_lat.copy()
    target_lon_degrees = np.rad2deg(target_lon_rad) if units == "radians" else target_lon.copy()
    lat_grid, lon_grid = np.meshgrid(target_lat_rad, target_lon_rad, indexing="ij")
    target_xyz = _unit_sphere(lat_grid, lon_grid).reshape(-1, 3)

    k = 1 if normalized_method == "nearest" else min(int(neighbors), source_xyz.shape[0])
    chord, indices = cKDTree(source_xyz).query(target_xyz, k=k, workers=1)
    chord = np.asarray(chord, dtype=np.float64).reshape(-1, k)
    indices = np.asarray(indices, dtype=np.int64).reshape(-1, k)
    if normalized_method == "nearest":
        weight = np.ones_like(chord)
    else:
        angular = 2.0 * np.arcsin(np.clip(chord * 0.5, 0.0, 1.0))
        exact = angular <= 8.0 * np.finfo(np.float64).eps
        weight = np.empty_like(angular)
        exact_rows = np.any(exact, axis=1)
        if np.any(exact_rows):
            exact_count = np.sum(exact[exact_rows], axis=1, keepdims=True)
            weight[exact_rows] = exact[exact_rows] / exact_count
        ordinary = ~exact_rows
        if np.any(ordinary):
            inverse = np.power(angular[ordinary], -float(power))
            weight[ordinary] = inverse / np.sum(inverse, axis=1, keepdims=True)

    grid_shape = (target_lat.size, target_lon.size, k)
    return RegridWeights(
        target_latitude=target_lat_degrees,
        target_longitude=target_lon_degrees,
        source_indices=indices.reshape(grid_shape),
        weights=weight.reshape(grid_shape),
        source_count=source_xyz.shape[0],
        source_fingerprint=_source_fingerprint(
            source_latitude, source_longitude, source_units
        ),
        method=normalized_method,
        power=float(power),
    )


def precompute_regrid_weights(
    mesh: object,
    target_latitude: Any,
    target_longitude: Any,
    **kwargs: Any,
) -> RegridWeights:
    return build_regrid_weights(
        mesh,
        target_latitude=target_latitude,
        target_longitude=target_longitude,
        **kwargs,
    )


def regrid_cell_field(values: Any, weights: RegridWeights, **kwargs: Any) -> NDArray[Any]:
    return weights.apply(values, **kwargs)


apply_regrid_weights = regrid_cell_field
compute_regrid_weights = build_regrid_weights


def save_regrid_weights(weights: RegridWeights, path: str | Path, *, overwrite: bool = False) -> Path:
    return weights.save(path, overwrite=overwrite)


def load_regrid_weights(path: str | Path) -> RegridWeights:
    return RegridWeights.load(path)


@dataclass(frozen=True, slots=True)
class LatLonField:
    """A regular-grid field with optional axis names and NetCDF metadata."""

    values: NDArray[Any] | Sequence[Any]
    dimensions: tuple[str, ...] | None = None
    attrs: Mapping[str, Any] = field(default_factory=dict)


def _datetime(value: datetime | str | np.datetime64) -> datetime:
    if isinstance(value, datetime):
        result = value
    elif isinstance(value, np.datetime64):
        if np.isnat(value):
            raise ValueError("valid_time must not be NaT")
        micros = value.astype("datetime64[us]").astype(np.int64)
        result = datetime.fromtimestamp(float(micros) / 1_000_000.0, timezone.utc)
    else:
        text = str(value).strip()
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        result = datetime.fromisoformat(text.replace("_", "T", 1))
    if result.tzinfo is not None:
        result = result.astimezone(timezone.utc).replace(tzinfo=None)
    return result.replace(microsecond=0)


def _times(value: Any | None) -> tuple[datetime, ...] | None:
    if value is None:
        return None
    if isinstance(value, (datetime, str, np.datetime64)):
        return (_datetime(value),)
    result = tuple(_datetime(item) for item in value)
    if not result:
        raise ValueError("valid_time must contain at least one record")
    return result


def _field_value(raw: Any) -> tuple[NDArray[Any], tuple[str, ...] | None, dict[str, Any]]:
    if isinstance(raw, LatLonField):
        return np.asarray(raw.values), raw.dimensions, dict(raw.attrs)
    # HistoryField is deliberately recognized structurally to avoid coupling
    # the visualization bridge to the history writer.
    if hasattr(raw, "values") and hasattr(raw, "dimensions") and hasattr(raw, "attrs"):
        return np.asarray(raw.values), raw.dimensions, dict(raw.attrs)
    return np.asarray(raw), None, {}


def _netcdf_dtype(array: NDArray[Any], file_format: str) -> np.dtype[Any]:
    dtype = np.dtype(array.dtype)
    if dtype.kind == "f" and dtype.itemsize in (4, 8):
        return dtype
    if dtype.kind == "b":
        return np.dtype("int8")
    if dtype.kind in "iu":
        if file_format.startswith("NETCDF3") and dtype.itemsize > 4:
            return np.dtype("int32")
        return dtype
    raise TypeError(f"unsupported renderer field dtype {dtype}")


def write_latlon_netcdf(
    path: str | Path,
    fields: Mapping[str, Any],
    latitude: Any | None = None,
    longitude: Any | None = None,
    *,
    latitudes: Any | None = None,
    longitudes: Any | None = None,
    valid_time: Any | None = None,
    initial_time: Any | None = None,
    field_dimensions: Mapping[str, Sequence[str]] | None = None,
    field_attrs: Mapping[str, Mapping[str, Any]] | None = None,
    global_attrs: Mapping[str, Any] | None = None,
    clobber: bool = False,
    file_format: str = "NETCDF4",
    regrid_algorithm: str | None = None,
    regrid_evidence: str | None = None,
) -> Path:
    """Write a compact CF regular-lat/lon file ready for product renderers.

    Field arrays end in ``(lat, lon)`` and may have arbitrary leading axes.
    When ``valid_time`` is supplied, a missing ``Time`` axis is broadcast into
    one or more records.  Native float32/float64 storage is preserved.
    """

    if latitude is not None and latitudes is not None:
        raise TypeError("pass latitude or latitudes, not both")
    if longitude is not None and longitudes is not None:
        raise TypeError("pass longitude or longitudes, not both")
    latitude = latitudes if latitude is None else latitude
    longitude = longitudes if longitude is None else longitude
    if latitude is None or longitude is None:
        raise TypeError("latitude and longitude axes are required")
    lat = np.asarray(latitude, dtype=np.float64)
    lon = np.asarray(longitude, dtype=np.float64)
    if lat.ndim != 1 or lon.ndim != 1 or not lat.size or not lon.size:
        raise ValueError("latitude and longitude must be non-empty one-dimensional axes")
    if not np.all(np.isfinite(lat)) or not np.all(np.isfinite(lon)):
        raise ValueError("latitude and longitude must be finite")
    if np.any(np.abs(lat) > 90.0 + 1.0e-10):
        raise ValueError("latitude lies outside [-90, 90] degrees")
    if file_format not in {"NETCDF4", "NETCDF3_64BIT_OFFSET"}:
        raise ConfigurationRefusal(
            "file_format",
            file_format,
            "the renderer bridge implements serial NetCDF-4 and 64-bit-offset NetCDF",
            "file_format='NETCDF4'",
        )

    timestamps = _times(valid_time)
    reference = (
        _datetime(initial_time)
        if initial_time is not None
        else (timestamps[0] if timestamps is not None else None)
    )
    destination = Path(path).expanduser().resolve()
    if destination.exists() and not clobber:
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)

    sizes: dict[str, int] = {"lat": lat.size, "lon": lon.size}
    prepared: dict[str, tuple[NDArray[Any], tuple[str, ...], dict[str, Any]]] = {}
    for name, raw in fields.items():
        if name in {"lat", "lon", "latitude", "longitude", "Time", "xtime"}:
            raise ValueError(f"renderer field name {name!r} is reserved")
        array, dimensions, attrs = _field_value(raw)
        if array.ndim < 2 or array.shape[-2:] != (lat.size, lon.size):
            raise ValueError(
                f"{name}: final axes {array.shape[-2:]} != regular grid {(lat.size, lon.size)}"
            )
        if field_dimensions is not None and name in field_dimensions:
            requested = tuple(field_dimensions[name])
            if dimensions is not None and tuple(dimensions) != requested:
                raise ValueError(f"{name}: field dimension declarations disagree")
            dimensions = requested
        if dimensions is None:
            leading_count = array.ndim - 2
            leading: list[str] = []
            if timestamps is not None and leading_count and len(timestamps) > 1 and array.shape[0] == len(timestamps):
                leading.append("Time")
                leading_count -= 1
            if leading_count == 1:
                leading.append("level")
            else:
                leading.extend(f"{name}_dim_{i}" for i in range(leading_count))
            dimensions = (*leading, "lat", "lon")
        else:
            dimensions = tuple("lat" if d == "latitude" else "lon" if d == "longitude" else d for d in dimensions)
        if len(dimensions) != array.ndim:
            raise ValueError(f"{name}: {len(dimensions)} dimensions for shape {array.shape}")
        if dimensions[-2:] != ("lat", "lon"):
            raise ValueError(f"{name}: final dimensions must be ('lat', 'lon')")
        if timestamps is not None and "Time" not in dimensions:
            array = np.broadcast_to(array, (len(timestamps), *array.shape))
            dimensions = ("Time", *dimensions)
        if timestamps is None and "Time" in dimensions:
            raise ValueError(f"{name}: Time dimension requires valid_time")
        for axis, dim in enumerate(dimensions):
            if dim == "Time":
                if array.shape[axis] != len(timestamps or ()):
                    raise ValueError(f"{name}: Time axis length does not match valid_time")
                continue
            old = sizes.get(dim)
            if old is not None and old != array.shape[axis]:
                raise ValueError(f"{name}: inconsistent dimension {dim}={array.shape[axis]} != {old}")
            sizes[dim] = int(array.shape[axis])
        metadata = dict(attrs)
        if field_attrs and name in field_attrs:
            metadata.update(field_attrs[name])
        metadata.setdefault("coordinates", "lat lon")
        prepared[name] = (np.asarray(array), tuple(dimensions), metadata)

    with Dataset(destination, "w", format=file_format) as dataset:
        if timestamps is not None:
            dataset.createDimension("Time", None)
            dataset.createDimension("StrLen", 64)
        for name, size in sizes.items():
            dataset.createDimension(name, size)
        dataset.setncatts(
            {
                "Conventions": "CF-1.10",
                "title": "MPAS field(s) regridded to a regular latitude-longitude grid",
                "source": "hexcore.regrid",
                "grid_type": "regular_ll",
                "regrid_algorithm": regrid_algorithm or "unspecified/already-gridded",
                "regrid_evidence": regrid_evidence or REGRID_EVIDENCE,
            }
        )
        if global_attrs:
            dataset.setncatts({k: v.item() if isinstance(v, np.generic) else v for k, v in global_attrs.items()})

        lat_var = dataset.createVariable("lat", "f8", ("lat",))
        lon_var = dataset.createVariable("lon", "f8", ("lon",))
        lat_var.setncatts({"standard_name": "latitude", "long_name": "latitude", "units": "degrees_north", "axis": "Y"})
        lon_var.setncatts({"standard_name": "longitude", "long_name": "longitude", "units": "degrees_east", "axis": "X"})
        lat_var[:] = lat
        lon_var[:] = lon
        # Descriptive aliases make the bridge easy for renderers that request
        # full coordinate names while dimensions remain the conventional lat/lon.
        latitude_var = dataset.createVariable("latitude", "f8", ("lat",))
        longitude_var = dataset.createVariable("longitude", "f8", ("lon",))
        latitude_var.setncatts({"standard_name": "latitude", "units": "degrees_north"})
        longitude_var.setncatts({"standard_name": "longitude", "units": "degrees_east"})
        latitude_var[:] = lat
        longitude_var[:] = lon

        if timestamps is not None and reference is not None:
            time_var = dataset.createVariable("Time", "f8", ("Time",))
            time_var.setncatts(
                {
                    "standard_name": "time",
                    "long_name": "valid time",
                    "calendar": "gregorian",
                    "units": f"seconds since {reference:%Y-%m-%d %H:%M:%S}",
                }
            )
            time_var[:] = [(stamp - reference).total_seconds() for stamp in timestamps]
            xtime = dataset.createVariable("xtime", "S1", ("Time", "StrLen"))
            chars = np.full((len(timestamps), 64), b" ", dtype="S1")
            for row, stamp in enumerate(timestamps):
                encoded = stamp.strftime("%Y-%m-%d_%H:%M:%S").encode("ascii")
                chars[row, : len(encoded)] = np.frombuffer(encoded, dtype="S1")
            xtime[:] = chars

        for name, (array, dimensions, attrs) in prepared.items():
            dtype = _netcdf_dtype(array, file_format)
            fill_value = attrs.get("_FillValue")
            kwargs: dict[str, Any] = {}
            if fill_value is not None:
                kwargs["fill_value"] = np.asarray(fill_value, dtype=dtype).item()
            variable = dataset.createVariable(name, dtype, dimensions, **kwargs)
            variable.setncatts(
                {
                    key: value.item() if isinstance(value, np.generic) else value
                    for key, value in attrs.items()
                    if key != "_FillValue" and value is not None
                }
            )
            variable[:] = np.asarray(array, dtype=dtype)
    return destination


def write_regridded_netcdf(
    path: str | Path,
    weights: RegridWeights,
    fields: Mapping[str, Any],
    *,
    cell_axis: int | Mapping[str, int] = -1,
    nan_policy: str = "propagate",
    **writer_options: Any,
) -> Path:
    """Apply saved weights to cell fields and write the renderer-ready file."""

    writer_options = dict(writer_options)
    provenance = dict(writer_options.pop("global_attrs", {}) or {})
    provenance.update(
        {
            "regrid_source_fingerprint": weights.source_fingerprint,
            "regrid_source_count": weights.source_count,
            "regrid_neighbor_count": weights.n_neighbors,
        }
    )
    gridded: dict[str, LatLonField] = {}
    for name, raw in fields.items():
        array, dimensions, attrs = _field_value(raw)
        axis = cell_axis[name] if isinstance(cell_axis, Mapping) else cell_axis
        normalized_axis = np.lib.array_utils.normalize_axis_index(axis, array.ndim)
        values = weights.apply(array, cell_axis=normalized_axis, nan_policy=nan_policy)
        output_dimensions: tuple[str, ...] | None = None
        if dimensions is not None:
            dims = list(dimensions)
            dims[normalized_axis : normalized_axis + 1] = ["lat", "lon"]
            output_dimensions = tuple(dims)
        gridded[name] = LatLonField(values, output_dimensions, attrs)
    return write_latlon_netcdf(
        path,
        gridded,
        weights.target_latitude,
        weights.target_longitude,
        regrid_algorithm=(
            "spherical nearest-neighbor"
            if weights.method == "nearest"
            else f"spherical inverse-distance, k={weights.n_neighbors}, power={weights.power:g}"
        ),
        regrid_evidence=weights.evidence,
        global_attrs=provenance,
        **writer_options,
    )


write_renderer_netcdf = write_regridded_netcdf
write_latlon_file = write_latlon_netcdf


__all__ = [
    "LatLonField",
    "LatLonRegridder",
    "MeshRegridder",
    "REGRID_EVIDENCE",
    "REGRID_WEIGHT_SCHEMA",
    "RegridWeights",
    "apply_regrid_weights",
    "build_regrid_weights",
    "compute_regrid_weights",
    "load_regrid_weights",
    "precompute_regrid_weights",
    "regrid_cell_field",
    "save_regrid_weights",
    "write_latlon_netcdf",
    "write_latlon_file",
    "write_regridded_netcdf",
    "write_renderer_netcdf",
]
