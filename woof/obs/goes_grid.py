"""``gpuwm-obs.goes-grid.v1``: gridded GOES CWP, on disk.

The satellite twin of :mod:`woof.obs.radar_grid`, and deliberately the
same shape of object: a netCDF file bound to one
:class:`~woof.obs.target_grid.TargetGrid` by an identity digest and a
per-array digest table, carrying the observation, its mask, its error
standard deviation, and enough accounting that no drop is silent.

It is a **2-D** product where the radar one is 3-D, because CWP is a
column integral.  The vertical information it does carry is
``obs_level``: the model level each column's single observation is
centred at, which the DA adapter expands into the ``(nz, ny, nx)`` mask
the filter wants.  See :func:`woof.obs.goes_cwp.grid_cwp` for what that
centre does and does not claim.

Three attributes carry the science that must not travel silently:
``join`` (the cross-grid interpolation the bridge refused to do and this
consumer chose), ``error_model`` (labelled UNCALIBRATED, with its
constants), and ``dqf_policy`` (the bridge's condemn mask, carried
forward).  A consumer that reads this file and does not read those three
cannot say how its observations were made.

Both halves are WOOF-ArWen's Rust, and this module imports no Python NetCDF
library at all: :mod:`woof.obs.grid_product` writes the classic container
and :mod:`woof.netcdf_bridge` decodes it back.  The radar twin keeps one
netCDF4 read for its character variables; this product has none.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import uuid

import numpy as np

from woof.obs.goes_cwp import GriddedCwp
from woof.obs.radar_grid import (_canonical_json, _coordinate_digests,
                                  _count_var, _digest, _grid_var, _mask_var,
                                  _obs_var, require_grid_binding)
from woof.obs.target_grid import GridMismatchError, TargetGrid

#: The contract string.  Consumers pin this exact value.
GOES_GRID_SCHEMA = "gpuwm-obs.goes-grid.v1"

#: What this lane can accurately claim.  Stronger wording than the radar
#: product's, for a reason that is not modesty: the radar error model is a
#: documented parameterization, and this one is a set of constants nobody
#: has calibrated (see :class:`woof.obs.goes_cwp.CwpErrorModel`).  No
#: forecast has been scored against a GOES CWP analysis.  Promotion gate:
#: an obs-skill result on the delayed-window scorecard with the error
#: constants A/B'd, decided by the owner.
GOES_GRID_STATUS = "EXPERIMENTAL_UNCALIBRATED_ERROR_MODEL"

_MASS = ("south_north", "west_east")

#: Every variable this schema defines, with the **dimension tuple** it must
#: carry, its storage type, and its units.  The tuple is the essential
#: part for exactly the reason :mod:`woof.obs.radar_grid` gives: a
#: transposed field on a square grid has the right shape and the wrong
#: values, and only the tuple tells them apart.
CANONICAL_VARIABLES: dict[str, tuple[tuple[str, ...], str, str | None]] = {
    "XLAT": (_MASS, "float32", "degree_north"),
    "XLONG": (_MASS, "float32", "degree_east"),
    "HGT": (_MASS, "float32", "m"),
    "cwp_obs": (_MASS, "float32", "g m-2"),
    "cwp_mask": (_MASS, "int8", None),
    "cwp_err": (_MASS, "float32", "g m-2"),
    "cwp_class": (_MASS, "int8", None),
    "cwp_count": (_MASS, "int32", "count"),
    "cwp_pixels": (_MASS, "int32", "count"),
    "cloud_top_height_m": (_MASS, "float32", "m"),
    "obs_level": (_MASS, "int32", None),
}


class GoesGridSchemaError(ValueError):
    """A goes-grid file that does not satisfy the contract."""


def write_goes_grid(path: str | Path, observations: GriddedCwp,
                    grid: TargetGrid, *, valid_time: str,
                    provenance: dict | None = None,
                    overwrite: bool = False) -> dict:
    """Write one ``gpuwm-obs.goes-grid.v1`` file, atomically."""

    from woof.obs.grid_product import (                 # noqa: PLC0415
        open_obs_grid_product)

    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite {path}")
    shape = (grid.ny, grid.nx)
    for name in ("cwp_obs", "cwp_mask", "cwp_err", "cwp_class", "cwp_count",
                 "cwp_pixels", "cloud_top_height_m", "obs_level"):
        array = getattr(observations, name)
        if array.shape != shape:
            raise GridMismatchError(
                f"{name} has shape {array.shape}, target grid is {shape}")

    mask = np.asarray(observations.cwp_mask).astype(bool)
    errors = np.asarray(observations.cwp_err, dtype=np.float64)
    if np.any(mask) and not np.all(np.isfinite(errors[mask])
                                   & (errors[mask] > 0.0)):
        raise GoesGridSchemaError(
            "cwp_err must be finite and strictly positive under cwp_mask; a "
            "zero observation error is an infinite weight and the filter "
            "will divide by it")
    values = np.asarray(observations.cwp_obs, dtype=np.float64)
    if np.any(mask) and not np.all(np.isfinite(values[mask])):
        raise GoesGridSchemaError(
            "cwp_obs is non-finite where cwp_mask says an observation "
            "exists; NaN means no observation everywhere else in this "
            "pipeline and it cannot mean something different here")
    if np.any(mask) and np.any(values[mask] < 0.0):
        raise GoesGridSchemaError(
            "cwp_obs is negative under the mask; a condensate path cannot "
            "be, and clear sky is a genuine 0.0 rather than a small "
            "negative")
    levels = np.asarray(observations.obs_level, dtype=np.int64)
    if np.any(mask) and not np.all((levels[mask] >= 0)
                                   & (levels[mask] < grid.nz)):
        raise GoesGridSchemaError(
            f"obs_level is outside 0..{grid.nz - 1} where cwp_mask says an "
            "observation exists; an observation the filter cannot place is "
            "not an observation")

    identity = grid.identity_sha256()
    coordinates = _coordinate_digests(grid)
    payload_provenance = dict(observations.provenance)
    if provenance:
        payload_provenance["build"] = provenance

    temp = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex[:8]}")
    try:
        with open_obs_grid_product(temp) as dataset:
            dataset.createDimension("south_north", grid.ny)
            dataset.createDimension("west_east", grid.nx)
            dataset.setncatts({
                "schema": GOES_GRID_SCHEMA,
                "status": GOES_GRID_STATUS,
                "valid_time": valid_time,
                "grid_identity_sha256": identity,
                "grid_coordinate_sha256": _canonical_json(coordinates),
                "grid_name": grid.name,
                "grid_source": grid.source,
                "nz": np.int32(grid.nz),
                "MAP_PROJ": np.int32(grid.projection.wrf_map_proj),
                "MAP_PROJ_CHAR": grid.projection.map_proj_char,
                "TRUELAT1": np.float64(grid.truelat1),
                "TRUELAT2": np.float64(grid.truelat2),
                "STAND_LON": np.float64(grid.stand_lon),
                "CEN_LAT": np.float64(grid.projection.cen_lat),
                "CEN_LON": np.float64(grid.projection.cen_lon),
                "DX": np.float64(grid.dx_m),
                "DY": np.float64(grid.dy_m),
                "GRIDTYPE": "C",
                "observable": (
                    "column-integrated cloud water path, g m-2; one "
                    "observation per column, centred at obs_level"),
                "join": _canonical_json(payload_provenance.get("join")),
                "error_model": _canonical_json(
                    payload_provenance.get("error_model")),
                "superob_params": _canonical_json(
                    payload_provenance.get("superob")),
                "dqf_policy": _canonical_json(
                    payload_provenance.get("dqf_policy")),
                "provenance": _canonical_json(payload_provenance),
            })
            _grid_var(dataset, "XLAT", _MASS, grid.lat,
                      "LATITUDE, SOUTH IS NEGATIVE", "degree_north")
            _grid_var(dataset, "XLONG", _MASS, grid.lon,
                      "LONGITUDE, WEST IS NEGATIVE", "degree_east")
            _grid_var(dataset, "HGT", _MASS, grid.terrain_m,
                      "Terrain Height", "m")
            _obs_var(dataset, "cwp_obs", _MASS, observations.cwp_obs,
                     "g m-2", "superobbed cloud water path")
            _mask_var(dataset, "cwp_mask", _MASS, observations.cwp_mask,
                      "1 where cwp_obs is a usable observation")
            _obs_var(dataset, "cwp_err", _MASS, observations.cwp_err,
                     "g m-2",
                     "observation error standard deviation for cwp_obs "
                     "(UNCALIBRATED; see the error_model attribute)")
            _count_var(dataset, "cwp_count", _MASS, observations.cwp_count,
                       "valid satellite pixels averaged into the cell")
            _count_var(dataset, "cwp_pixels", _MASS, observations.cwp_pixels,
                       "satellite pixels that landed in the cell at all")
            _obs_var(dataset, "cloud_top_height_m", _MASS,
                     observations.cloud_top_height_m, "m",
                     "joined ABI ACHA cloud-top height above MSL, cell "
                     "mean over cloudy pixels; NaN where none")
            klass = dataset.createVariable("cwp_class", "i1", _MASS)
            klass.description = ("phase class the cell's error model and "
                                 "the operator's condensate composition "
                                 "were taken from")
            klass.flag_values = np.array([-1, 0, 1, 2], dtype=np.int8)
            klass.flag_meanings = "none clear liquid ice"
            klass.coordinates = "XLONG XLAT"
            klass[:] = np.asarray(observations.cwp_class, dtype=np.int8)
            level = dataset.createVariable("obs_level", "i4", _MASS)
            level.description = (
                "model level the column observation is centred at, or -1. "
                "CWP is a column integral: this is the centre of the "
                "vertical localisation lens, not a height for the water")
            level.coordinates = "XLONG XLAT"
            level[:] = np.asarray(observations.obs_level, dtype=np.int32)

        raw = temp.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        os.replace(temp, path)
    finally:
        if temp.exists():
            temp.unlink()

    return {
        "schema": GOES_GRID_SCHEMA,
        "status": GOES_GRID_STATUS,
        "path": str(path),
        "bytes": len(raw),
        "sha256": digest,
        "grid_identity_sha256": identity,
        "valid_time": valid_time,
        "observations": int(np.count_nonzero(mask)),
        "dims": {"south_north": grid.ny, "west_east": grid.nx,
                 "nz": grid.nz},
    }


def read_goes_grid(path: str | Path, *,
                   expected_grid_identity: str | None = None,
                   expected_grid: TargetGrid | None = None) -> dict:
    """Read a goes-grid file back, checking the contract before the data.

    Same binding discipline as :func:`woof.obs.radar_grid.read_radar_grid`,
    and for the same reason: the identity is a digest over arrays the file
    stores only in part, at float32, and none of them vertical.
    Only a caller holding the grid can reach ``z_w`` -- and here ``z_w`` is
    what ``obs_level`` indexes, so a file bound to the wrong vertical
    structure centres every observation in the wrong layer.
    """

    from woof import netcdf_bridge                       # noqa: PLC0415

    path = Path(path)
    if expected_grid is not None:
        demanded = expected_grid.identity_sha256()
        if (expected_grid_identity is not None
                and not expected_grid.matches_identity(
                    expected_grid_identity)):
            raise GridMismatchError(
                f"expected_grid hashes to {demanded} but "
                f"expected_grid_identity demands {expected_grid_identity}; "
                "the caller is asking for two different grids")
        expected_grid_identity = demanded

    # Reading back the gridded CWP product means decoding observation
    # fields (cwp_obs, its mask and error, obs_level, XLAT/XLONG), so the
    # read goes through the Rust bridge even though woof wrote the file.
    # The WRITE above is Rust too now: "writing is not decoding" was the
    # sentence that kept a whole gridded-product writer on the C library,
    # and the 2026-08-18 audit retired it.
    with netcdf_bridge.open_dataset(path) as dataset:
        schema = getattr(dataset, "schema", None)
        if schema != GOES_GRID_SCHEMA:
            raise GoesGridSchemaError(
                f"{path.name}: schema {schema!r}, expected "
                f"{GOES_GRID_SCHEMA!r}")
        identity = getattr(dataset, "grid_identity_sha256", None)
        if not identity:
            raise GoesGridSchemaError(
                f"{path.name}: no grid_identity_sha256; a gridded "
                "observation set without its grid identity cannot be "
                "assimilated safely")
        if (expected_grid_identity is not None
                and not (expected_grid.matches_identity(identity)
                         if expected_grid is not None
                         else identity == expected_grid_identity)):
            raise GridMismatchError(
                f"{path.name} is bound to grid {identity}, the caller "
                f"requires {expected_grid_identity}")
        _require_structure(path, dataset)
        stored = _require_coordinates(path, dataset)
        if expected_grid is not None:
            from woof.obs.radar_grid import _require_grid  # noqa: PLC0415

            _require_grid(path.name, expected_grid, stored,
                          {name: np.asarray(dataset.variables[name][:])
                           for name in ("XLAT", "XLONG")})

        nz = getattr(dataset, "nz", None)
        if nz is None:
            raise GoesGridSchemaError(
                f"{path.name}: no nz attribute; obs_level indexes a vertical "
                "structure the file must at least state the size of")
        result = {
            "schema": schema,
            "status": getattr(dataset, "status", None),
            "valid_time": getattr(dataset, "valid_time", None),
            "grid_identity_sha256": identity,
            "grid_coordinate_sha256": stored,
            "dims": {"south_north": len(dataset.dimensions["south_north"]),
                     "west_east": len(dataset.dimensions["west_east"]),
                     "nz": int(nz)},
            "join": json.loads(getattr(dataset, "join", "null")),
            "error_model": json.loads(getattr(dataset, "error_model",
                                              "null")),
            "superob_params": json.loads(getattr(dataset, "superob_params",
                                                 "{}")),
            "dqf_policy": json.loads(getattr(dataset, "dqf_policy", "null")),
            "provenance": json.loads(getattr(dataset, "provenance", "{}")),
            "variables": {},
        }
        dataset.set_auto_mask(False)
        for name in CANONICAL_VARIABLES:
            result["variables"][name] = np.asarray(dataset.variables[name][:])

    _require_consistency(path, result)
    return result


def require_goes_grid_binding(document, grid: TargetGrid, *,
                              label: str = "document") -> None:
    """Bind an already-read goes-grid document to a grid in hand."""

    require_grid_binding(document, grid, label=label)
    if int(document["dims"]["nz"]) != int(grid.nz):
        raise GridMismatchError(
            f"{label}: obs_level was assigned against {document['dims']['nz']}"
            f" model levels, the caller's grid has {grid.nz}")


def _require_structure(path: Path, dataset) -> None:
    for dimension in _MASS:
        if dimension not in dataset.dimensions:
            raise GoesGridSchemaError(
                f"{path.name}: missing dimension {dimension!r}")
    missing = [name for name in CANONICAL_VARIABLES
               if name not in dataset.variables]
    if missing:
        raise GoesGridSchemaError(f"{path.name}: missing variables {missing}")
    for name, (dims, dtype, units) in CANONICAL_VARIABLES.items():
        variable = dataset.variables[name]
        if tuple(variable.dimensions) != dims:
            raise GoesGridSchemaError(
                f"{path.name}: {name} is declared over "
                f"{tuple(variable.dimensions)}, this schema defines it over "
                f"{dims}")
        if np.dtype(variable.dtype) != np.dtype(dtype):
            raise GoesGridSchemaError(
                f"{path.name}: {name} is stored as "
                f"{np.dtype(variable.dtype)}, this schema defines it as "
                f"{np.dtype(dtype)}")
        if units is None:
            continue
        declared = getattr(variable, "units", None)
        if declared != units:
            raise GoesGridSchemaError(
                f"{path.name}: {name} declares units {declared!r}, this "
                f"schema defines {units!r}. cwp_err is a standard deviation "
                "in the same units as cwp_obs; a unit that says otherwise "
                "is a different quantity wearing the same name")


def _require_coordinates(path: Path, dataset) -> dict:
    raw = getattr(dataset, "grid_coordinate_sha256", None)
    if not raw:
        raise GoesGridSchemaError(
            f"{path.name}: no grid_coordinate_sha256; the identity string "
            "would be an unbound claim")
    try:
        stored = json.loads(raw)
    except json.JSONDecodeError as error:
        raise GoesGridSchemaError(
            f"{path.name}: grid_coordinate_sha256 is not JSON: "
            f"{error}") from error
    expected_keys = {"XLAT", "XLONG", "HGT", "z_w"}
    if set(stored) != expected_keys:
        raise GoesGridSchemaError(
            f"{path.name}: grid_coordinate_sha256 carries {sorted(stored)}, "
            f"expected {sorted(expected_keys)}")
    for name in ("XLAT", "XLONG", "HGT"):
        actual = _digest(dataset.variables[name][:], np.float32)
        if actual != stored[name]:
            raise GridMismatchError(
                f"{path.name}: the stored {name} hashes to {actual}, but the "
                f"file's own grid_coordinate_sha256 says {stored[name]}")
    return stored


def _require_consistency(path: Path, document: dict) -> None:
    """The file's own mask, errors and levels must agree with each other."""

    variables = document["variables"]
    mask = np.asarray(variables["cwp_mask"]).astype(bool)
    if not np.any(mask):
        return
    nz = int(document["dims"]["nz"])
    errors = np.asarray(variables["cwp_err"], dtype=np.float64)
    if not np.all(np.isfinite(errors[mask]) & (errors[mask] > 0.0)):
        raise GoesGridSchemaError(
            f"{path.name}: cwp_err is not finite and positive everywhere "
            "cwp_mask claims an observation")
    values = np.asarray(variables["cwp_obs"], dtype=np.float64)
    if not np.all(np.isfinite(values[mask]) & (values[mask] >= 0.0)):
        raise GoesGridSchemaError(
            f"{path.name}: cwp_obs is non-finite or negative under the mask")
    levels = np.asarray(variables["obs_level"], dtype=np.int64)
    if not np.all((levels[mask] >= 0) & (levels[mask] < nz)):
        raise GoesGridSchemaError(
            f"{path.name}: obs_level is outside 0..{nz - 1} under the mask")
    classes = np.asarray(variables["cwp_class"], dtype=np.int64)
    if not np.all(np.isin(classes[mask], (0, 1, 2))):
        raise GoesGridSchemaError(
            f"{path.name}: cwp_class is not clear/liquid/ice under the mask; "
            "the operator composes model condensate from this and has no "
            "branch for anything else")
    if np.any(classes[~mask] != -1):
        raise GoesGridSchemaError(
            f"{path.name}: cwp_class is set where cwp_mask is 0")
    if np.any(np.asarray(variables["cwp_class"])[mask] == 0):
        clear = mask & (classes == 0)
        if np.any(values[clear] != 0.0):
            raise GoesGridSchemaError(
                f"{path.name}: a clear-sky cell carries a non-zero cwp_obs. "
                "A clear-sky zero is 0.0 exactly; anything else is a "
                "cloudy observation wearing the clear class, and it would "
                "be given the clear-sky error")
