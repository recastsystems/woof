"""WOOF's selectable column suite behind the v8.4.1 seam, as a backend row.

The frozen row (:mod:`woof.hex.cuda_arwen_physics_v841`) binds woof's
``mpas_column_batch``, whose phase 1 is hard-wired to YSU, the revised MM5
surface layer, Noah-MP on four soil layers and no drag.  This module binds
the engine's selectable suite instead (``woof.core.mpas_column_suite``),
with the selector values of a NAMED PROFILE read from
``data/profiles/physics-suites.v1.json``.  The first profile,
``mynn-ruc-gsl``, is WOOF's HRRR-family set: aerosol-aware Thompson, the MYNN
surface layer, MYNN-EDMF, RUC on nine soil levels and the GSL orographic drag
in the PBL slot, fed by the mesh's orographic statistics file.

WHAT DOES NOT MOVE.  This is a second row, not an edit of the first: the
frozen adapter, its sixteen-file engine pin, its constructor, its execution
digests and every contract digest stay byte-identical.  The class below
subclasses the frozen backend and overrides only what a selected slot
changes -- which engine factory is loaded and how its bytes are recorded,
the sealed constructor (soil geometry, sea-ice switch, drag statistics), the
land-surface census, and the soil width of the public diagnostic snapshot.
The transaction, rollback, restart and phase-two paths are the frozen ones.

THE DRAG.  The drag runs INSIDE the engine's PBL slot (WRF gwd_opt=3, the GSL
suite), so this row takes no hex-side YSU-GWDO static and refuses one: two
drag operators on one momentum tendency would double count.  The frozen
execution provenance object names its phase-one composition "gwd off" from
the hex side's point of view; the receipt adds ``engine_drag`` saying the
engine slot carried the GSL drag.

THE ENGINE PIN.  The row is unanchored (``anchored=False`` in its table row)
and records the bytes it ran instead of comparing them with a frozen
constant: :func:`pin_column_batch` hashes the suite module, the frozen batch
it subclasses, and the physics files its selected slots execute, and the
forecast's git verifier gates exactly those files clean for the run.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from functools import lru_cache
from hashlib import sha256
from importlib import resources
import inspect
import json
from pathlib import Path
import sys
from types import MappingProxyType
from typing import Any

import numpy as np

from .cuda_arwen_physics_v841 import (
    CudaArwenDiagnosticSnapshotV841,
    PersistentTwoPhaseCudaPhysicsBackendV841,
    _CONSTRUCTOR_KEYS,
    _ISICE_TABLE,
    _OPTIONAL_ARWEN_EXPORT_FIELDS,
    _REQUIRED_ARWEN_EXPORT_FIELDS,
    _SOIL_DIAGNOSTIC_FIELDS,
    _SURFACE_FLOAT_FIELDS,
    _SURFACE_INT_FIELDS,
    _array_identity,
    _cadence_steps,
    _host_exact_array,
    _json_digest,
    _positive_real,
    _snapshot_nbytes,
    history_cumulus_rain_rate,
)
from .diagnostic_export import diagnostic_export
from .cuda_backend.containers import require_resident_array
from .cuda_backend.runtime import KernelCache
from .cuda_physics_prep_v841 import CudaMpasToPhysGeometryV841
from .errors import ConfigurationRefusal
from .species_row import registered_species_rows

__all__ = [
    "CUDA_SUITE_PHYSICS_V841_SCHEMA",
    "PersistentSuiteCudaPhysicsBackendV841",
    "SealedSuiteConstructorV841",
    "build_column_backend",
    "constructor_values_for",
    "load_physics_suites",
    "physics_suite",
    "physics_suite_record",
    "pin_problem",
    "pin_column_batch",
    "read_orographic_statistics",
    "register_profile",
    "required_precipitation_keys",
    "require_init_soil_geometry",
    "scheme_alias_for",
    "seal_constructor",
    "seam_options_for",
    "soil_levels_for",
    "suite_config",
]

CUDA_SUITE_PHYSICS_V841_SCHEMA = "mpas-port.cuda-suite-physics-v841/v1"
SUITES_SCHEMA = "gpuwm-hex.physics-suites/v1"
SUITE_MODULE = "woof.core.mpas_column_suite"
SUITE_ENTRYPOINT = "run_mpas_column_suite"
#: The engine's generic entry: one mapping of RunConfig assignments instead
#: of five slot names.  A profile carrying ``physics`` (resolved from WOOF's
#: registry by woof.hex.physics_suites) is built through it.
GENERIC_ENTRYPOINT = "run_mpas_column_physics"
GENERIC_PLAN = "column_physics_plan"
_SELECTOR_KEYS = (
    "radiation_scheme",
    "surface_layer_scheme",
    "land_surface_scheme",
    "pbl_scheme",
    "drag_scheme",
)

#: Engine files whose bytes the suite row executes, by selected slot.  The
#: pin records their digests; the forecast's git verifier keeps them clean.
_ALWAYS_PINNED = (
    "woof/core/held_suarez_mesh.py",
    "woof/core/kernels/held_suarez_mesh.cu",
    "woof/core/mpas_column_suite.py",
    "woof/core/mpas_column_batch.py",
    "woof/core/physics.py",
    "woof/core/microphysics.py",
    "woof/core/rrtmg_legacy.py",
    "woof/config.py",
    "woof/io/restart.py",
)
_SLOT_PINNED = {
    ("microphysics", "tempo"): (
        "woof/core/tempo.py", "woof/core/tempo_runtime.py",
        "woof/core/tempo_rust.py", "woof/tempo_assets.py",
        "woof/core/kernels/tempo/tempo_common.cuh",
        "woof/core/kernels/tempo/tempo_column.cuh",
        "woof/core/kernels/tempo/tempo_processes.cuh",
        "woof/core/kernels/tempo/tempo_diags.cuh",
        "woof/core/kernels/tempo/tempo_cloud_ml.cuh",
        "woof/core/kernels/glibc_flt64.cuh",
        ),
    ("surface_layer_scheme", "mynn"): (
        "woof/core/mynn_sfclay.py", "woof/core/mynn_surface.py"),
    ("pbl_scheme", "mynn"): (
        "woof/core/mynn_pbl.py", "woof/core/mynn_pbl_gpu.py",
        "woof/core/mynn_pbl_runtime.py", "woof/core/mynn_pbl_scratch.py"),
    ("land_surface_scheme", "ruc"): (
        "woof/core/ruc.py", "woof/core/ruc_gpu.py",
        "woof/core/ruc_runtime.py", "woof/core/ruc_contract.py"),
    ("land_surface_scheme", "noahmp"): ("woof/core/noahmp_runtime.py",),
    ("drag_scheme", "gsl"): (
        "woof/core/terrain_drag.py", "woof/core/kernels/terrain_drag.cu",
        "woof/static/orographic.py"),
}


# --------------------------------------------------------------------------
# the profile table
# --------------------------------------------------------------------------
@lru_cache(maxsize=1)
def load_physics_suites() -> Mapping[str, Mapping[str, Any]]:
    """The profile table, validated, keyed by profile id."""

    text = (
        resources.files("woof.hex")
        .joinpath("data/profiles/physics-suites.v1.json")
        .read_text(encoding="utf-8")
    )
    table = json.loads(text)
    if table.get("schema") != SUITES_SCHEMA:
        raise ValueError(
            f"physics-suites table schema {table.get('schema')!r} is not "
            f"{SUITES_SCHEMA!r}"
        )
    rows: dict[str, Mapping[str, Any]] = {}
    for row in table["suites"]:
        ident = str(row["id"])
        if ident in rows:
            raise ValueError(f"physics suite {ident!r} is declared twice")
        selection = dict(row["selection"])
        if tuple(sorted(selection)) != tuple(sorted(_SELECTOR_KEYS)):
            raise ValueError(
                f"physics suite {ident!r} must name every selector "
                f"{sorted(_SELECTOR_KEYS)}, got {sorted(selection)}"
            )
        soil = int(row["n_soil_levels"])
        depths = tuple(float(value) for value in row["soil_level_depths_m"])
        if len(depths) != soil:
            raise ValueError(
                f"physics suite {ident!r} declares {soil} soil levels and "
                f"{len(depths)} depths"
            )
        if row["fractional_seaice"] not in (0, 1, None):
            raise ValueError(f"physics suite {ident!r}: fractional_seaice")
        rows[ident] = MappingProxyType(dict(row))
    return MappingProxyType(rows)


#: Profiles synthesized by :mod:`woof.hex.physics_suites` for a selection no
#: table row names (a per-scheme override).  Keyed by profile id; a profile
#: is registered once and never replaced, exactly as backend rows are.
_SYNTHESIZED: dict[str, Mapping[str, Any]] = {}


def required_precipitation_keys(*, generic: bool, microphysics: str) -> frozenset[str]:
    """The accumulators a suite seam must publish at a committed boundary.

    The slot-table seams publish RAINC, RAINNC and SNOWNC.  The generic seam
    publishes RAINC plus exactly the buckets the species row declares, and
    that row was derived from the engine's plan (plan.surface_accumulators).
    THE BREAKAGE THE ROW-SIZED SET PREVENTS: the literal {"RAINC", "RAINNC"}
    refused microphysics off, whose plan has no grid-scale bucket at all, at
    rank construction ("suite precipitation inventory changed: ['RAINC']",
    60 km global, 2026-10-11).  A bucket the row declares and the seam does
    not publish is still refused by the caller.
    """

    if not generic:
        return frozenset({"RAINC", "RAINNC", "SNOWNC"})
    from .species_row import species_row_for_scheme

    row = species_row_for_scheme(microphysics)
    return frozenset({"RAINC"} | {item.history_name for item in row.surface_accumulators})


def register_profile(profile: Mapping[str, Any]) -> Mapping[str, Any]:
    """Add one synthesized profile.  An id already present with other
    content is refused: a replaceable profile would change what a backend
    row runs by import order."""

    ident = str(profile["id"])
    frozen = MappingProxyType(dict(profile))
    if ident in load_physics_suites():
        raise ValueError(f"profile {ident!r} is a table row and is not synthesized")
    existing = _SYNTHESIZED.get(ident)
    if existing is not None:
        if dict(existing) != dict(frozen):
            raise ValueError(f"synthesized profile {ident!r} is already registered differently")
        return existing
    _SYNTHESIZED[ident] = frozen
    return frozen


def _all_profiles() -> Mapping[str, Mapping[str, Any]]:
    return {**load_physics_suites(), **_SYNTHESIZED}


def physics_suite(profile_id: str) -> Mapping[str, Any]:
    """One profile row, or a refusal naming the ones that exist."""

    rows = _all_profiles()
    if profile_id not in rows:
        raise ConfigurationRefusal(
            "physics_suite",
            profile_id,
            "no physics-suite profile carries this id",
            f"one of {sorted(rows)}",
        )
    return rows[profile_id]


def _suite_for_row(row_name: str) -> Mapping[str, Any]:
    for suite in _all_profiles().values():
        if suite["backend_row"] == row_name:
            return suite
    raise ConfigurationRefusal(
        "physics_backend",
        row_name,
        "no physics-suite profile names this backend row",
        "a row listed in data/profiles/physics-suites.v1.json",
    )


def soil_levels_for(row_name: str) -> int:
    """The soil column the row's land surface integrates."""

    return int(_suite_for_row(row_name)["n_soil_levels"])


# --------------------------------------------------------------------------
# the provider-row protocol the forecast driver reads
# --------------------------------------------------------------------------
def scheme_alias_for(row_name: str) -> str:
    """The ``config_microp_scheme`` the row's species block is."""

    return str(_suite_for_row(row_name)["microphysics"])


def seam_options_for(*, source_table: str | None = None) -> dict[str, Any]:
    """The suite row takes no point-source table; anything else refuses."""

    if source_table is not None:
        raise ConfigurationRefusal(
            "source_table",
            str(source_table),
            "the column-suite row carries no point source; the table would "
            "be accepted and never read",
            "no --source-table",
        )
    return {}


def read_orographic_statistics(
    path: str | Path, *, row_name: str, n_cells: int
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """The row's drag statistics from the mesh's orographic file.

    Decoded by the engine's Rust NetCDF bridge (``woof.netcdf_bridge``,
    rw_netcdf on netcrust): no Python NetCDF library reads these bytes.
    Each variable must be ``float32 [nCells]`` and finite; the engine's
    upper-case field names are the keys.
    """

    from woof.netcdf_bridge import open_dataset

    suite = _suite_for_row(row_name)
    spec = suite["orographic_statistics"]
    source = Path(path).expanduser().resolve(strict=True)
    if spec is None:
        raise ConfigurationRefusal(
            "orographic_statistics", str(source),
            f"the {suite['id']} profile reads no orographic statistics; the "
            "file would be accepted and never read",
            "no --orographic-statistics",
        )
    if not source.name.endswith(str(spec["file_suffix"])):
        raise ConfigurationRefusal(
            "orographic_statistics",
            str(source),
            f"the {suite['id']} profile reads a mesh's "
            f"'*{spec['file_suffix']}' statistics file",
            f"the mesh's {spec['file_suffix']} file",
        )
    dataset = open_dataset(source)
    out: dict[str, np.ndarray] = {}
    hashes: dict[str, Any] = {}
    for engine_name, file_name in spec["variables"].items():
        if file_name not in dataset.variables:
            raise ConfigurationRefusal(
                "orographic_statistics",
                str(source),
                f"{source.name} carries no {file_name!r}, which the "
                f"{engine_name} drag statistic is read from",
                f"a statistics file with {sorted(spec['variables'].values())}",
            )
        variable = dataset.variables[file_name]
        if tuple(variable.dimensions) != (str(spec["dimension"]),):
            raise ValueError(
                f"{file_name} dimensions {variable.dimensions} != "
                f"({spec['dimension']!r},)"
            )
        values = np.ascontiguousarray(np.asarray(variable[:]), dtype=np.float32)
        if values.shape != (int(n_cells),):
            raise ValueError(
                f"{file_name} has {values.shape} cells, the mesh {n_cells}"
            )
        if not np.all(np.isfinite(values)):
            raise FloatingPointError(f"{file_name} carries non-finite values")
        out[engine_name] = values
        hashes[engine_name] = {"variable": file_name, **_array_identity(values)}
    receipt = {
        "path": str(source),
        "file_sha256": sha256(source.read_bytes()).hexdigest(),
        "reader": "woof.netcdf_bridge (rw_netcdf)",
        "variables": hashes,
    }
    return out, receipt


def require_init_soil_geometry(init_path: str | Path, *, row_name: str) -> dict[str, Any]:
    """Refuse an init whose soil levels are not the land surface's own.

    THE BREAKAGE THIS PREVENTS: a land surface handed soil temperature and
    moisture interpolated to another scheme's depths integrates them as if
    they sat at its own -- a nine-value column built for other depths runs
    without any error and with the wrong soil.  The init's own ``zs`` (the
    depths its interpolation targeted) must equal the profile's level
    depths in every column, at float32.  Decoded through the engine's Rust
    NetCDF bridge.
    """

    from woof.netcdf_bridge import open_dataset

    suite = _suite_for_row(row_name)
    if suite["soil_level_depths_m"] is None:
        return {"zs_m": None, "columns_checked": 0,
                "note": "this selection's land surface declares no soil depths "
                        "to hold the init to"}
    want = np.asarray(suite["soil_level_depths_m"], dtype=np.float32)
    dataset = open_dataset(Path(init_path))
    if "zs" not in dataset.variables:
        raise ConfigurationRefusal(
            "init", str(init_path),
            "the init carries no zs, so nothing says which soil depths its "
            f"{want.size} levels were interpolated to",
            "an init_atmosphere output written with the land surface's own "
            "soil levels",
        )
    zs = np.asarray(dataset.variables["zs"][:], dtype=np.float32).reshape(-1, want.size)
    if not np.all(zs == want[None, :]):
        bad = int(np.count_nonzero(np.any(zs != want[None, :], axis=1)))
        raise ConfigurationRefusal(
            "init", str(init_path),
            f"{bad} columns carry soil depths other than the {suite['id']} "
            f"profile's {want.tolist()} m",
            "an init interpolated to the land surface's own soil depths",
        )
    return {"zs_m": want.tolist(), "columns_checked": int(zs.shape[0]),
            "reader": "woof.netcdf_bridge (rw_netcdf)"}


def constructor_values_for(
    values: Mapping[str, Any],
    *,
    row_name: str,
    orographic_statistics: Mapping[str, np.ndarray] | None,
    cell_area_m2=None,
) -> dict[str, Any]:
    """The frozen door's mapping, re-shaped to the suite row's seam.

    The door builds the same exact-real mapping for every row.  The suite
    row replaces the one knob that is Noah-MP's alone (``xice_threshold``)
    with the land surface's own sea-ice switch, adds the profile's
    selectors and the drag statistics, and keeps every other value
    verbatim -- soil included, which the door already read at this row's
    level count.
    """

    suite = _suite_for_row(row_name)
    mapped = {key: value for key, value in values.items() if key != "xice_threshold"}
    selection = dict(suite["selection"])
    mapped.update(selection)
    if selection["land_surface_scheme"] == "noahmp":
        mapped["xice_threshold"] = values["xice_threshold"]
        mapped["fractional_seaice"] = None
    elif suite.get("physics") is not None:
        # The sea-ice switch of a registry-resolved selection is one of its
        # RunConfig assignments (or the engine's default when it names none).
        mapped["fractional_seaice"] = suite["fractional_seaice"]
    else:
        mapped["fractional_seaice"] = int(suite["fractional_seaice"])
    mapped["orographic_statistics"] = (
        None if orographic_statistics is None else dict(orographic_statistics)
    )
    if suite.get("physics") is not None:
        # A registry-resolved selection: the engine's generic entry takes the
        # RunConfig assignments themselves (generation selectors included),
        # and every scheme may weigh by cell area.
        mapped["physics"] = dict(suite["physics"])
        if cell_area_m2 is not None:
            mapped["cell_area_m2"] = np.asarray(cell_area_m2, dtype=np.float64)
        return mapped
    generation = dict(suite.get("generation", {}) or {})
    if generation:
        # Only a profile that selects a generation hands one to the seam, so
        # a generation-less profile builds the seam exactly as before.
        mapped["generation"] = generation
    if cell_area_m2 is not None and values["microphysics_scheme"] == "tempo":
        mapped["cell_area_m2"] = np.asarray(cell_area_m2, dtype=np.float64)
    return mapped


# --------------------------------------------------------------------------
# the sealed constructor
# --------------------------------------------------------------------------
_SUITE_CONSTRUCTOR_KEYS = frozenset(
    (_CONSTRUCTOR_KEYS - {"xice_threshold"})
    | {"fractional_seaice", "orographic_statistics", *_SELECTOR_KEYS}
)
_CONSTRUCTOR_SEAL = object()
#: The soil column's two arrays, which the generic entry takes exactly when
#: the engine plan integrates soil (a land surface is selected).
_GENERIC_SOIL_KEYS = frozenset({"soil_temperature", "soil_moisture"})


@dataclass(frozen=True, slots=True)
class SealedSuiteConstructorV841:
    """Complete, immutable constructor mapping for the selectable suite."""

    _values: Mapping[str, Any] = field(repr=False, compare=False)
    identity_sha256: str
    host_array_bytes: int
    n_soil_levels: int
    _seal: object = field(repr=False, compare=False)

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "SealedSuiteConstructorV841":
        if not isinstance(values, Mapping):
            raise TypeError("suite constructor values must be a mapping")
        from . import correct_mode

        keys = set(values) - correct_mode.OPTIONAL_CONSTRUCTOR_KEYS
        if "xice_threshold" in keys and values.get("land_surface_scheme") == "noahmp":
            keys.discard("xice_threshold")
        generation = values.get("generation")
        keys.discard("generation")
        keys.discard("cell_area_m2")
        if generation is not None and not isinstance(generation, Mapping):
            raise TypeError("generation must be a mapping of generation selectors")
        if values.get("physics") is not None:
            keys.discard("physics")
            keys.discard("xice_threshold")
            # The soil column is the engine plan's to require: present
            # exactly when the selection integrates soil (held to that in
            # _from_generic_mapping), absent under sf_surface_physics=0.
            missing = sorted(_SUITE_CONSTRUCTOR_KEYS - _GENERIC_SOIL_KEYS - keys)
            extra = sorted(keys - _SUITE_CONSTRUCTOR_KEYS)
            if missing or extra:
                raise ValueError(
                    "suite constructor mapping is not exhaustive: "
                    f"missing={missing}, extra={extra}"
                )
            return cls._from_generic_mapping(values)
        missing = sorted(_SUITE_CONSTRUCTOR_KEYS - keys)
        extra = sorted(keys - _SUITE_CONSTRUCTOR_KEYS)
        if missing or extra:
            raise ValueError(
                "suite constructor mapping is not exhaustive: "
                f"missing={missing}, extra={extra}"
            )
        from woof.core.mpas_column_suite import (
            LAND_SURFACE_SCHEMES,
            orographic_statistic_names,
            resolve_suite_selection,
        )

        selection = {key: values[key] for key in _SELECTOR_KEYS}
        resolve_suite_selection(**selection)
        if generation:
            from woof.core.mpas_column_suite import resolve_generation

            resolve_generation(
                generation,
                microphysics_scheme=values["microphysics_scheme"],
                selection=selection,
            )
        nlev, ncol = int(values["n_levels"]), int(values["n_columns"])
        if nlev < 3 or ncol < 1:
            raise ValueError("n_levels must be >= 3 and n_columns >= 1")
        if not isinstance(values["start_time"], datetime):
            raise TypeError("start_time must be a UTC datetime")
        scheme = values["cumulus_scheme"]
        if scheme not in ("gf", "kf", "ntiedtke", None):
            raise ValueError("cumulus_scheme must be 'gf', 'kf', 'ntiedtke', or None")
        dt = _positive_real("dt", values["dt"])
        radiation_seconds = _positive_real("radiation_seconds", values["radiation_seconds"])
        surface_pbl_seconds = _positive_real("surface_pbl_seconds", values["surface_pbl_seconds"])
        _cadence_steps("radiation_seconds", radiation_seconds, dt)
        _cadence_steps("surface_pbl_seconds", surface_pbl_seconds, dt)
        cumulus_seconds = None
        if values["cumulus_seconds"] is not None:
            cumulus_seconds = _positive_real("cumulus_seconds", values["cumulus_seconds"])
            if scheme in ("gf", "ntiedtke") and _cadence_steps("cumulus_seconds", cumulus_seconds, dt) != 1:
                raise ValueError(f"cumulus_scheme={scheme!r} requires cumulus_seconds == dt")
        microphysics = values["microphysics_scheme"]
        engine_schemes = tuple(
            row.engine_scheme
            for row in registered_species_rows().values()
            if row.engine_scheme is not None
        )
        if microphysics not in engine_schemes:
            raise ValueError(
                f"microphysics_scheme must be one of {sorted(engine_schemes)}, "
                f"got {microphysics!r}"
            )
        if int(values["wsm6_hail_opt"]) not in (0, 1) or (
            microphysics != "wsm6" and int(values["wsm6_hail_opt"]) != 0
        ):
            raise ValueError("wsm6_hail_opt must be 0 off WSM6, 0 or 1 on it")
        ishallow = int(values["gf_ishallow"])
        if ishallow not in (0, 1) or (ishallow and scheme != "gf"):
            raise ValueError("gf_ishallow must be 0, or 1 with cumulus_scheme='gf'")
        land = selection["land_surface_scheme"]
        n_soil = int(LAND_SURFACE_SCHEMES[land]["num_soil_layers"])
        if land == "noahmp":
            threshold = values.get("xice_threshold")
            if values["fractional_seaice"] is not None or threshold is None:
                raise ValueError("Noah-MP takes xice_threshold and no fractional_seaice")
            seaice = {"xice_threshold": float(threshold)}
        else:
            if values["fractional_seaice"] not in (0, 1):
                raise ValueError(f"{land} needs fractional_seaice 0 or 1")
            seaice = {"fractional_seaice": int(values["fractional_seaice"])}
        scalars = {
            "n_levels": nlev,
            "n_columns": ncol,
            "dt": dt,
            "radiation_seconds": radiation_seconds,
            "surface_pbl_seconds": surface_pbl_seconds,
            "cumulus_seconds": cumulus_seconds,
            "cumulus_scheme": scheme,
            "microphysics_scheme": microphysics,
            "start_time": values["start_time"],
            "p_top_pa": _positive_real("p_top_pa", values["p_top_pa"]),
            "dx_m": _positive_real("dx_m", values["dx_m"]),
            "gf_ishallow": ishallow,
            "wsm6_hail_opt": int(values["wsm6_hail_opt"]),
            **selection,
            **seaice,
            # physics.mode: present only in correct mode (woof.hex.correct_mode).
            **correct_mode.sealed_scalars(values),
        }
        if generation:
            scalars["generation"] = MappingProxyType(dict(generation))
        arrays: dict[str, np.ndarray] = {}
        for name in _SURFACE_FLOAT_FIELDS:
            arrays[name] = _host_exact_array(values, name, dtype=np.dtype(np.float32), shape=(ncol,))
        for name in _SURFACE_INT_FIELDS:
            arrays[name] = _host_exact_array(values, name, dtype=np.dtype(np.int32), shape=(ncol,))
        for name in ("soil_temperature", "soil_moisture"):
            arrays[name] = _host_exact_array(
                values, name, dtype=np.dtype(np.float32), shape=(n_soil, ncol)
            )
        dx_column = _host_exact_array(values, "dx_column_m", dtype=np.dtype(np.float32), shape=(ncol,))
        if np.any(dx_column <= np.float32(0.0)):
            raise ValueError("dx_column_m must be positive")
        arrays["dx_column_m"] = dx_column
        if values.get("cell_area_m2") is not None:
            area = _host_exact_array(values, "cell_area_m2", dtype=np.dtype(np.float64), shape=(ncol,))
            if np.any(area <= 0):
                raise ValueError("cell_area_m2 must be positive")
            arrays["cell_area_m2"] = area
        z = np.asarray(values["z_interface_nominal_m"])
        if z.shape != (nlev + 1,) or not np.all(np.isfinite(z)) or np.any(np.diff(z) <= 0):
            raise ValueError("z_interface_nominal_m must be finite, increasing, nlev+1 long")
        arrays["z_interface_nominal_m"] = np.array(z, copy=True, order="C")
        arrays["z_interface_nominal_m"].setflags(write=False)
        needed = orographic_statistic_names(selection["drag_scheme"])
        given = values["orographic_statistics"]
        if (not needed) != (given is None):
            raise ValueError(
                f"drag_scheme={selection['drag_scheme']!r} needs "
                f"{list(needed)}; orographic_statistics must be given exactly "
                "when the drag reads them"
            )
        oro: dict[str, np.ndarray] = {}
        if given is not None:
            if sorted(given) != sorted(needed):
                raise ValueError(
                    f"orographic statistics {sorted(given)} != {sorted(needed)}"
                )
            for name in needed:
                oro[name] = _host_exact_array(
                    given, name, dtype=np.dtype(np.float32), shape=(ncol,)
                )
        sealed: dict[str, Any] = {
            **scalars,
            **arrays,
            "orographic_statistics": MappingProxyType(oro) if oro else None,
        }
        identity = {
            name: (
                _array_identity(value)
                if isinstance(value, np.ndarray)
                else value.isoformat()
                if isinstance(value, datetime)
                else dict(value)
                if name == "generation"
                else {key: _array_identity(item) for key, item in value.items()}
                if isinstance(value, Mapping)
                else value
            )
            for name, value in sorted(sealed.items())
        }
        host_bytes = sum(int(value.nbytes) for value in arrays.values())
        host_bytes += sum(int(value.nbytes) for value in oro.values())
        return cls(
            _values=MappingProxyType(sealed),
            identity_sha256=_json_digest(identity),
            host_array_bytes=host_bytes,
            n_soil_levels=n_soil,
            _seal=_CONSTRUCTOR_SEAL,
        )

    @classmethod
    def _from_generic_mapping(cls, values: Mapping[str, Any]) -> "SealedSuiteConstructorV841":
        """Seal a registry-resolved selection for the engine's generic entry.

        No scheme list is consulted here.  The physics mapping gains the
        cumulus the mesh bind decided (``cu_physics``) and the two scheme
        flags the door carries (``ishallow``, ``wsm6_hail_opt``), and the
        ENGINE plans it with the run's own level count and cadences
        (``column_physics_plan``): WOOF's validators judge the combination,
        and the plan names the soil column and the statics the arrays are
        held to.
        """

        from . import convection_admission
        import woof.core.mpas_column_suite as suite_module

        nlev, ncol = int(values["n_levels"]), int(values["n_columns"])
        if nlev < 3 or ncol < 1:
            raise ValueError("n_levels must be >= 3 and n_columns >= 1")
        if not isinstance(values["start_time"], datetime):
            raise TypeError("start_time must be a UTC datetime")
        dt = _positive_real("dt", values["dt"])
        radiation_seconds = _positive_real("radiation_seconds", values["radiation_seconds"])
        surface_pbl_seconds = _positive_real("surface_pbl_seconds", values["surface_pbl_seconds"])
        scheme = values["cumulus_scheme"]
        physics = dict(values["physics"])
        physics["cu_physics"] = convection_admission.cu_physics_for_constructor_scheme(scheme)
        if int(values["gf_ishallow"]):
            physics.setdefault("ishallow", int(values["gf_ishallow"]))
        # The seal records the shallow switch the ENGINE is handed, not the
        # name-derived default: a selection that sets ishallow=0 ran with the
        # shallow arm off, and a sealed gf_ishallow=1 beside it said it ran on
        # (measured: global 30 km, cu_grell_freitas with ishallow=0, receipt
        # gf_ishallow 1 against constructor physics ishallow 0).
        shallow = int(physics.get("ishallow", 0))
        if int(values["wsm6_hail_opt"]):
            physics.setdefault("wsm6_hail_opt", int(values["wsm6_hail_opt"]))
        # physics.mode on the generic entry: the engine's correct-mode keys
        # are RunConfig assignments like every other selector here.  Exact
        # mode adds nothing, so the mapping is the one sealed before the key.
        from . import correct_mode

        correct = correct_mode.sealed_scalars(values)
        if correct:
            # A gap (engine without correct mode) leaves the settings empty;
            # the backend's receipt names it (suite_correct_mode_receipt).
            settings = correct_mode.engine_settings(
                correct["physics_mode"], correct["physics_correct_skip"],
                report=False)["settings"]
            for key, value in settings.items():
                if key in physics and physics[key] != value:
                    raise ValueError(
                        f"physics selects {key}={physics[key]!r} and physics.mode "
                        f"= 'correct' sets it to {value!r}; name one")
                physics[key] = value
        # The mesh bind hands every scheme the model step as its cadence;
        # the engine's spelling of "every step" is no hold at all.
        cumulus_seconds = None
        if values["cumulus_seconds"] is not None:
            cumulus_seconds = _positive_real("cumulus_seconds", values["cumulus_seconds"])
            if cumulus_seconds == dt:
                cumulus_seconds = None
        plan = getattr(suite_module, GENERIC_PLAN)(
            physics, n_levels=nlev, dt=dt, radiation_seconds=radiation_seconds,
            surface_pbl_seconds=surface_pbl_seconds, cumulus_seconds=cumulus_seconds)
        n_soil = int(plan.n_soil_levels)
        # THE BREAKAGE THIS PREVENTS, both ways: a land surface sealed without
        # its soil column, and a soil column sealed for a selection with no
        # land surface (accepted and never read).  Land surface off
        # (sf_surface_physics=0) plans zero soil levels and the forecast
        # driver reads no tslb/smois for it.
        present = sorted(_GENERIC_SOIL_KEYS & set(values))
        if n_soil > 0 and len(present) != len(_GENERIC_SOIL_KEYS):
            raise ValueError(
                f"the selection integrates {n_soil} soil levels and the mapping "
                f"carries only {present} of {sorted(_GENERIC_SOIL_KEYS)}")
        if n_soil == 0 and present:
            raise ValueError(
                f"the selection integrates no soil column and the mapping carries "
                f"{present}, which no scheme would read")
        selection = {key: str(values[key]) for key in _SELECTOR_KEYS}
        land = selection["land_surface_scheme"]
        seaice: dict[str, Any] = {}
        if land == "noahmp":
            if values.get("xice_threshold") is None:
                raise ValueError("Noah-MP takes xice_threshold")
            seaice = {"xice_threshold": float(values["xice_threshold"])}
        scalars = {
            "n_levels": nlev,
            "n_columns": ncol,
            "dt": dt,
            "radiation_seconds": radiation_seconds,
            "surface_pbl_seconds": surface_pbl_seconds,
            "cumulus_seconds": cumulus_seconds,
            "cumulus_scheme": scheme,
            "microphysics_scheme": values["microphysics_scheme"],
            "start_time": values["start_time"],
            "p_top_pa": _positive_real("p_top_pa", values["p_top_pa"]),
            "dx_m": _positive_real("dx_m", values["dx_m"]),
            "gf_ishallow": shallow,
            "wsm6_hail_opt": int(values["wsm6_hail_opt"]),
            "fractional_seaice": physics.get("fractional_seaice"),
            "physics": MappingProxyType(dict(sorted(physics.items()))),
            **selection,
            **seaice,
            # physics.mode: present only in correct mode, as on the five-slot
            # entry, so the sealed mode never has to be inferred from the
            # physics keys (a skip of every finding injects none).
            **correct,
        }
        arrays: dict[str, np.ndarray] = {}
        for name in _SURFACE_FLOAT_FIELDS:
            arrays[name] = _host_exact_array(values, name, dtype=np.dtype(np.float32), shape=(ncol,))
        for name in _SURFACE_INT_FIELDS:
            arrays[name] = _host_exact_array(values, name, dtype=np.dtype(np.int32), shape=(ncol,))
        if n_soil > 0:
            for name in ("soil_temperature", "soil_moisture"):
                arrays[name] = _host_exact_array(
                    values, name, dtype=np.dtype(np.float32), shape=(n_soil, ncol)
                )
        dx_column = _host_exact_array(values, "dx_column_m", dtype=np.dtype(np.float32), shape=(ncol,))
        if np.any(dx_column <= np.float32(0.0)):
            raise ValueError("dx_column_m must be positive")
        arrays["dx_column_m"] = dx_column
        if values.get("cell_area_m2") is not None:
            area = _host_exact_array(values, "cell_area_m2", dtype=np.dtype(np.float64), shape=(ncol,))
            if np.any(area <= 0):
                raise ValueError("cell_area_m2 must be positive")
            arrays["cell_area_m2"] = area
        z = np.asarray(values["z_interface_nominal_m"])
        if z.shape != (nlev + 1,) or not np.all(np.isfinite(z)) or np.any(np.diff(z) <= 0):
            raise ValueError("z_interface_nominal_m must be finite, increasing, nlev+1 long")
        arrays["z_interface_nominal_m"] = np.array(z, copy=True, order="C")
        arrays["z_interface_nominal_m"].setflags(write=False)
        needed = tuple(str(name) for name in plan.static_fields)
        given = values["orographic_statistics"]
        if (not needed) != (given is None):
            raise ValueError(
                f"this selection reads the statics {list(needed)}; "
                "orographic_statistics must be given exactly when it reads them"
            )
        oro: dict[str, np.ndarray] = {}
        if given is not None:
            if sorted(given) != sorted(needed):
                raise ValueError(
                    f"orographic statistics {sorted(given)} != {sorted(needed)}"
                )
            for name in needed:
                oro[name] = _host_exact_array(
                    given, name, dtype=np.dtype(np.float32), shape=(ncol,)
                )
        sealed: dict[str, Any] = {
            **scalars,
            **arrays,
            "orographic_statistics": MappingProxyType(oro) if oro else None,
            "plan_species": tuple(str(name) for name in plan.species),
            "plan_identity": json.loads(json.dumps(getattr(plan, "identity", {}), default=str)),
        }
        def identity_of(value: Any) -> Any:
            if isinstance(value, np.ndarray):
                return _array_identity(value)
            if isinstance(value, datetime):
                return value.isoformat()
            if isinstance(value, Mapping):
                return {str(key): identity_of(item) for key, item in value.items()}
            if isinstance(value, (tuple, list)):
                return [identity_of(item) for item in value]
            return value

        identity = {name: identity_of(value) for name, value in sorted(sealed.items())}
        host_bytes = sum(int(value.nbytes) for value in arrays.values())
        host_bytes += sum(int(value.nbytes) for value in oro.values())
        return cls(
            _values=MappingProxyType(sealed),
            identity_sha256=_json_digest(identity),
            host_array_bytes=host_bytes,
            n_soil_levels=n_soil,
            _seal=_CONSTRUCTOR_SEAL,
        )

    @property
    def physics(self) -> dict[str, Any] | None:
        """The RunConfig assignments of a registry-resolved selection, or
        ``None`` for a slot-named profile."""

        value = self._values.get("physics")
        return None if value is None else dict(value)

    @property
    def n_levels(self) -> int:
        return int(self._values["n_levels"])

    @property
    def n_columns(self) -> int:
        return int(self._values["n_columns"])

    @property
    def dt(self) -> float:
        return float(self._values["dt"])

    @property
    def microphysics_scheme(self) -> str:
        return str(self._values["microphysics_scheme"])

    @property
    def cumulus_scheme(self) -> str | None:
        """The cumulus scheme, as the frozen constructor names it.

        The frozen backend's receipt and seam choice read it (the New
        Tiedtke row branches on it); without it the first receipt of a
        column-suite backend stopped with AttributeError (measured
        2026-10-07, conus12km).  The suite seam admits gf and None.
        """

        scheme = self._values["cumulus_scheme"]
        return None if scheme is None else str(scheme)

    @property
    def selection(self) -> dict[str, str]:
        return {key: str(self._values[key]) for key in _SELECTOR_KEYS}

    @property
    def physics_mode(self) -> str:
        """``physics.mode`` as sealed (woof.hex.correct_mode), on either entry:
        ``"exact"`` unless correct was asked."""
        return str(self._values.get("physics_mode", "exact"))

    @property
    def physics_correct_skip(self) -> tuple[str, ...]:
        return tuple(self._values.get("physics_correct_skip", ()))

    @property
    def xice_threshold(self) -> float | None:
        value = self._values.get("xice_threshold")
        return None if value is None else float(value)

    def expected_surface_classification(self) -> Mapping[str, Any]:
        """The engine's classification receipt for this land surface.

        Every land surface publishes the XLAND source and split; Noah-MP
        additionally publishes its own sea-ice / glacier classes.
        """

        xland = np.asarray(self._values["xland"], dtype=np.float32)
        result: dict[str, Any] = {
            "xland_source": "native",
            "xland_land_columns": int(np.count_nonzero(xland < np.float32(1.5))),
            "xland_water_columns": int(np.count_nonzero(xland >= np.float32(1.5))),
        }
        if self.selection["land_surface_scheme"] == "noahmp" and self.xice_threshold is not None:
            xice = np.asarray(self._values["xice"], dtype=np.float32)
            ivgtyp = np.asarray(self._values["ivgtyp"], dtype=np.int32)
            threshold = np.float32(self.xice_threshold)
            sea_ice = xice >= threshold
            open_water = (xland >= np.float32(1.5)) & ~sea_ice
            land = ~(sea_ice | open_water)
            glacier = land & (ivgtyp == np.int32(_ISICE_TABLE))
            result.update({
                "xice_threshold": float(threshold),
                "sea_ice_columns": int(np.count_nonzero(sea_ice)),
                "open_water_columns": int(np.count_nonzero(open_water)),
                "sflx_land_columns": int(np.count_nonzero(land & ~glacier)),
                "glacier_columns": int(np.count_nonzero(glacier)),
            })
        return MappingProxyType(result)

    def arwen_kwargs(self) -> dict[str, Any]:
        """The suite seam's keyword set (``run_mpas_column_suite``)."""

        if self._seal is not _CONSTRUCTOR_SEAL:
            raise TypeError("suite constructor mapping is not sealed")
        if self._values.get("physics") is not None:
            return self._generic_kwargs()
        kwargs = dict(self._values)
        oro = kwargs.pop("orographic_statistics")
        kwargs["orographic_statistics"] = None if oro is None else dict(oro)
        if "generation" in kwargs:
            kwargs["generation"] = dict(kwargs["generation"])
        return kwargs

    def _generic_kwargs(self) -> dict[str, Any]:
        """The generic entry's keyword set (``run_mpas_column_physics``).

        ``surface`` is the engine's own ``initialize_physics`` keyword set;
        ``xice_threshold`` rides in it only under Noah-MP, whose keyword it
        is.  ``statics`` is exactly the plan's static fields.
        """

        values = self._values
        surface = {
            name: values[name]
            for name in (*_SURFACE_FLOAT_FIELDS, *_SURFACE_INT_FIELDS)
            if name not in ("latitude_deg", "longitude_deg", "terrain_height_m")
        }
        for name in ("soil_temperature", "soil_moisture"):
            if name in values:
                surface[name] = values[name]
        if values.get("xice_threshold") is not None:
            surface["xice_threshold"] = float(values["xice_threshold"])
        oro = values["orographic_statistics"]
        return {
            "physics": dict(values["physics"]),
            "n_levels": values["n_levels"],
            "n_columns": values["n_columns"],
            "dt": values["dt"],
            "radiation_seconds": values["radiation_seconds"],
            "surface_pbl_seconds": values["surface_pbl_seconds"],
            "cumulus_seconds": values["cumulus_seconds"],
            "start_time": values["start_time"],
            "latitude_deg": values["latitude_deg"],
            "longitude_deg": values["longitude_deg"],
            "terrain_height_m": values["terrain_height_m"],
            "z_interface_nominal_m": values["z_interface_nominal_m"],
            "p_top_pa": values["p_top_pa"],
            "dx_m": values["dx_m"],
            "dx_column_m": values["dx_column_m"],
            "cell_area_m2": values.get("cell_area_m2"),
            "surface": surface,
            "statics": {} if oro is None else dict(oro),
        }

    def receipt(self) -> Mapping[str, Any]:
        return MappingProxyType(
            {
                "identity_sha256": self.identity_sha256,
                "physics": self.physics,
                "n_levels": self.n_levels,
                "n_columns": self.n_columns,
                "n_soil_levels": self.n_soil_levels,
                "dt": self.dt,
                "microphysics_scheme": self.microphysics_scheme,
                "selection": self.selection,
                "generation": dict(self._values.get("generation", {})),
                "fractional_seaice": self._values.get("fractional_seaice"),
                "xice_threshold": self.xice_threshold,
                "host_array_bytes": self.host_array_bytes,
                "defaults_used": False,
                "orographic_statistics": (
                    None
                    if self._values["orographic_statistics"] is None
                    else sorted(self._values["orographic_statistics"])
                ),
                "expected_surface_classification": dict(
                    self.expected_surface_classification()
                ),
                # Present only in correct mode (MODE-KEY.md rule 3).
                **(
                    {"physics_mode": self.physics_mode,
                     "physics_correct_skip": list(self.physics_correct_skip)}
                    if self.physics_mode != "exact" else {}
                ),
            }
        )


def seal_constructor(values: Mapping[str, Any]) -> SealedSuiteConstructorV841:
    return SealedSuiteConstructorV841.from_mapping(values)


# --------------------------------------------------------------------------
# the engine the row binds, recorded rather than frozen
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class SuiteEngineIdentity:
    """The measured identity of the engine tree the suite row ran."""

    version: str
    build_commit: str | None
    manifest: Mapping[str, str]
    root: str
    admitted: bool = False
    glacier_composed_tu_sha256: str | None = None


def _pinned_paths(selection: Mapping[str, str]) -> tuple[str, ...]:
    paths = list(_ALWAYS_PINNED)
    for (key, value), files in _SLOT_PINNED.items():
        if selection.get(key) == value:
            paths.extend(files)
    return tuple(dict.fromkeys(paths))


def _engine_root(checkout: str | Path | None) -> Path:
    if checkout is not None:
        requested = Path(checkout).resolve()
        already = sys.modules.get("woof")
        if already is not None:
            loaded = Path(inspect.getfile(already)).resolve().parent.parent
            if loaded != requested:
                raise RuntimeError(
                    "woof was already imported from a different tree than "
                    f"the named engine checkout {requested}"
                )
        elif not any(isinstance(entry, str) and Path(entry or ".").resolve() == requested.resolve()
                     for entry in sys.path):
            sys.path.insert(0, str(requested))
    import woof.core.mpas_column_suite as suite_module

    return Path(inspect.getfile(suite_module)).resolve().parents[2]


def _measure_engine(root: Path, selection: Mapping[str, str], microphysics=None) -> SuiteEngineIdentity:
    if microphysics is not None:
        selection = dict(selection, microphysics=microphysics)
    manifest: dict[str, str] = {}
    for relative in _pinned_paths(selection):
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(
                f"the suite row executes {relative}, absent from {root}"
            )
        manifest[relative] = sha256(path.read_bytes()).hexdigest()
    if selection.get("microphysics") == "tempo":
        from woof.core.tempo_rust import resolve
        library = resolve()
        manifest["runtime/rw-tempo/" + library.name] = sha256(library.read_bytes()).hexdigest()
    try:
        from importlib.metadata import version as _version

        version = _version("woof")
    except Exception:
        version = "unknown"
    pyproject = root / "pyproject.toml"
    if pyproject.is_file():
        for line in pyproject.read_text(encoding="utf-8").splitlines():
            if line.startswith("version = "):
                version = line.split("=", 1)[1].strip().strip('"')
                break
    return SuiteEngineIdentity(
        version=version,
        build_commit=None,
        manifest=MappingProxyType(manifest),
        root=str(root),
    )


_GENERIC_SEAM_CLASSES: dict[type, type] = {}


def _generic_seam_class(base: type) -> type:
    """The engine's generic seam class, spoken to in the suite seam's words.

    The hex backend calls its seam with the species as keyword arguments and
    reads ``selection`` off it; it also copies the
    seam object and re-points its ``_cp`` for device-resident rollback
    (woof.hex.cuda_seam_snapshot_v841), so the seam must stay the engine's
    own object.  The generic entry (``run_mpas_column_physics``) takes
    ``species={...}`` and always takes ``rho_dry`` and ``w`` in phase two.
    This subclass is that
    translation and nothing else: no scheme name appears in it, and every
    other attribute and every byte of state is the engine seam's own.
    """

    cached = _GENERIC_SEAM_CLASSES.get(base)
    if cached is not None:
        return cached

    def _split(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        wanted = tuple(str(name) for name in self.plan.species)
        missing = [name for name in wanted if name not in kwargs]
        if missing:
            raise ValueError(
                f"the generic seam transports {list(wanted)} and the backend "
                f"handed no {missing}")
        return {name: kwargs.pop(name) for name in wanted}

    def run_phase1(self, **kwargs: Any) -> Any:
        # The engine seam publishes the step's extra rates itself, as
        # ``last_scalar_rates`` (the attribute the backend reads).
        return base.run_phase1(self, species=_split(self, kwargs), **kwargs)

    def run_phase2(self, **kwargs: Any) -> Any:
        return base.run_phase2(self, species=_split(self, kwargs), **kwargs)

    cls = type(
        f"{base.__name__}HexWords",
        (base,),
        {
            "__module__": __name__,
            "phase2_takes_density_and_w": True,
            "phase2_takes_winds": True,
            "run_phase1": run_phase1,
            "run_phase2": run_phase2,
            "selection": property(lambda self: dict(self._hex_selection)),
        },
    )
    _GENERIC_SEAM_CLASSES[base] = cls
    return cls


def _load_generic_factory(checkout: str | Path | None, selection: Mapping[str, str]):
    """The generic entry, with the backend's seam contract laid over it."""

    root = _engine_root(checkout)
    import woof.core.mpas_column_suite as suite_module

    entry = getattr(suite_module, GENERIC_ENTRYPOINT, None)
    if entry is None:
        raise ConfigurationRefusal(
            "gpuwm_checkout", str(root),
            f"the engine tree carries no {SUITE_MODULE}.{GENERIC_ENTRYPOINT}, so a "
            "registry-resolved physics selection has no seam to be handed to",
            "an engine carrying the generic column seam",
        )

    def factory(**kwargs: Any) -> Any:
        seam = entry(**kwargs)
        if type(seam).__module__ != SUITE_MODULE:
            raise TypeError("the generic entry no longer returns the suite module's seam")
        seam.__class__ = _generic_seam_class(type(seam))
        seam._hex_selection = dict(selection)
        return seam

    return factory, root, _measure_engine(root, selection)


def _load_suite_factory(checkout: str | Path | None, selection: Mapping[str, str], microphysics=None):
    root = _engine_root(checkout)
    import woof.core.microphysics as microphysics_module
    import woof.core.mpas_column_batch as column_batch
    import woof.core.mpas_column_suite as suite_module
    import woof.core.physics as physics

    factory = getattr(suite_module, SUITE_ENTRYPOINT)
    if not issubclass(suite_module.MpasColumnSuitePhysics, column_batch.MpasColumnBatchPhysics):
        raise TypeError("the suite seam no longer subclasses the frozen batch")
    if suite_module.MpasColumnSuitePhysics._PHASE1_ORCHESTRATION is not physics.PhysicsDriver.compute:
        raise ValueError("suite phase one is no longer PhysicsDriver.compute")
    if suite_module.MpasColumnSuitePhysics._PHASE2_MICROPHYSICS is not microphysics_module.apply:
        raise ValueError("suite phase two is no longer microphysics.apply")
    return factory, root, _measure_engine(root, selection, microphysics)


def _manifest_path(root: Path, relative: str) -> Path:
    if relative.startswith("runtime/rw-tempo/"):
        from woof.core.tempo_rust import resolve
        return resolve()
    return root / relative


def pin_column_batch(checkout: str | Path, row_name: str | None = None) -> dict[str, Any]:
    """The byte record the forecast's git verifier gates for this row."""

    suite = _suite_for_row(row_name or _ROW_NAME_DEFAULT)
    root = Path(checkout).resolve()
    identity = _measure_engine(
        root, suite["selection"],
        None if suite.get("physics") is not None
        else registered_species_rows()[suite["microphysics"]].engine_scheme)
    files = {
        relative: {"sha256": digest, "bytes": (_manifest_path(root, relative)).stat().st_size}
        for relative, digest in identity.manifest.items()
    }
    return {
        "pin": {
            "row": suite["backend_row"],
            "profile": suite["id"],
            "engine_version": identity.version,
            "files": files,
            "frozen_batch_cross_pin": {},
        }
    }


_ROW_NAME_DEFAULT = "mynn_ruc_gsl_column"


def pin_problem(checkout: str | Path, row_name: str | None = None) -> str | None:
    """The door's question for this row: can the named engine run it?

    The row's bytes are measured, not frozen (it is unanchored), so the
    problem is an engine that cannot carry the row at all: a tree without
    ``woof.core.mpas_column_suite`` or without one of the files the
    selected slots execute.  ``None`` when every one is present.
    """

    suite = _suite_for_row(row_name or _ROW_NAME_DEFAULT)
    root = Path(checkout)
    missing = [
        relative
        for relative in _pinned_paths(suite["selection"])
        if not (root / relative).is_file()
    ]
    if missing:
        return (
            f"the engine tree at {root} cannot run the {suite['id']} column "
            f"suite: it carries no {missing}.  REMEDY: an engine carrying "
            "woof.core.mpas_column_suite (gpuwm 2.8.8 or later)"
        )
    if suite.get("physics") is not None:
        from . import engine_registry

        return engine_registry.generic_seam_problem(root)
    generation = dict(suite.get("generation", {}) or {})
    if generation:
        # The seam takes generation selectors only from an engine whose seam has
        # woof.core.mpas_column_suite.resolve_generation.
        # On an older seam the sealed constructor's import fails after the
        # mesh bind, with an ImportError that names no selector; the door
        # asks first.
        seam_source = root / "woof/core/mpas_column_suite.py"
        text = seam_source.read_text(encoding="utf-8", errors="replace")
        if "def resolve_generation" not in text:
            return (
                f"the {suite['id']} column suite selects generation "
                f"{sorted(generation)}, and the engine tree at {root} carries a "
                "woof.core.mpas_column_suite with no resolve_generation, so the "
                "selectors could not reach its RunConfig.  REMEDY: an engine "
                "carrying the generation selectors (gpuwm 2.8.8 or later), or drop the "
                "generation selectors"
            )
    return None


# --------------------------------------------------------------------------
# the backend
# --------------------------------------------------------------------------
def suite_correct_mode_receipt(constructor: Any) -> tuple[Mapping[str, Any] | None, bool]:
    """``(receipt, engine_takes_mode)`` for a suite row; ``(None, True)`` in
    exact mode.

    Both entries record the receipt, so a correct-mode run lists the findings
    it applied (MODE-KEY rule 3) whichever entry built it:

    * the five-slot seam builds its own RunConfig and must take the
      correct-mode keywords; an engine whose seam takes none runs the matched
      arithmetic, keeps the keywords out of the call, and names each finding
      as a gap in the receipt (woof.hex.correct_mode: never a refusal);
    * the generic (registry-resolved) entry carries the corrections as
      RunConfig keys in the physics mapping sealed by ``_from_generic_mapping``;
      an engine without ``woof.correct_mode`` sealed none and the receipt
      names the gaps.

    Exact mode imports nothing and returns ``(None, True)``, so an exact
    receipt is the one written before the key existed.
    """

    if getattr(constructor, "physics_mode", "exact") == "exact":
        return None, True
    from . import correct_mode

    seam = None
    if getattr(constructor, "physics", None) is None:
        try:
            import woof.core.mpas_column_suite as suite_module
        except ImportError:
            suite_module = None
        if suite_module is not None:
            seam = suite_module.MpasColumnSuitePhysics.__init__
    resolved = correct_mode.engine_settings(
        constructor.physics_mode, constructor.physics_correct_skip,
        seam=seam, seam_label="MpasColumnSuitePhysics")
    return MappingProxyType(dict(resolved["receipt"])), bool(resolved["engine_takes_mode"])


#: Whether the whole mesh holds ocean ice, on a run whose ranks each hold a part of it (bind_domain_ocean_ice).
_DOMAIN_OCEAN_ICE = False


def bind_domain_ocean_ice(present: bool) -> None:
    """What rank construction says of the whole mesh, before this rank builds its seam."""

    global _DOMAIN_OCEAN_ICE
    _DOMAIN_OCEAN_ICE = bool(present)


def prime_rank_ocean_ice(seam: Any) -> list[str]:
    """Give a rank whose own columns hold no ocean ice the engine's sea-ice carriers; returns the ones it made.

    The engine's sea ice activates only on a domain containing ocean ice, so the carriers are a property of the
    DOMAIN (woof.ocean.seaice.prime_like says so of a tile).  A rank's columns are not the domain.  THE BREAKAGE
    THIS PREVENTS (rrfs_v2_dev on four cards, 3 km zoom globe, 2026-10-31): the rank with no polar columns held none,
    its fingerprint boundary differed from rank 0's by 18 fields, and the whole forecast stopped at its end with
    "boundary entry paths diverge".  It would also have stepped those columns differently from a run on one card, which
    runs the ice kernel on every water column.  Each kernel is column-local, so this seeds them exactly as the domain's.
    """

    from woof.ocean.seaice import COLD_SEEDED, cold_seed_in_place
    import cupy as cp

    fields, cfg = seam._driver.fields, seam._cfg
    if "ice_volume" in fields or not cfg.seaice_prognostic or not cfg.sf_surface_physics:
        return []
    absent = sorted(key for key in COLD_SEEDED if key not in fields)
    for key in absent:
        fields[key] = cp.zeros_like(fields["tsk"])
    cold_seed_in_place(fields, cfg, frozenset(absent))
    return absent


class PersistentSuiteCudaPhysicsBackendV841(PersistentTwoPhaseCudaPhysicsBackendV841):
    """The frozen transactional backend, bound to the selectable suite."""

    def __init__(
        self,
        *,
        constructor: SealedSuiteConstructorV841,
        prep_geometry: CudaMpasToPhysGeometryV841,
        kernel_cache: KernelCache,
        profile_id: str,
        gwdo_static: Any = None,
        gwdo_kernel_cache: Any = None,
        arwen_checkout: str | Path | None = None,
        rollback_snapshot: bool = True,
    ) -> None:
        if not isinstance(constructor, SealedSuiteConstructorV841) or constructor._seal is not _CONSTRUCTOR_SEAL:
            raise TypeError("constructor must be SealedSuiteConstructorV841.from_mapping")
        if gwdo_static is not None or gwdo_kernel_cache is not None:
            raise ConfigurationRefusal(
                "gwdo_static",
                "supplied",
                "the suite row carries its drag inside the engine's PBL slot "
                "(GSL, gwd_opt=3); the hex-side YSU-GWDO on the same momentum "
                "tendency would count orographic drag twice",
                "no gwdo_static for the suite row",
            )
        if not isinstance(rollback_snapshot, bool):
            raise TypeError("rollback_snapshot must be a bool")
        if not isinstance(prep_geometry, CudaMpasToPhysGeometryV841):
            raise TypeError("prep_geometry must be sealed v8.4.1 MPAS preparation geometry")
        prep_geometry.validate()
        if prep_geometry.n_cells != constructor.n_columns:
            raise ValueError("preparation geometry and suite constructor column counts differ")
        suite = physics_suite(profile_id)
        if dict(suite["selection"]) != constructor.selection:
            raise ValueError(
                f"constructor selection {constructor.selection} is not profile "
                f"{profile_id!r}'s {dict(suite['selection'])}"
            )
        if constructor.physics is not None:
            factory, root, identity = _load_generic_factory(arwen_checkout, constructor.selection)
        else:
            factory, root, identity = _load_suite_factory(arwen_checkout, constructor.selection, constructor.microphysics_scheme)
        self._correct_mode_receipt, self._engine_takes_mode = suite_correct_mode_receipt(constructor)
        self._profile = suite
        self._constructor = constructor
        self._engine = identity
        self._prep_geometry = prep_geometry
        self._kernel_cache = kernel_cache
        self._gwdo_static = None
        self._gwdo_kernel_cache = None
        self._factory = factory
        self._arwen_root = root
        # The radiation helper scope the frozen backend binds in its own
        # constructor (woof.hex.cuda_arwen_helper_scope_v2, landed with the
        # hex-busy radiation work after this row was written).  Phase one
        # runs inside it for every row; without it the first step stopped
        # with AttributeError '_helper_scope' (measured 2026-10-07,
        # conus12km).
        from .cuda_arwen_helper_scope_v2 import bind_arwen_helpers_v2

        self._helper_scope = bind_arwen_helpers_v2(root)
        self._rollback_snapshot = rollback_snapshot
        self._seam = self._new_seam()
        self._phase = "boundary"
        self._boundary_snapshot = None
        self._step_start = None
        self._candidate_scalar_target = None
        self._candidate_scalar_backup = None
        self._pending_gwdo_result = None
        self._last_gwdo_result = None
        self._pending_refl10cm = None
        self._committed_refl10cm = None
        self._gwdo_calls = 0
        # SG2 (woof.hex.lagged_heating_v841): this __init__ does not call the
        # frozen one, so the carry's three slots are set here as well.  THE
        # BREAKAGE: without them restart_state() before the first committed
        # step raised AttributeError '_lagged_diabatic_theta' on every suite
        # and generic-seam row -- the step-0 boundary fingerprint, the
        # samehost identity's initial boundary and the cycling step-0
        # background capture (verifier, correct/storm-growth, 2026-10-13).
        # tests/test_lagged_heating_sg2_wiring.py holds this __init__ to
        # every slot the frozen __init__ sets.
        self._pending_lagged_diabatic_theta = None
        self._lagged_diabatic_theta = None
        self._lagged_heating_restored = None
        self._gf_forcing_seen = False
        self._last_receipt = self._base_receipt()
        self._private_binding_guard()
        from . import kernel_prewarm
        kernel_prewarm.start_static(self._seam._cfg)

    def _new_seam(self) -> Any:
        kwargs = self._engine_kwargs()
        physics = kwargs.get("physics", {})
        longwave = physics.get("ra_lw_physics")
        if longwave is None:
            longwave = physics.get("ra_physics")
        if longwave == 31:
            from woof.core.held_suarez_mesh import HeldSuarezMeshMomentum
            # Rollback reconstructs the column seam. The stateless RK binding
            # remains attached to the same dycore and must survive that rebuild.
            if getattr(self, "_held_suarez_momentum", None) is None:
                self._held_suarez_momentum = HeldSuarezMeshMomentum()
            kwargs["rk_momentum"] = self._held_suarez_momentum
        seam = self._factory(**kwargs)
        expected = {
            "run_phase1", "run_phase2", "export_state", "restore_state",
            "accumulated_precipitation", "surface_classification", "selection",
        }
        missing = sorted(name for name in expected if not hasattr(seam, name))
        if missing:
            raise TypeError(f"suite seam lacks {missing}")
        if dict(seam.selection) != self._constructor.selection:
            raise ValueError("the suite seam was built with another selection")
        self._validate_surface_classification(seam)
        if _DOMAIN_OCEAN_ICE:
            prime_rank_ocean_ice(seam)
        # A coupled hex ocean owns ocean ice through the Noah-MP sea-ice hook,
        # as on the frozen route. Without it here the ocean was attached but
        # never stepped, woof's own ice model stayed live, and every
        # storm-following move republished the frozen ocean ice over the
        # engine's: polar columns flipped water/sea ice/glacier, the history
        # census refused, Noah-MP's glacier water budget failed and the next
        # move refused "carried ice volume has no carried ice area" (the storm
        # hex storm-following, box M 2026-10-27). Prescribed runs keep the
        # route's published ice owner.
        from .ocean_pwp import hex_owns_ice
        if hex_owns_ice() and self._constructor.selection.get("land_surface_scheme") == "noahmp":
            from .arwen_seaice import install_noahmp_seaice
            install_noahmp_seaice(self, seam)
        return seam

    def noahmp_routing_census(self) -> dict[str, Any] | None:
        """Noah-MP's pre-LSM layout counts; a run on another land surface has none to give.

        THE BREAKAGE THIS PREVENTS (rrfs_v2_dev, one card and four, 2026-10-31): the frozen backend's receipt reads
        this census, whose keys only a Noah-MP classification carries, so every column-suite run on RUC or Noah
        stopped building its physics with KeyError 'sflx_land_columns' from 0.4.5 on.  Every reader takes None
        as "no census" and judges this one only when the land surface is Noah-MP.
        """

        if self._constructor.selection["land_surface_scheme"] != "noahmp":
            return None
        return super().noahmp_routing_census()

    def _validate_noahmp_census(self, *, require: bool) -> Mapping[str, Any] | None:
        """The land surface's own census (RUC's or Noah-MP's)."""

        land = self._constructor.selection["land_surface_scheme"]
        if land == "noahmp":
            return super()._validate_noahmp_census(require=require)
        raw = getattr(self._seam, "last_ruc_census", None)
        if raw is None and land != "ruc":
            # A land surface with no execution census of its own (Noah, or
            # none at all, on the generic seam).  The receipt still says
            # which columns the surface call saw: the engine's own published
            # land/water split, labelled as that and nothing more.  Found on
            # G1 2026-10-11: Morrison over Noah stopped at step 1 with
            # "'NoneType' object is not iterable" where the frozen backend
            # copies the census into its step receipt.
            return MappingProxyType({
                "land_surface_scheme": land,
                "census": "none published by this land surface; the engine's "
                          "surface classification follows",
                **json.loads(json.dumps(dict(self._seam.surface_classification), default=str)),
            })
        if raw is None:
            if require and land == "ruc":
                raise ValueError("the RUC land surface published no census")
            return None
        if not isinstance(raw, Mapping):
            raise TypeError("RUC census is not a mapping")
        return MappingProxyType(json.loads(json.dumps(dict(raw), default=str)))

    def _private_binding_guard(self) -> None:
        generic = self._constructor.physics is not None
        # The generic seam is the engine's own class with hex's calling
        # words laid over it (_generic_seam_class): its base is the suite
        # module's.
        bound = type(self._seam).__mro__[1] if generic else type(self._seam)
        if bound.__module__ != SUITE_MODULE:
            raise TypeError("private radius binding requires the suite seam class")
        measured = _measure_engine(
            self._arwen_root, self._constructor.selection,
            None if generic else self._constructor.microphysics_scheme)
        if dict(measured.manifest) != dict(self._engine.manifest):
            raise ValueError("the suite engine's executed bytes moved during the run")
        state = getattr(self._seam, "_state", None)
        radii = ("effc", "effi", "effs")
        if generic:
            # The radii a scheme publishes are the species row's, which was
            # derived from the engine's plan; a scheme with none binds none.
            from .species_row import species_row_for_scheme

            radii = tuple(species_row_for_scheme(str(self._profile["microphysics"])).radius_names)
        for name in radii:
            if state is None or not hasattr(state, name):
                raise TypeError(f"suite radius carrier {name!r} is missing")

    def computes_reflectivity(self) -> bool:
        """False only on the generic seam with microphysics off.

        THE BREAKAGE: the engine refuses refl_10cm_due with mp_physics=0
        ("no scheme computes a reflectivity"), so a microphysics-off run
        stopped at its first history step.  The rule is the engine's own.
        """

        if self._constructor.physics is None:
            return super().computes_reflectivity()
        return int(dict(self._constructor.physics).get("mp_physics", 1)) != 0

    def runs_microphysics(self) -> bool:
        """False only on the generic seam with microphysics off (mp_physics=0).

        SG2: native calls microphysics_to_MPAS, the only producer of
        rt_diabatic_tend, only when config_microp_scheme /= 'off'
        (mpas_atm_time_integration.F:2806); such a run carries no heating.
        """

        if self._constructor.physics is None:
            return super().runs_microphysics()
        return int(dict(self._constructor.physics).get("mp_physics", 1)) != 0

    def _species_row(self, scalar_names: Sequence[str]) -> Any:
        """On the generic seam, the row the selection resolved to.

        THE BREAKAGE THIS PREVENTS: Kessler and microphysics off both carry
        (qv, qc, qr), so the names-only lookup gave microphysics off
        Kessler's row and phase two refused the off seam's own bucket set
        ("frozen Arwen v2 cumulative precipitation keys changed: ['RAINC'],
        expected ['RAINC', 'RAINNC']", 60 km and 30 km global, 2026-10-11).
        """

        if self._constructor.physics is None:
            return super()._species_row(scalar_names)
        from .species_row import species_row_for_scheme

        row = species_row_for_scheme(str(self._profile["microphysics"]))
        if tuple(row.names()) != tuple(scalar_names):
            raise ValueError(
                f"the selection's species row {row.scheme_name!r} carries "
                f"{list(row.names())}, not the scalar block {list(scalar_names)}")
        return row

    def _base_receipt(self) -> dict[str, Any]:
        receipt = super()._base_receipt()
        receipt["schema"] = CUDA_SUITE_PHYSICS_V841_SCHEMA
        receipt["frozen_adapter_schema"] = "mpas-port.cuda-woof-physics-v841/v3"
        census = receipt.pop("last_noahmp_census", None)
        receipt["last_land_surface_census"] = census
        receipt["suite"] = {
            "profile": self._profile["id"],
            "backend_row": self._profile["backend_row"],
            "selection": dict(self._constructor.selection),
            "n_soil_levels": self._constructor.n_soil_levels,
            "engine_root": self._engine.root,
            "engine_version": self._engine.version,
            "differences_from_shaped_after": list(
                self._profile["differences_from_shaped_after"]
            ),
        }
        receipt["gwdo"] = {
            "enabled": False,
            "hex_side_ysu_gwdo": False,
            "engine_drag": self._constructor.selection["drag_scheme"],
        }
        receipt["nonclaims"] = list(receipt.get("nonclaims", [])) + [
            "no claim of parity with the configuration the profile is shaped after",
            "the execution-provenance object's 'gwd off' names the hex-side composer; "
            "the engine PBL slot carried the selected drag",
        ]
        return receipt

    def begin_step(self, **kwargs: Any):
        raw = super().begin_step(**kwargs)
        try:
            raw, extra = self._with_scalar_rates(raw, kwargs["scalar_names"])
        except Exception:
            # Same transaction law as a failure inside the frozen phase one:
            # the boundary comes back and the error propagates.
            self._restore_boundary(self._boundary_snapshot)
            raise
        receipt = dict(self._last_receipt)
        cfg = self._seam._cfg
        if not any(getattr(cfg, key) for key in
                   ("sf_sfclay_physics", "sf_surface_physics", "bl_pbl_physics")):
            # A forcing-only selection has no surface stack to schedule.
            # Retain actual counters so a stray call still fails the proof.
            receipt["cadence"] = {
                **receipt["cadence"], "surface_pbl_enabled": False}
        if extra:
            receipt["extra_scalar_rates"] = list(extra)
        census = receipt.pop("noahmp_census", None)
        receipt["land_surface_census"] = census
        receipt["engine_drag"] = {
            "scheme": self._constructor.selection["drag_scheme"],
            "where": "engine PBL slot, added to RUBLTEN/RVBLTEN before the raw rates",
        }
        self._last_receipt = receipt
        return raw

    def _with_scalar_rates(self, raw: Any, scalar_names: Sequence[str]):
        """Attach the seam's extra phase-one rates (MYNN's nwfa/nifa).

        The suite seam publishes them as ``last_scalar_rates`` (woof
        mpas_column_suite run_phase1, from MpasColumnPhysicsTendencies.
        scalar_rates); the frozen phase one this backend inherits builds
        its carrier from the six masses only, and its bytes are pinned by
        the x4 release, so the rates join here.  With none, the frozen
        carrier is returned untouched.  With some, the carrier is rebuilt
        against the run's own species row, so a rate for a species the row
        does not transport refuses instead of being coupled into nothing,
        and hex couples each one by dry mass like every other species
        (couple_cells_v841_f32), which is how NOAA MPAS adds rnwfablten and
        rnifablten (mpas_atmphys_todynamics.F:394-405, 1d6e370e).
        """

        rates = dict(getattr(self._seam, "last_scalar_rates", None) or {})
        if not rates:
            return raw, ()
        from .cuda_physics_v841 import CudaRawColumnPhysicsV841

        names = tuple(str(name).strip().lower() for name in scalar_names)
        clash = sorted(set(rates) & set(raw.dscalars))
        if clash:
            raise ValueError(
                f"phase one returned {clash} both as mass rates and as extra "
                "scalar rates")
        shape = (self._constructor.n_levels, self._constructor.n_columns)
        for name, rate in rates.items():
            require_resident_array(
                f"suite_phase1.scalar_rates[{name!r}]", rate,
                dtype=np.float32, shape=shape)
        dscalars = {name: rate for name, rate in raw.dscalars.items() if name in names}
        dscalars.update(rates)
        rebuilt = CudaRawColumnPhysicsV841(
            du=raw.du,
            dv=raw.dv,
            dtheta=raw.dtheta,
            dscalars=dscalars,
            time_seconds=raw.time_seconds,
            execution_provenance=raw.execution_provenance,
            scalar_names=names,
        )
        rebuilt.validate(n_vert_levels=shape[0], n_cells=shape[1])
        return rebuilt, tuple(sorted(rates))

    def diagnostic_snapshot(self, *, selective=False, resident=False) -> CudaArwenDiagnosticSnapshotV841:
        """The frozen snapshot with the row's soil width."""

        if self._phase != "boundary":
            raise RuntimeError("diagnostic_snapshot is legal only at a committed boundary")
        self._private_binding_guard()
        names = (*_REQUIRED_ARWEN_EXPORT_FIELDS, *_OPTIONAL_ARWEN_EXPORT_FIELDS)
        exported, export_inventory = diagnostic_export(self._seam, names, selective=selective, resident=resident)
        if (
            not isinstance(exported, Mapping)
            or set(exported) != {"identity", "arrays", "scalars"}
            or not isinstance(exported["arrays"], Mapping)
        ):
            raise ValueError("suite export_state schema changed")
        cp = __import__("cupy")
        nlev = self._constructor.n_levels
        ncol = self._constructor.n_columns
        nsoil = self._constructor.n_soil_levels
        export_arrays = exported["arrays"]
        selected: dict[str, Any] = {}
        hashes: dict[str, Any] = {}
        h2d = 0
        for name in names:
            key = f"fields/{name}"
            if key not in export_arrays and f"diag/{name}" in export_arrays:
                key = f"diag/{name}"
            if key not in export_arrays:
                if name in _REQUIRED_ARWEN_EXPORT_FIELDS:
                    raise ValueError(f"suite export lacks required {key!r}")
                continue
            host = export_arrays[key] if resident else np.asarray(export_arrays[key])
            if host.dtype != np.dtype(np.float32):
                raise TypeError(f"suite export {key!r} is not FP32")
            if name in _SOIL_DIAGNOSTIC_FIELDS:
                if nsoil == 0:
                    # Land surface off (sf_surface_physics=0): the engine
                    # still exports tslb/smois at its allocated width (4
                    # levels), a column no scheme initialized or integrated.
                    # It is NOT published: the snapshot carries no soil
                    # group, and the publication (capture_snapshot's required
                    # set, physical_snapshot_gate, write_snapshot_netcdf)
                    # carries an absent soil column for a zero-level row.
                    # This retires guard hexphys-no-soil-publication (
                    # global 30 km, 2026-10-11), which refused the run here.
                    continue
                if host.shape == (nsoil, 1, ncol):
                    normalized = host[:, 0, :]
                elif host.shape == (nsoil, ncol):
                    normalized = host
                else:
                    raise ValueError(f"suite export {key!r} has shape {host.shape}")
                shape = (nsoil, ncol)
            else:
                normalized = host.reshape(-1)
                if normalized.shape != (ncol,):
                    raise ValueError(f"suite export {key!r} has shape {host.shape}")
                shape = (ncol,)
            if resident:
                device_value = cp.ascontiguousarray(normalized)
            else:
                normalized = np.array(normalized, copy=True, order="C")
                device_value = cp.asarray(normalized)
            require_resident_array(f"suite_diagnostic.{name}", device_value, dtype=np.float32, shape=shape)
            selected[name] = device_value
            if not resident:
                hashes[key] = _array_identity(normalized)
                h2d += int(device_value.nbytes)
        public_precip = self._seam.accumulated_precipitation()
        required_precip = required_precipitation_keys(
            generic=self._constructor.physics is not None,
            microphysics=str(self._profile["microphysics"]))
        if not required_precip <= set(public_precip):
            raise ValueError(f"suite precipitation inventory changed: {sorted(public_precip)}")
        precipitation = {
            key.lower(): cp.array(value, copy=True, order="C")
            for key, value in public_precip.items()
        }
        gwdo = {
            "dusfcg": cp.zeros((ncol,), dtype=cp.float32),
            "dvsfcg": cp.zeros((ncol,), dtype=cp.float32),
            "dtaux3d": cp.zeros((nlev, ncol), dtype=cp.float32),
            "dtauy3d": cp.zeros((nlev, ncol), dtype=cp.float32),
            "rubldiff": cp.zeros((nlev, ncol), dtype=cp.float32),
            "rvbldiff": cp.zeros((nlev, ncol), dtype=cp.float32),
        }
        surface = {k: v for k, v in selected.items() if k not in _SOIL_DIAGNOSTIC_FIELDS}
        surface.update(history_cumulus_rain_rate(self, self._seam, precipitation["rainc"], cp))
        soil = {k: v for k, v in selected.items() if k in _SOIL_DIAGNOSTIC_FIELDS}
        classification = self._validate_surface_classification()
        census = self._validate_noahmp_census(require=False)
        receipt = {
            "schema": CUDA_SUITE_PHYSICS_V841_SCHEMA,
            "boundary": "committed",
            "full_export_d2h_bytes": _snapshot_nbytes(exported),
            "exported_array_count": len(export_arrays),
            "available_array_count": len(export_inventory),
            "selected_export_hashes": hashes,
            "selected_export_h2d_bytes": h2d,
            "n_soil_levels": nsoil,
            "surface_classification": dict(classification),
            "last_land_surface_census": None if census is None else dict(census),
            "gwdo_note": "hex-side YSU-GWDO absent; zero-filled carriers",
        }
        return CudaArwenDiagnosticSnapshotV841(
            surface=MappingProxyType(surface),
            soil=MappingProxyType(soil),
            precipitation=MappingProxyType(precipitation),
            gwdo=MappingProxyType(gwdo),
            metadata=MappingProxyType(
                {
                    "step_index": int(self._seam.step_index),
                    "time_seconds": float(self._seam.elapsed_seconds),
                    "call_counts": MappingProxyType(dict(self._seam.call_counts)),
                    "surface_classification": classification,
                    "last_noahmp_census": None,
                    "last_land_surface_census": census,
                    "gwdo_enabled": False,
                    "gwdo_calls": 0,
                    "gwdo_has_last_result": False,
                }
            ),
            receipt=MappingProxyType(receipt),
        )

    def restart_state(self) -> Mapping[str, Any]:
        """The frozen boundary payload under this adapter's own schema, so a
        payload and the receipt beside it name the same adapter."""

        payload = dict(super().restart_state())
        payload["schema"] = CUDA_SUITE_PHYSICS_V841_SCHEMA
        return payload

    def restore_restart_state(self, payload: Mapping[str, Any]) -> None:
        if not isinstance(payload, Mapping) or payload.get("schema") != CUDA_SUITE_PHYSICS_V841_SCHEMA:
            raise ValueError("backend restart schema is not the column suite's")
        frozen = dict(payload)
        frozen["schema"] = "mpas-port.cuda-woof-physics-v841/v3"
        super().restore_restart_state(frozen)

    def _restart_identity(self) -> Mapping[str, Any]:
        # The frozen adapter's identity fields first, then the suite's own.
        # THE BREAKAGE: the step transaction verifies a rollback by reading
        # the frozen identity fields (adapter contract, engine commit and
        # manifest, contract surface, constructor) back from this identity
        # against the receipt, which this class inherits unchanged.  Without
        # them every rollback on a suite or generic-seam backend reported
        # "adapter rollback restart identity changed field
        # 'adapter_contract_sha256'" and buried the refusal that caused it
        # (hex 30 km global at TEMPO, 2026-10-11).
        identity = dict(super()._restart_identity())
        identity.update({
            "schema": CUDA_SUITE_PHYSICS_V841_SCHEMA,
            "profile": self._profile["id"],
            "selection": dict(self._constructor.selection),
            "engine_manifest": dict(self._engine.manifest),
            "engine_version": self._engine.version,
            "constructor_identity_sha256": self._constructor.identity_sha256,
        })
        return identity


def build_column_backend(
    row: Any,
    *,
    constructor: Any,
    prep_geometry: Any,
    kernel_cache: Any,
    gwdo_static: Any = None,
    gwdo_kernel_cache: Any = None,
    checkout: Any = None,
    cell_area_m2: Any = None,
    seam_options: Mapping[str, Any] | None = None,
    rollback_snapshot: bool = True,
) -> PersistentSuiteCudaPhysicsBackendV841:
    """The provider-row builder the forecast's stack build calls."""

    if seam_options:
        raise ConfigurationRefusal(
            "seam_options", sorted(seam_options),
            "the suite row reads no seam options", "no seam options",
        )
    del cell_area_m2  # every suite slot is columnar; nothing weighs by area
    suite = _suite_for_row(row.name)
    return PersistentSuiteCudaPhysicsBackendV841(
        constructor=constructor,
        prep_geometry=prep_geometry,
        kernel_cache=kernel_cache,
        profile_id=str(suite["id"]),
        gwdo_static=gwdo_static,
        gwdo_kernel_cache=gwdo_kernel_cache,
        arwen_checkout=checkout,
        rollback_snapshot=bool(rollback_snapshot),
    )


# --------------------------------------------------------------------------
# what the forecast records about the row
# --------------------------------------------------------------------------
def physics_suite_record(row_name: str) -> Mapping[str, Any]:
    """The profile row as the forecast receipt and host carry it."""

    from woof.core.mpas_column_suite import orographic_statistic_names

    suite = _suite_for_row(row_name)
    selection = dict(suite["selection"])
    return MappingProxyType(
        {
            "profile": suite["id"],
            "backend_row": suite["backend_row"],
            "selection": selection,
            "physics": None if suite.get("physics") is None else dict(suite["physics"]),
            "microphysics": suite["microphysics"],
            "n_soil_levels": int(suite["n_soil_levels"]),
            "soil_level_depths_m": (
                None if suite["soil_level_depths_m"] is None
                else list(suite["soil_level_depths_m"])),
            "fractional_seaice": suite["fractional_seaice"],
            "mpas_selectors": dict(suite["mpas_selectors"]),
            "generation": dict(suite.get("generation", {}) or {}),
            "synthesized": bool(suite.get("synthesized", False)),
            "drag_reads_statistics": bool(
                suite.get("orographic_statistics")
                if suite.get("physics") is not None
                else orographic_statistic_names(selection["drag_scheme"])
            ),
            "differences_from_shaped_after": list(
                suite["differences_from_shaped_after"]
            ),
        }
    )


_SUITE_CONFIG_CLASSES: dict[tuple, type] = {}


def _suite_config_base(run_type: type) -> type:
    """The non-GWDO counterpart a suite run records (see :func:`suite_config`, "The drag")."""

    from .config_v841 import (
        V841MpasColumnPhysicsConfig,
        V841MpasColumnPhysicsGwdoConfig,
        V841UpperDampingColumnConfig,
    )

    if not issubclass(run_type, V841MpasColumnPhysicsGwdoConfig):
        return run_type
    from dataclasses import fields

    if any(item.name == "config_number_cam_damping_levels" for item in fields(run_type)):
        return V841UpperDampingColumnConfig
    return V841MpasColumnPhysicsConfig


def _suite_config_class(run_type: type, profile_id: str, selectors: Mapping[str, str]) -> type:
    """The configuration class of one (run type, profile, selectors), built once per process.

    The class is made here because the fields a suite records depend on the run's type, so no module attribute
    can name it.  THE BREAKAGE ``__reduce__`` PREVENTS (rrfs_v2_dev on four cards, run_fb4825f5121040b2,
    2026-10-31): each rank process's request carries the run's configuration, pickle finds a class by its module
    attribute, and every column-suite run on rank processes stopped before its first step with PicklingError.  An
    instance now travels as what built its class plus its field values, and the rank's class comes from this
    function, so rank 0 and a rank process hold one recipe and the same words.
    """

    from dataclasses import fields, make_dataclass

    selectors = dict(selectors)
    key = (run_type, profile_id, tuple(sorted(selectors.items())))
    cls = _SUITE_CONFIG_CLASSES.get(key)
    if cls is not None:
        return cls
    base = _suite_config_base(run_type)
    base_fields = [item.name for item in fields(base)]
    defaults = {item.name: item.default for item in fields(run_type)}
    unknown = sorted(name for name in selectors if name not in defaults)
    if unknown:
        raise ConfigurationRefusal(
            "physics_suite", profile_id,
            f"the profile names selectors {unknown} the configuration "
            f"type {run_type.__name__} does not carry",
            "a profile whose selectors are configuration fields",
        )

    def validate(self) -> None:
        neutral = {name: getattr(self, name) for name in base_fields}
        for name in selectors:
            if name != "config_microp_scheme":
                neutral[name] = defaults[name]
        # Every frozen value through the run's own type, its YSU-GWDO
        # composer fields at the values the run's bind gave them (held,
        # not executed).
        neutral.update(dict(self.config_frozen_lane_drag_fields))
        run_type(**neutral).validate()
        for name, want in selectors.items():
            if getattr(self, name) != want:
                raise ConfigurationRefusal(
                    name, getattr(self, name),
                    f"the {profile_id} column suite executes {want!r}",
                    f"{name}={want!r}",
                )
        if self.config_physics_suite_profile != profile_id:
            raise ConfigurationRefusal(
                "config_physics_suite_profile",
                self.config_physics_suite_profile,
                "the configuration names another physics-suite profile",
                f"config_physics_suite_profile={profile_id!r}",
            )

    def __reduce__(self):
        values = {item.name: getattr(self, item.name) for item in fields(self) if item.init}
        return _restore_suite_config, (run_type, profile_id, tuple(sorted(selectors.items())), values)

    cls = make_dataclass(
        f"{run_type.__name__}Suite",
        [("config_physics_suite_profile", str, ""),
         ("config_frozen_lane_drag_fields", tuple, ())],
        bases=(base,),
        frozen=True,
        slots=True,
        namespace={"validate": validate, "__reduce__": __reduce__, "__module__": __name__},
    )
    _SUITE_CONFIG_CLASSES[key] = cls
    return cls


def _restore_suite_config(run_type: type, profile_id: str, selectors: tuple, values: Mapping[str, Any]) -> Any:
    """Pickle's way back: the instance's class from its recipe, this process's own copy of it."""

    return _suite_config_class(run_type, profile_id, dict(selectors))(**values)


def suite_config(
    config: Any, row_name: str, mpas_selectors: Mapping[str, str] | None = None
) -> Any:
    """The run's configuration, recording the selectors it executes.

    The frozen lane's configuration types pin their physics selectors to the
    frozen row's literals (YSU, revised MO, Noah-MP, YSU-GWDO).  A suite run
    under that type would write those literals into every receipt while
    MYNN, RUC and the GSL drag executed.  Two things follow:

    * The selectors.  The returned configuration validates every frozen
      value through the run's OWN type -- with that type's selector literals
      in place of the suite's, exactly as the GWDO and Smagorinsky subtypes
      neutralize their own knobs -- and then holds the suite's selectors to
      the profile.
    * The drag.  The dycore's receipt lane (``cuda_driver``) reads a GWDO
      configuration type as "the hex-side YSU-GWDO composed phase one" and
      requires that composer's execution provenance.  The suite runs no
      hex-side drag (its drag is the engine PBL slot's), so the recorded
      type is the run type's NON-GWDO counterpart: every field the run type
      carries except the external YSU composer's six, mixing and damping
      included.  The dycore then records the hex-side drag as off, which is
      what the hex side ran; the backend receipt names the engine drag.

    Nothing in the dycore reads the physics selectors; the row's adapter is
    what selects the schemes, and this is the record of it.
    """

    from dataclasses import fields

    suite = _suite_for_row(row_name)
    # A named suite records the MPAS spelling it was resolved from (the
    # fork's bl_mynnedmf, say, for the engine's MYNN); the profile's own
    # spelling otherwise.  Both name the same executed slots: the profile's
    # selection is what the seam is built with either way.
    selectors = dict(suite["mpas_selectors"] if mpas_selectors is None else mpas_selectors)
    run_type = type(config)
    base = _suite_config_base(run_type)
    base_fields = [item.name for item in fields(base)]
    dropped = sorted({item.name for item in fields(run_type)} - set(base_fields))
    if any(not name.startswith("gwdo_") for name in dropped):
        raise ConfigurationRefusal(
            "physics_suite", suite["id"],
            f"the run type {run_type.__name__} carries {dropped}, which the "
            f"suite's recorded type {base.__name__} would drop; only the "
            "external YSU-GWDO composer's fields may be dropped",
            "a run type whose non-drag fields the column configuration carries",
        )
    cls = _suite_config_class(run_type, str(suite["id"]), selectors)
    values = {name: getattr(config, name) for name in base_fields}
    values.update(selectors)
    held = tuple((name, getattr(config, name)) for name in dropped)
    return cls(
        **values,
        config_physics_suite_profile=str(suite["id"]),
        config_frozen_lane_drag_fields=held,
    )
