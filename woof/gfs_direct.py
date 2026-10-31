"""GFS GRIB2 -> native GPU initialization -> stock-WRF direct export."""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
from collections.abc import Sequence as _ABCSequence
from types import SimpleNamespace
from typing import Mapping, Sequence

import numpy as np

from woof import __version__
from woof.bridges import decode_failure_message
from woof.explain import warn
from woof.vertical_adaptation import (
    adapt_experiment_for_statics,
    vertical_coordinate_receipt as _vertical_coordinate_receipt,
)
from woof.era5_direct import (
    _announce_adaptation,
    _canonical_surface,
    _load_static,
    _sha256,
    _static_from_geog,
    _survey_static_catalog,
    _write_geometry_receipt,
    _write_static_cache,
)
from woof.experiment import load_experiment
from woof.ingest.grib import Era5Snapshot
from woof.ingest.pressure_ladder import (
    CERTIFIED_PRESSURE_LEVELS_HPA, manifest_source_top_pa, validate_ladder,
)

from woof.ingest.source_coverage import (ForcingSeriesRefusal,
                                          PreparationRefusal,
                                          owns_source_coverage_refusal)
from woof.ingest.horiz import (
    _regular_coordinates,
    interpolate_era5_to_lambert,
    orient_global_source_longitudes,
    unrolled_source_ring,
)
from woof.ingest.lateral_bc import (
    StateBoundaryFrames,
    attach_lateral_boundaries,
    start_last_forcing_order,
)
from woof.ingest.soil_downscale import (
    declared_soil_texture_downscale, soil_mesh_plan_from_case)
from woof.ingest.cg_topo import RootTerrainBlend
from woof.ingest.boundary_stream import (
    HIERARCHY_HEAD_DIRNAME,
    PreparedTreeWriter,
    TreeStartStates,
    as_posted_placeholder,
    chained_enabled,
    input_plan,
    input_plan_sha256,
    domain_tree_head_fields,
    prepared_head_urban_columns,
    producer_device_bytes,
    remove_unfinished_tree,
)
from woof.ingest.prepared_cache import (
    prepared_cache_identity,
)
from woof.ingest.cpu_backend import host_step_workers
from woof.ingest.memory_refusal import InitializationMemoryRefused
from woof.ingest.preparation_price import (
    price_forcing_preparation, price_preparation_floor)
from woof.ingest.preprocess_backend import (
    preprocess_math_call,
    admit_preparation,
    preprocess_identity,
    release_backend_memory,
    resolve_preprocess_backend,
)
from woof.ingest.real import initialize_real
from woof.ingest.land_state import land_state_context
from woof.ingest.ruc_soil import preprocess_land_surface_soil
from woof.ingest.soil import (door_reconciled_soil_category,
                               soil_source_orography)
from woof.ingest.water_temperature import (
    MODIS_LAKE_CATEGORY, WaterTemperatureStatics,
    announce_water_temperature, assemble_for_route)
from woof.moisture_floor_receipt import moisture_floor_proof_entry
from woof.native_domain_artifacts import (
    _atomic_staging_sibling,
    published_path_refusal,
    root_domain_artifact_binding,
    write_child_domain_artifacts,
    write_domain_static_files,
)
from woof.native_hierarchy import hierarchy_moisture_floor_receipts
from woof.native_wrf_contract import (
    native_geometry_contract,
    native_static_export_fields,
    require_land_terrain,
    validate_native_lambert_contract,
    validate_native_lambert_contracts,
    verify_native_static_receipt,
)
from woof.source_hierarchy import (
    initialize_and_export_regular_source_hierarchy,
    prepare_regular_source_hierarchy_head,
    seal_regular_source_hierarchy,
)
from woof.open_files import (
    ensure_descriptor_headroom,
    required_descriptors,
)
from woof.physics_compat import (
    acknowledgement_delivery,
    multi_domain_physics_selection,
    single_domain_physics_selection,
)
from woof.wrf_direct import (
    StockWrfExportUnsupported,
    export_prepared_wrf,
    stock_wrf_export_not_requested,
    stock_wrf_export_refused,
)


INPUT_MANIFEST_SCHEMA = "gpuwm-gfs-direct-input-manifest-v1"
BRIDGE_SCHEMA = "gpuwm-gfs-grib2-bridge-v1"

#: The live progress document this stage publishes while it runs.
#: Shaped like the forecast runner's ``progress.json`` -- ``schema`` and
#: ``status`` -- because ``woof go``'s heartbeat reads ``status`` off
#: whichever file the running stage names, and a second shape would mean
#: a second reader.
PREPARE_PROGRESS_SCHEMA = "gpuwm.prepare-progress/v1"

#: The phases this stage announces, in the order it passes through them.
#: Named as one tuple so the publisher, the acceptance test and any
#: reader agree about what "how far in is it" can say.
PREPARE_PHASES = (
    "verify_inputs", "static_build", "decode", "initialize_all_times",
    "write_prepared_cache", "direct_wrf_export", "publish",
)
PROOF_SCHEMA = "gpuwm-gfs-direct-wrf-proof-v3"
#: The multi-domain product.  v2 adds the front-door physics receipt the
#: v3 single-domain proof already carried; v1 documents written by 1.1.0
#: and earlier stay independently verifiable and are never promoted to
#: v2 by inference, exactly as the direct proof's v2 is not.
HIERARCHY_PROOF_SCHEMA = "gpuwm-gfs-native-hierarchy-proof-v2"
#: The isobaric ladder every GFS subset carries, and the floor this
#: front door requires -- not the ceiling.  A case whose model top sits
#: above 100 hPa is fetched with the ladder EXTENDED UPWARD
#: (``woof fetch --source gfs --p-top-pa 5000`` adds the 70 and 50 hPa
#: levels), and the bridge reports the ladder it actually decoded in its
#: gate.  This constant is what that ladder must still contain, and what
#: stands in for a directory fetched before the ladder was recorded.
_CERTIFIED_PRESSURE_LEVELS_HPA = CERTIFIED_PRESSURE_LEVELS_HPA["GFS"]
#: Kept as the historical spelling: several tests and downstream readers
#: import it by this name, and it means exactly what it always did --
#: the certified 21-level ladder.
_PRESSURE_LEVELS_HPA = _CERTIFIED_PRESSURE_LEVELS_HPA
_THREE_D = ("GHT", "T", "RH", "U", "V")
_TWO_D = (
    "PSFC", "SOURCE_OROGRAPHY", "SKINTEMP", "SNOW", "SNOWH",
    "T2", "RH2", "U10", "V10", "LANDSEA", "XICE",
    "GFS_ST000010", "GFS_ST010040", "GFS_ST040100", "GFS_ST100200",
    "GFS_SM000010", "GFS_SM010040", "GFS_SM040100", "GFS_SM100200",
)
_IMPLEMENTATION_PATHS = (
    "woof/gfs_direct.py",
    "woof/era5_direct.py",
    "woof/native_wrf_contract.py",
    "woof/source_hierarchy.py",
    "woof/native_hierarchy.py",
    "woof/native_domain_artifacts.py",
    "woof/source_cli.py",
    "woof/source_adapters.py",
    "woof/experiment.py",
    "woof/core/grid.py",
    "woof/vertical_adaptation.py",
    "woof/core/noah.py",
    "woof/ingest/grib.py",
    "woof/ingest/horiz.py",
    "woof/ingest/cpu_backend.py",
    "woof/ingest/preprocess_backend.py",
    "woof/ingest/soil.py",
    "woof/ingest/real.py",
    "woof/ingest/lateral_bc.py",
    "woof/ingest/prepared_cache.py",
    "woof/ingest/nest_init.py",
    "woof/static/lambert.py",
    "woof/static/build.py",
    "woof/static/highres_production.py",
    "woof/static/highres.py",
    "woof/static/highres_fetch.py",
    "woof/wrf_direct.py",
    "woof/data/noah_tables/GENPARM.TBL",
    "woof/data/noah_tables/SOILPARM.TBL",
    "woof/data/noah_tables/VEGPARM.TBL",
    "tools/prepare_gfs_cpu_wrf.py",
    "tools/grib1_bridge/Cargo.lock",
    "tools/grib1_bridge/Cargo.toml",
    "tools/grib1_bridge/src/bin/gfs_grib2_bridge.rs",
    "tools/grib1_bridge/vendor/grib-core/src/grib2/mod.rs",
    "tools/grib1_bridge/vendor/grib-core/src/grib2/parser.rs",
    "tools/grib1_bridge/vendor/grib-core/src/grib2/search.rs",
    "tools/grib1_bridge/vendor/grib-core/src/grib2/unpack.rs",
)


#: The GFS adapter's name in every water-temperature refusal and
#: receipt.
_WATER_ROUTE = "the GFS direct route"


def _implementation_sha256() -> dict[str, str]:
    root = Path(__file__).resolve().parent.parent
    result = {
        name: _sha256(root / name)
        for name in _IMPLEMENTATION_PATHS
        if (root / name).is_file()
    }
    missing = [name for name in _IMPLEMENTATION_PATHS
               if not (root / name).is_file()]
    if missing:
        # Installed wheels do not carry the repo-only paths (tools/,
        # Rust workspace sources).  A sealed runtime binds its own
        # distribution manifest here; a plain pip install records the
        # absent inventory accurately instead of demanding one.
        if os.environ.get("WOOF_NATIVE_DISTRIBUTION_MANIFEST"):
            manifest_path, _ = _distribution_manifest()
            result["distribution/manifest.json"] = _sha256(manifest_path)
        result["distribution/repo_only_paths.json"] = hashlib.sha256(
            json.dumps(missing, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
    return result


class GfsRouteError(ValueError):
    """A refusal by this door, with a remedy the reader can act on.

    Subclasses ValueError like every other refusal class in the tree
    (``GoRefusal``, ``HrrrRouteInputError``, ``ManifestError``, ...) so
    ``main``'s handler renders it as one sentence at rc 2.  The point of
    the distinct type is what it is NOT used for: a violated internal
    invariant stays a bare ``RuntimeError`` and keeps its traceback,
    because that is a defect here rather than a problem with the inputs,
    and flattening the two together would report our bug as the reader's
    mistake.
    """


def _distribution_manifest() -> tuple[Path, dict[str, object]]:
    """The sealed manifest, validated against its WHOLE schema."""

    from woof.runtime_manifest import manifest_from_environment

    bound = manifest_from_environment()
    if bound is None:
        raise GfsRouteError(
            "installed GFS runtime requires WOOF_NATIVE_DISTRIBUTION_MANIFEST")
    return bound


def _git_source_identity() -> dict[str, object]:
    root = Path(__file__).resolve().parent.parent

    if os.environ.get("WOOF_NATIVE_DISTRIBUTION_MANIFEST"):
        manifest_path, manifest = _distribution_manifest()
        source = manifest["source"]
        return {
            "available": True,
            "commit": source["commit"],
            "tree": source["tree"],
            "tracked_worktree_clean": True,
            "identity_source": "gpuwm-native-distribution-manifest",
            "distribution_manifest_sha256": _sha256(manifest_path),
        }

    def run(*arguments: str) -> str:
        completed = subprocess.run(
            ["git", "-C", str(root), *arguments], check=True,
            text=True, capture_output=True, timeout=10)
        return completed.stdout.strip()

    # A wheel's version/build hash come from package metadata. A tree id
    # and working-tree cleanliness cannot be inferred from that stamp;
    # the input manifest, decoder and implementation hashes still bind it.
    unavailable = {
        "available": False,
        "commit": None,
        "tree": None,
        "tracked_worktree_clean": None,
        "identity_source": "unavailable-installed-runtime",
    }
    try:
        if (root / ".git").exists():
            top_level = Path(run("rev-parse", "--show-toplevel")).resolve()
            if top_level == root.resolve():
                return {
                    "available": True,
                    "commit": run("rev-parse", "HEAD"),
                    "tree": run("rev-parse", "HEAD^{tree}"),
                    "tracked_worktree_clean": not bool(
                        run("status", "--porcelain", "--untracked-files=no")),
                    "identity_source": "git",
                }
    except (OSError, subprocess.SubprocessError):
        pass
    from woof.build_provenance import package_build_identity
    identity = package_build_identity()
    unavailable.update(commit=identity["commit"], package_provenance=identity)
    if identity["commit"] != "unknown":
        unavailable.update(available=True, identity_source=identity["identity_source"])
    return unavailable


def _geometry_contract(grid, cfg) -> dict[str, object]:
    return native_geometry_contract(grid, cfg)


#: The source top a fetch directory implies when its receipt predates the
#: recorded ladder: the certified ladder is the only thing it can have
#: been, because nothing before this release could fetch another.
_CERTIFIED_SOURCE_TOP_PA = float(_CERTIFIED_PRESSURE_LEVELS_HPA.min() * 100.0)


def _validate_ladder(levels: np.ndarray, context: str) -> np.ndarray:
    return validate_ladder(levels, context, source="GFS")


def _manifest_source_top_pa(manifest: Mapping[str, object]) -> float:
    return manifest_source_top_pa(manifest, source="GFS")


def _decode_levels_above_model_top(manifest, p_top_pa):
    """Keep the receipt's levels and at least one complete GFS level above p_top.

    Older full-file manifests declare a truncated ladder although their GRIB
    contains the extra levels. Request them from the same hash-bound input.
    A genuinely truncated subset is refused by the bridge's inventory check,
    naming the absent level, rather than silently extrapolated.
    """
    from tools.download_gfs_native_subset import levels_for_top

    identity = manifest.get("source", {})
    declared = identity.get("pressure_levels_hpa") if isinstance(identity, dict) else None
    levels = tuple(float(v) for v in (declared or _CERTIFIED_PRESSURE_LEVELS_HPA))
    return levels_for_top(float(p_top_pa), certified=levels)


def _refuse_a_fetch_below_the_model_top(exp, source_top_pressure_pa: float):
    """Refuse a download that stops under the config's model top, with the way out.

    Every level above the source top would be extrapolated past the top
    of the analysis, which is the vertical contract's own refusal.  Said
    here first because on this route the way out is a fetch flag, not the
    eta ladder: ``woof go`` and ``woof run-plan`` ask for the top on
    their own, so only a folder fetched by hand gets here.

    It is the vertical-ladder member of the preparation refusal family,
    so the door prints it as the refusal and its own remedy at the
    family's exit status, the way it printed the vertical contract's
    refusal for the same breakage, rather than as one flat error line.
    """

    from woof.ingest.source_coverage import VerticalLadderRefusal
    from woof.source_adapters import fetch_model_top_pa

    p_top = float(exp.vertical.p_top)
    source_top = float(source_top_pressure_pa)
    if not source_top > p_top:
        return
    asked = fetch_model_top_pa("gfs", p_top)
    if asked is None:
        remedy = (f"remedy: raise [shared].p_top in the experiment config "
                  f"to {source_top:g} Pa or above.")
    else:
        remedy = (f"remedy: woof fetch --source gfs --p-top-pa {asked:g} "
                  "into a new folder fetches the levels this top needs "
                  "(woof go and woof run-plan ask for them on their own), "
                  f"or raise [shared].p_top to {source_top:g} Pa or above.")
    raise VerticalLadderRefusal(
        f"GFS source atmosphere stops at {source_top:g} Pa but this config's "
        f"model top is {p_top:g} Pa, so every level above {source_top:g} Pa "
        "would be extrapolated past the top of the analysis; this folder "
        "was fetched without the levels above it",
        remedy=remedy)


def _validate_grid_and_vertical_contract(exp, wps_namelist: Path, *,
                                         source_top_pressure_pa: float
                                         = _CERTIFIED_SOURCE_TOP_PA):
    return validate_native_lambert_contract(
        exp, wps_namelist, source_name="GFS",
        source_top_pressure_pa=float(source_top_pressure_pa),
    )


def _verify_static_receipt(
    receipt_path: Path,
    static_input: Path,
    grid,
    cfg,
) -> dict[str, object]:
    return verify_native_static_receipt(
        receipt_path, static_input, grid, cfg)


def _source_coverage_receipt(snapshot: Era5Snapshot, grid, lake_mask) -> dict[str, object]:
    ny = snapshot.latitude.size
    nx = snapshot.longitude.size
    result: dict[str, object] = {
        "source_shape": [int(ny), int(nx)],
        "parabolic_required_offsets": [-1, 2],
        "masked_deterministic_donor_offsets": [-1, 2],
        "masked_search_reach":
            "full-cropped-source (WPS FIFO BFS, depth 1200)",
        "staggerings": {},
    }
    coordinates = {
        "mass": grid.latlon_mass(),
        "u": grid.latlon_u(),
        "v": grid.latlon_v(),
    }
    mapped: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for name, (latitude, longitude) in coordinates.items():
        y, x = _regular_coordinates(
            snapshot.latitude, snapshot.longitude, latitude, longitude)
        mapped[name] = (y, x)
        floor_y = np.floor(y).astype(np.int64)
        floor_x = np.floor(x).astype(np.int64)
        donor_bounds = {
            "x": [int(floor_x.min()) - 1, int(floor_x.max()) + 2],
            "y": [int(floor_y.min()) - 1, int(floor_y.max()) + 2],
        }
        if (donor_bounds["x"][0] < 0 or donor_bounds["x"][1] >= nx
                or donor_bounds["y"][0] < 0 or donor_bounds["y"][1] >= ny):
            raise ValueError(
                f"GFS crop lacks full parabolic donor halo for {name}")
        result["staggerings"][name] = {
            "source_x_range": [float(x.min()), float(x.max())],
            "source_y_range": [float(y.min()), float(y.max())],
            "parabolic_donor_x": donor_bounds["x"],
            "parabolic_donor_y": donor_bounds["y"],
        }

    # Masked-chain coverage contract (v2): deterministic operators reach the
    # floor-based [-1, +2] stencil; the WPS FIFO search fallback reaches the
    # whole cropped source, so the crop must retain usable donors of both
    # surface classes instead of certifying a fixed radius.
    mass_y, mass_x = mapped["mass"]
    floor_my = np.floor(mass_y).astype(np.int64)
    floor_mx = np.floor(mass_x).astype(np.int64)
    masked_bounds = {
        "x": [int(floor_mx.min()) - 1, int(floor_mx.max()) + 2],
        "y": [int(floor_my.min()) - 1, int(floor_my.max()) + 2],
    }
    if (masked_bounds["x"][0] < 0 or masked_bounds["x"][1] >= nx
            or masked_bounds["y"][0] < 0 or masked_bounds["y"][1] >= ny):
        raise ValueError(
            "GFS crop lacks the complete masked-surface deterministic "
            "donor halo")
    result["masked_surface_donor_x"] = masked_bounds["x"]
    result["masked_surface_donor_y"] = masked_bounds["y"]
    # Class support is reported, not enforced: single-class crops are
    # legitimate and an absent class also fills on the full WPS grid.
    source_landsea = np.asarray(
        snapshot.fields["LANDSEA"], dtype=np.float64)
    result["masked_class_support"] = {
        "land": int(np.sum(source_landsea >= 0.5)),
        "water": int(np.sum(source_landsea < 0.5)),
    }

    # A census of how far each lake's nearest GFS water lies inside the
    # crop.  It sets no value: lake skin temperature is the masked WPS
    # interpolation, whose whole-source search (gfs_lake_skin = "wps")
    # also stops at the crop's edge.  A lake whose nearer
    # water could lie past the crop's edge is counted here and announced
    # (:func:`_announce_lake_source_water`).
    lakes = np.asarray(lake_mask, dtype=np.bool_)
    source_land = np.asarray(snapshot.fields["LANDSEA"], dtype=np.float64)
    skin = np.asarray(snapshot.fields["SKINTEMP"], dtype=np.float64)
    water = np.isfinite(source_land) & (source_land < 0.5) & np.isfinite(skin)
    (water,), lake_x, period = unrolled_source_ring(
        (water,), snapshot.longitude, mass_x)
    lake_nx = water.shape[1]
    max_nearest_distance = 0.0
    max_search_radius = 0.0
    unproven = 0
    for j, i in (np.argwhere(lakes) if np.any(water) else ()):
        y = float(mass_y[j, i])
        x = float(lake_x[j, i])
        # Initial window only; the loop doubles it until the nearest water
        # donor is provably inside the searched rectangle.
        search_radius = 8.0
        while True:
            j0 = max(0, int(np.ceil(y - search_radius)))
            j1 = min(ny - 1, int(np.floor(y + search_radius)))
            i0 = max(0, int(np.ceil(x - search_radius)))
            i1 = min(lake_nx - 1, int(np.floor(x + search_radius)))
            rows, columns = np.nonzero(water[j0:j1 + 1, i0:i1 + 1])
            if rows.size:
                rows = rows + j0
                columns = columns + i0
                best_squared = float(
                    np.min((rows - y) ** 2 + (columns - x) ** 2))
                unseen_distance = []
                if j0 > 0:
                    unseen_distance.append(y - (j0 - 1))
                if j1 < ny - 1:
                    unseen_distance.append((j1 + 1) - y)
                if i0 > 0:
                    unseen_distance.append(x - (i0 - 1))
                if i1 < lake_nx - 1:
                    unseen_distance.append((i1 + 1) - x)
                if (not unseen_distance
                        or best_squared < min(unseen_distance) ** 2):
                    break
            search_radius *= 2.0
        outside_squared = min(
            (y + 1.0, ny - y) if period is not None
            else (x + 1.0, lake_nx - x, y + 1.0, ny - y)) ** 2
        if not best_squared < outside_squared:
            unproven += 1
        max_nearest_distance = max(max_nearest_distance, best_squared ** 0.5)
        max_search_radius = max(max_search_radius, search_radius)
    result["lake_cells"] = int(lakes.sum())
    result["lake_source_search"] = "global expanding nearest-water (coverage diagnostic only)"
    result["max_lake_source_search_radius_cells"] = max_search_radius
    result["max_lake_source_water_distance_cells"] = max_nearest_distance
    # Present only when they happened, so every other receipt keeps the
    # bytes it had.
    if np.any(lakes) and not np.any(water):
        result["lake_cells_without_source_water"] = int(lakes.sum())
    if unproven:
        result["lake_cells_nearest_water_past_crop"] = unproven
    return result


def _announce_lake_source_water(past_crop, *, lake_cells):
    """One line when a lake's nearest GFS water could lie past the crop.

    A crop that holds no GFS water at all needs no line here: the masked
    chain gives such a lake the skin GFS has there and its repair receipt
    counts it (``other_surface``).  This route's 2.8.x nearest-water lake
    override (``lake_skin_with_source_skin_fallback``) is retired: lakes
    are water targets of the masked chain, and its whole-source search
    takes the nearest GFS water inside the crop.
    """
    if past_crop:
        print(
            f"lake water temperature: of {lake_cells} lake cell(s), "
            f"{past_crop} took the nearest GFS water inside the fetched "
            "area, though nearer GFS water could lie past its edge "
            "(counted under source_coverage in the preparation proof); a "
            "wider fetch area gives every lake its nearest GFS water",
            file=sys.stderr)


def _read_series(path: Path) -> tuple[tuple[int, Path], ...]:
    parent = path.parent
    records: list[tuple[int, Path]] = []
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        columns = raw.split("\t")
        if not 2 <= len(columns) <= 3:
            raise ValueError(
                "GFS series line "
                f"{line_number} must be "
                "HOUR<TAB>GRIB2[<TAB>FORECAST_PROCESS_ID]")
        hour = int(columns[0])
        expected_process_id = int(columns[2]) if len(columns) == 3 else 81
        if expected_process_id not in {81, 96}:
            raise ValueError(
                f"GFS series line {line_number} declares uncertified "
                f"forecast process ID {expected_process_id}")
        if hour == 0 and expected_process_id != 81:
            raise ValueError(
                f"GFS series line {line_number} must declare analysis "
                "process ID 81 for f000")
        source = Path(columns[1])
        if not source.is_absolute():
            source = parent / source
        source = source.resolve()
        if not source.is_file():
            raise FileNotFoundError(f"missing GFS series file: {source}")
        records.append((hour, source))
    hours = [hour for hour, _ in records]
    # The series names the SOURCE leads it carries, and its first entry is
    # not required to be f000.  Initializing a model from a forecast lead
    # is routine practice -- WPS/real does it from any met_em set -- and
    # the f000 anchor here was the reason a user who wanted the f174..f240
    # window had to integrate 240 hours to reach it.  What a series must
    # still be is at least two times, nonnegative, strictly increasing on
    # one positive uniform cadence, inside the published lead ladder: the
    # first is the initial condition and the rest are its boundaries.
    if len(hours) < 2:
        raise ForcingSeriesRefusal(
            "GFS series must have at least two times")
    if hours[0] < 0:
        raise ValueError("GFS series forecast hours must be nonnegative")
    deltas = [later - earlier for earlier, later in zip(hours, hours[1:])]
    if not deltas or any(delta <= 0 or delta != deltas[0] for delta in deltas):
        raise ValueError("GFS series cadence must be positive and uniform")
    from woof.fetch import gfs_forecast_hours
    gfs_forecast_hours(hours[-1] - hours[0], deltas[0], hours[0])
    return tuple(records)


def resolve_initial_forecast_lead(
        *, start_time: datetime, cycle_time: datetime,
        source_hours: Sequence[int]) -> int:
    """The source lead the model starts from, or a precise refusal.

    ``start_time = cycle + K hours`` for any K the series actually
    carries.  K = 0 is the analysis and is what every GFS run did
    before; a positive K initializes from a forecast lead, which is
    ordinary practice (WPS/real does it) and is what the front door used
    to refuse outright with "GFS cycle must equal experiment start_time".
    A user who wanted the f174..f240 window therefore had to integrate
    240 hours to reach it.

    Three things are refused here and nothing else: a start BEFORE the
    cycle (no such product exists), a start that is not a whole number of
    hours after it (the series is hourly), and a lead the fetched set does
    not contain.  The last one names the set, because "which leads do I
    have" is the question a caller who hits it is actually asking.
    """

    offset = start_time - cycle_time
    seconds = offset.total_seconds()
    if seconds < 0:
        raise ValueError(
            f"experiment start_time {start_time:%Y-%m-%d %H:%M:%S} is "
            f"BEFORE GFS cycle {cycle_time:%Y-%m-%d %H:%M:%S}; a model "
            "cannot be initialized from a product its source has not "
            "produced yet")
    if seconds % 3600:
        raise ValueError(
            f"experiment start_time {start_time:%Y-%m-%d %H:%M:%S} is "
            f"{seconds / 3600.0:g} h after GFS cycle "
            f"{cycle_time:%Y-%m-%d %H:%M:%S}; a GFS forecast lead is a "
            "whole number of hours")
    lead = int(seconds // 3600)
    hours = [int(hour) for hour in source_hours]
    if lead not in hours:
        raise ValueError(
            f"experiment start_time {start_time:%Y-%m-%d %H:%M:%S} asks "
            f"GFS cycle {cycle_time:%Y-%m-%d %H:%M:%S} for forecast lead "
            f"f{lead:03d}, which this series does not carry.  The fetched "
            "forecast hours are "
            + ", ".join(f"f{hour:03d}" for hour in hours)
            + ".  Fetch that lead (`woof fetch --source gfs "
            f"--forecast-start-hour {lead}`) or set start_time to a lead "
            "the series has")
    return lead


def initial_condition_provenance(
        *, cycle_time: datetime, lead_hours: int) -> dict[str, object]:
    """What the initial condition IS, said in one sentence and in fields.

    A forecast lead is never relabelled as an analysis: at K = 0 this
    says analysis and names process 81; at K > 0 it says forecast, names
    process 96, and says out loud that the initial condition is itself a
    K-hour forecast.  Every receipt this door writes carries this block,
    so a reader can never recover the cycle without also recovering the
    lead.
    """

    analysis = lead_hours == 0
    start_time = cycle_time + timedelta(hours=lead_hours)
    return {
        "schema": "gpuwm-gfs-initial-condition-provenance-v1",
        "cycle": cycle_time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "initial_forecast_lead_hours": int(lead_hours),
        "model_start_time": start_time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "initial_condition_kind": "analysis" if analysis else "forecast",
        "forecast_generating_process_id": 81 if analysis else 96,
        "statement": (
            f"initialized from GFS cycle "
            f"{cycle_time:%Y-%m-%dT%H:%M:%SZ} analysis (f000)"
            if analysis else
            f"initialized from GFS cycle "
            f"{cycle_time:%Y-%m-%dT%H:%M:%SZ} at lead f{lead_hours:03d}: "
            f"the initial condition is itself a {lead_hours} h forecast"),
    }


def _verify_input_manifest(
    path: Path,
    expected_sha256: str,
    roles: Mapping[str, Path],
) -> dict[str, object]:
    observed_manifest_sha256 = _sha256(path)
    if observed_manifest_sha256 != expected_sha256.lower():
        raise ValueError(
            "GFS input manifest SHA mismatch: expected "
            f"{expected_sha256}, got {observed_manifest_sha256}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema") != INPUT_MANIFEST_SCHEMA:
        raise ValueError("unsupported GFS direct-input manifest schema")
    declared = manifest.get("files")
    if not isinstance(declared, dict) or set(declared) != set(roles):
        raise ValueError("GFS input manifest role inventory differs from request")
    for role, source in roles.items():
        spec = declared[role]
        if not isinstance(spec, dict) or spec.get("name") != source.name:
            raise ValueError(f"GFS input manifest path mismatch for {role}")
    # Each GRIB hashed on host threads, held to its role in order.
    from woof.ingest.host_parallel import map_ordered
    for (role, _), observed in zip(roles.items(), map_ordered(
            _sha256, list(roles.values()))):
        spec = declared[role]
        if spec.get("sha256") != observed:
            raise ValueError(
                f"GFS input digest mismatch for {role}: "
                f"expected {spec.get('sha256')}, got {observed}")
    return manifest


def _parse_gate(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        columns = raw.split("\t", 1)
        if len(columns) != 2 or columns[0] in result:
            raise ValueError("malformed GFS bridge gate")
        result[columns[0]] = columns[1]
    if result.get("status") != "PASS" or result.get("schema") != BRIDGE_SCHEMA:
        raise ValueError("GFS bridge did not publish a recognized PASS gate")
    return result


def _read_f32(path: Path, shape: tuple[int, ...]) -> np.ndarray:
    expected = int(np.prod(shape))
    value = np.fromfile(path, dtype="<f4")
    if value.size != expected:
        raise ValueError(
            f"{path.name} contains {value.size} FP32 values; expected {expected}")
    return value.reshape(shape).astype(np.float64)


def _load_bridge_snapshots(
    root: Path,
    cycle: datetime,
    records: tuple[tuple[int, Path], ...],
    expected_source_sha256: Mapping[int, str] | None = None,
    expected_levels_hpa: tuple[float, ...] | None = None,
    expected_snow_convention: str | None = None,
) -> tuple[Era5Snapshot, ...]:
    gate = _parse_gate(root / "gate.tsv")
    if (expected_snow_convention is not None
            and gate.get("snow_convention") != expected_snow_convention):
        raise ValueError("GFS bridge snow convention differs from preparation policy; rebuild the bridge")
    hours = tuple(int(value) for value in gate["forecast_hours"].split(","))
    if hours != tuple(hour for hour, _ in records):
        raise ValueError("GFS bridge forecast-hour inventory differs from series")
    if gate["cycle"] != cycle.strftime("%Y-%m-%d %H:%M:%S"):
        raise ValueError("GFS bridge cycle differs from requested cycle")
    # The ladder is the bridge's report of what it decoded, not a constant
    # to match: a case whose model top sits above 100 hPa is fetched with
    # extra levels on top, and asserting equality against the certified 21
    # is what made 10000 Pa a silent ceiling on every GFS run.  It still
    # has to BE the certified ladder extended upward -- same rule the
    # bridge applies to the source -- and when the caller knows which
    # ladder was fetched, the two must agree exactly.
    levels_hpa = _validate_ladder(
        np.asarray(
            [float(value) / 100.0
             for value in gate["pressure_levels_pa"].split(",")],
            dtype=np.float64),
        "GFS bridge gate")
    if expected_levels_hpa is not None and not np.array_equal(
            levels_hpa, np.asarray(expected_levels_hpa, dtype=np.float64)):
        raise ValueError(
            "GFS bridge decoded a pressure ladder the input manifest does "
            "not declare")
    expected_identity = {
        "originating_center": "7",
        "master_table_version": "2",
        "local_table_version": "1",
        "forecast_generating_process_ids": ",".join(
            f"{hour}:{81 if hour == 0 else 96}" for hour, _ in records),
    }
    if any(gate.get(key) != value for key, value in expected_identity.items()):
        raise ValueError("GFS bridge product identity differs from contract")
    if gate.get("land_mask_parameter") not in {"0", "218"}:
        raise ValueError("GFS bridge did not record a recognized LAND mask choice")
    if (gate.get("invariant_fields") != "SOURCE_OROGRAPHY,LANDSEA"
            or not gate.get("invariant_fingerprint_fnv1a64")):
        raise ValueError("GFS bridge did not prove invariant terrain/LAND fields")
    try:
        source_sha256 = {
            int(item.split(":", 1)[0]): item.split(":", 1)[1]
            for item in gate["source_sha256"].split(",")
        }
    except (KeyError, ValueError, IndexError) as error:
        raise ValueError("GFS bridge source SHA inventory is malformed") from error
    from woof.ingest.host_parallel import map_ordered
    if expected_source_sha256 is None:
        expected_source_sha256 = dict(zip(
            (hour for hour, _ in records),
            map_ordered(_sha256, [path for _, path in records])))
    if source_sha256 != dict(expected_source_sha256):
        raise ValueError("GFS bridge decoded source bytes differ from input manifest")
    inventory_path = root / "inventory.tsv"
    if _sha256(inventory_path) != gate.get("inventory_sha256"):
        raise ValueError("GFS bridge inventory SHA mismatch")
    decoded_manifest_path = root / "decoded-sha256.tsv"
    if _sha256(decoded_manifest_path) != gate.get("decoded_manifest_sha256"):
        raise ValueError("GFS decoded-output manifest SHA mismatch")
    nx = int(gate["nx"])
    ny = int(gate["ny"])
    if int(gate["num_data_points"]) != nx * ny:
        raise ValueError("GFS bridge grid point count differs from nx*ny")
    dx = float(gate["dx"])
    dy = float(gate["dy"])
    if dx != 0.25 or dy != 0.25:
        raise ValueError("GFS bridge grid is not pgrb2.0p25")
    lat1 = float(gate["lat1"])
    lon1 = float(gate["lon1"])
    if (float(gate["lat2"]) != lat1 + (ny - 1) * dy
            or float(gate["lon2"]) != lon1 + (nx - 1) * dx):
        raise ValueError("GFS bridge grid endpoints differ from declared increments")
    latitude = lat1 + np.arange(ny, dtype=np.float64) * dy
    # Continuous ascending longitude axis anchored at the crop's western
    # edge wrapped into [-180, 180); a crop crossing the antimeridian
    # simply continues past 180 (e.g. 177.0 .. 181.0).  The horizontal
    # interpolator unwraps every target longitude onto the branch
    # nearest the axis midpoint, so no seam rotation is required -- the
    # previous rotate-and-argsort produced a non-uniform axis for
    # antimeridian crops and failed downstream.
    lon_start = (lon1 + 180.0) % 360.0 - 180.0
    longitude = lon_start + np.arange(nx, dtype=np.float64) * dx
    if not np.all(np.diff(latitude) > 0.0) or not np.all(np.diff(longitude) > 0.0):
        raise ValueError("GFS regular-grid axes are not strictly increasing")

    decoded_rows: dict[tuple[int, str], tuple[int, str, str]] = {}
    lines = decoded_manifest_path.read_text(encoding="utf-8").splitlines()
    if not lines or lines[0] != "hour\tvariable\tbytes\tsha256\tfilename":
        raise ValueError("GFS decoded-output manifest header is malformed")
    for raw in lines[1:]:
        columns = raw.split("\t")
        if len(columns) != 5:
            raise ValueError("GFS decoded-output manifest row is malformed")
        hour = int(columns[0])
        name = columns[1]
        key = (hour, name)
        if key in decoded_rows:
            raise ValueError("GFS decoded-output manifest contains a duplicate")
        decoded_rows[key] = (int(columns[2]), columns[3], columns[4])
    expected_decoded = {
        (hour, name) for hour, _ in records for name in (*_THREE_D, *_TWO_D)}
    if set(decoded_rows) != expected_decoded:
        raise ValueError("GFS decoded-output manifest inventory differs from contract")
    checked_rows = []
    for (hour, name), (declared_bytes, declared_sha256, declared_name) in decoded_rows.items():
        relative = f"f{hour:03d}/{name}.f32le"
        path = root / relative
        if declared_name != relative or not path.is_file():
            raise ValueError("GFS decoded-output manifest path differs from contract")
        checked_rows.append((relative, path, declared_bytes, declared_sha256))
    # Every decoded array hashed on host threads (one file per item), then
    # held to its row in manifest order: the first mismatch named is the
    # one the serial walk named.
    observed = map_ordered(
        lambda row: (row[1].stat().st_size == row[2]
                     and _sha256(row[1]) == row[3]), checked_rows)
    for (relative, *_), matches in zip(checked_rows, observed):
        if not matches:
            raise ValueError(f"GFS decoded array identity mismatch for {relative}")

    # Read when asked, not here.  Every decoded array was verified above;
    # the leads are loaded one at a time by whoever indexes them, so the
    # preparation holds a window of leads instead of the whole series.
    return _BridgeSnapshots(
        root=root, cycle=cycle, hours=tuple(hour for hour, _ in records),
        levels_hpa=levels_hpa, latitude=latitude, longitude=longitude,
        ny=ny, nx=nx)


def _bridge_snapshot(root: Path, cycle: datetime, hour: int, levels_hpa,
                     latitude, longitude, ny: int, nx: int,
                     *, with_fields: bool = True) -> Era5Snapshot:
    """One decoded lead read off the bridge's verified arrays.

    ``with_fields=False`` is the lead's grid alone (axes, valid time), for
    a caller that asks where a lead is without needing its values.
    """

    fields: dict[str, np.ndarray] = {}
    if with_fields:
        # Every field is its own file: read and widen them on host threads
        # (woof.ingest.host_parallel), in the serial order, so each array
        # is the one the serial loop read.  Serially this was a single-core
        # pass over every decoded lead, taken twice per lead.
        from woof.ingest.host_parallel import map_ordered

        time_root = Path(root) / f"f{hour:03d}"
        reads = ([(name, (levels_hpa.size, ny, nx)) for name in _THREE_D]
                 + [(name, (ny, nx)) for name in _TWO_D])
        arrays = map_ordered(
            lambda item: _read_f32(time_root / f"{item[0]}.f32le", item[1]),
            reads)
        for (name, _), array in zip(reads, arrays):
            fields[name] = array
    return Era5Snapshot(
        valid_time=cycle + timedelta(hours=hour),
        levels_hpa=levels_hpa,
        latitude=latitude,
        longitude=longitude,
        fields=fields,
    )


#: Decoded leads one lazy series keeps after they were read: the build of a
#: forcing time and the boundary interval it closes ask for the same two.
_SNAPSHOT_CACHE_LEADS = 2


class _BridgeSnapshots(_ABCSequence):
    """A bridge decode's leads, read from its verified arrays on access.

    Held whole, a 240 h whole-globe GFS series is 81 float64 leads of about
    1.1 GB each, which is what killed the 64 GiB prepare of 2026-10-05T00
    (:mod:`woof.ingest.host_decode_window`).  This sequence keeps the last
    :data:`_SNAPSHOT_CACHE_LEADS` leads it read and reads any other again
    from the same verified files, so every lead it returns holds exactly
    the values the eager load held.  ``transform`` (the longitude
    re-cut) is applied on every read.  A slice is another lazy series.
    """

    def __init__(self, *, root, cycle, hours, levels_hpa, latitude,
                 longitude, ny, nx, transform=None):
        self.root = Path(root)
        self.cycle = cycle
        self.hours = tuple(int(hour) for hour in hours)
        self.levels_hpa = levels_hpa
        self.latitude = latitude
        self.longitude = longitude
        self.ny = int(ny)
        self.nx = int(nx)
        self.transform = transform
        self._cache: dict[int, Era5Snapshot] = {}

    def _derived(self, hours, transform):
        return _BridgeSnapshots(
            root=self.root, cycle=self.cycle, hours=hours,
            levels_hpa=self.levels_hpa, latitude=self.latitude,
            longitude=self.longitude, ny=self.ny, nx=self.nx,
            transform=transform)

    def mapped(self, transform) -> "_BridgeSnapshots":
        """The same leads with ``transform`` applied after any earlier one."""

        earlier = self.transform
        if earlier is None:
            combined = transform
        else:
            def combined(snapshot):
                return transform(earlier(snapshot))
        return self._derived(self.hours, combined)

    @property
    def valid_times(self) -> tuple:
        return tuple(self.cycle + timedelta(hours=hour) for hour in self.hours)

    def metadata(self, index: int) -> Era5Snapshot:
        """The lead's grid, after ``transform``, with no field read."""

        snapshot = _bridge_snapshot(
            self.root, self.cycle, self.hours[int(index)], self.levels_hpa,
            self.latitude, self.longitude, self.ny, self.nx,
            with_fields=False)
        return snapshot if self.transform is None else self.transform(snapshot)

    def snapshot_metadata(self, index: int):
        """:func:`woof.ingest.source_metadata.snapshot_metadata`, unread."""

        from woof.ingest.source_metadata import SourceSnapshotMetadata

        index = int(index)
        if index < 0:
            index += len(self.hours)
        if not 0 <= index < len(self.hours):
            raise IndexError(index)
        snapshot = self.metadata(index)
        return SourceSnapshotMetadata(
            type(snapshot), getattr(snapshot, "latitude", ()),
            getattr(snapshot, "longitude", ()),
            getattr(snapshot, "projection", None))

    def __len__(self) -> int:
        return len(self.hours)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return self._derived(self.hours[index], self.transform)
        index = int(index)
        if index < 0:
            index += len(self.hours)
        if not 0 <= index < len(self.hours):
            raise IndexError(index)
        cached = self._cache.get(index)
        if cached is not None:
            self._cache[index] = self._cache.pop(index)
            return cached
        snapshot = _bridge_snapshot(
            self.root, self.cycle, self.hours[index], self.levels_hpa,
            self.latitude, self.longitude, self.ny, self.nx)
        if self.transform is not None:
            snapshot = self.transform(snapshot)
        self._cache[index] = snapshot
        while len(self._cache) > _SNAPSHOT_CACHE_LEADS:
            self._cache.pop(next(iter(self._cache)))
        return snapshot

    def __iter__(self):
        for index in range(len(self.hours)):
            yield self[index]


def _map_snapshots(snapshots, transform):
    """``transform`` over a snapshot sequence, lazily when it is lazy."""

    mapped = getattr(snapshots, "mapped", None)
    if mapped is not None:
        return mapped(transform)
    return tuple(transform(snapshot) for snapshot in snapshots)


def _sequence_valid_times(snapshots) -> tuple:
    """Every snapshot's valid time, without reading a lazy series' fields."""

    times = getattr(snapshots, "valid_times", None)
    if times is not None:
        return tuple(times)
    return tuple(snapshot.valid_time for snapshot in snapshots)


#: An as-posted preparation's manifest roles: every lead's payload is
#: ``grib-fNNN``, and the series file lists every lead's object, so its
#: digest follows the lead rows (``boundary_stream.input_plan``).
_AS_POSTED_LEAD_ROLE_PREFIX = "grib-f"
_AS_POSTED_DERIVED_ROLES = ("series",)
#: The proof keys that read every lead, so an as-posted head leaves them
#: out and its seal writes them: the manifest's digest and rows, each
#: lead's source coverage, and the decoder's report on the whole series.
_AS_POSTED_SEAL_KEYS = ("decoder_stdout", "input_manifest_sha256",
                        "source_coverage", "source_inputs")
#: The same keys of a domain tree's proof, which carries no source_inputs.
_AS_POSTED_TREE_SEAL_KEYS = ("decoder_stdout", "input_manifest_sha256",
                             "source_coverage")


def _series_row(hour: int, name: str) -> str:
    """One series line, as the fetch writes it (fetch.py publish_manifest)."""

    return f"{hour}\t{name}\t{81 if hour == 0 else 96}\n"


def _as_posted_plan(*, posting: Path, series: Path, cycle_time: datetime,
                    roles: Mapping[str, Path], captured_decoder=None):
    """The input plan of an as-posted GFS preparation, from its first lead.

    ``posting`` is the as-posted fetch's ``posting/`` folder beside
    ``series`` (``<out>/gfs-series.tsv``, which grows with the fetch).  The
    schedule names every lead of the window.  A lead's object is named
    as the first lead's is, with its own lead in place of the first
    lead's (the fetch names a lead's object from its lead alone); the
    seal holds every lead's posted object to this plan and refuses one
    that differs.  Returns ``(posted, records, manifest)``: the lead wait,
    the series records the preparation reads, and the manifest the seal
    will write with every lead's payload digest (and the series digest,
    which follows them) not yet known.

    A physical member may supply the decoder record from its pinned ordinary
    source head. It still binds current configuration bytes and the complete
    posting schedule, then checks the resulting plan against that source.
    """

    from woof.ingest.boundary_stream import (
        POSTING_SCHEDULE_NAME, PostedLeads, read_replaced_json,
    )

    out = Path(series).parent
    schedule_path = Path(posting) / POSTING_SCHEDULE_NAME
    try:
        # The fetch replaces the schedule as leads move, and on Windows a
        # read that meets the replace fails for a moment.
        schedule = read_replaced_json(schedule_path)
    except (OSError, ValueError) as error:
        raise ValueError(
            f"an as-posted GFS preparation reads the fetch's posting "
            f"schedule, and {schedule_path} is not readable: {error}") from None
    cycle_label = cycle_time.strftime("%Y-%m-%dT%H")
    if (schedule.get("source") != "gfs"
            or str(schedule.get("cycle", ""))[:13] != cycle_label):
        raise ValueError(
            f"{schedule_path} schedules {schedule.get('source')} "
            f"{schedule.get('cycle')}, not the gfs {cycle_label} cycle this "
            "preparation reads")
    hours = [int(row["lead"]) for row in schedule.get("leads") or ()]
    if len(hours) < 2:
        raise ForcingSeriesRefusal("GFS series must have at least two times")
    posted = PostedLeads(Path(posting), source="gfs",
                         cycle=str(schedule["cycle"]))
    first = posted.wait(hours[0])
    objects = [item for item in first.get("objects") or ()
               if str(item.get("role", "")).startswith("gfs-")]
    if len(objects) != 1:
        raise ValueError(
            f"the posted marker of f{hours[0]:03d} names {len(objects)} GFS "
            "payload objects; a GFS lead is one object")
    token = f"f{hours[0]:03d}"
    first_name = str(objects[0]["name"])
    if first_name.count(token) != 1:
        raise ValueError(
            f"the posted GFS object {first_name} does not name its lead "
            f"{token} once, so the other leads' objects cannot be planned")
    records = tuple(
        (hour, (out / first_name.replace(token, f"f{hour:03d}")).resolve())
        for hour in hours)
    # Republished by the fetch with each verified prefix: read through
    # the replace, as the schedule is.
    fetched = read_replaced_json(out / "fetch-manifest.json")
    identity = {
        "model": "GFS",
        "product": "pgrb2.0p25",
        "cycle": cycle_time.strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    levels = fetched.get("pressure_levels_hpa")
    if isinstance(levels, list) and levels and all(
            isinstance(level, (int, float)) for level in levels):
        identity["pressure_levels_hpa"] = [float(level) for level in levels]
        identity["top_pressure_pa"] = float(min(levels)) * 100.0
    files = {
        role: dict(captured_decoder) if role == "bridge" and captured_decoder is not None else {
               "name": path.name,
               "sha256": None if role in _AS_POSTED_DERIVED_ROLES
               else _sha256(path)}
        for role, path in roles.items()
    }
    for hour, path in records:
        files[f"{_AS_POSTED_LEAD_ROLE_PREFIX}{hour:03d}"] = {
            "name": path.name, "sha256": None}
    manifest = {"schema": INPUT_MANIFEST_SCHEMA, "source": identity,
                "files": files}
    return posted, records, manifest


class _DecodedLeads(Mapping):
    """Decoded leads by forecast hour, each read from its batch on access.

    ``bind`` records where a lead lives (a lazy :class:`_BridgeSnapshots`
    and its position, or any sequence of snapshots); indexing reads it.
    Holding the leads themselves is what put all 81 whole-globe leads of
    a 240 h GFS window in one process (:mod:`woof.ingest.host_decode_window`).
    """

    def __init__(self):
        self._where: dict[int, tuple] = {}

    def bind(self, hour: int, series, position: int) -> None:
        self._where[int(hour)] = (series, int(position))

    def __getitem__(self, hour):
        series, position = self._where[int(hour)]
        return series[position]

    def metadata(self, hour):
        """The lead's grid, without reading its fields when it is lazy."""

        series, position = self._where[int(hour)]
        read = getattr(series, "metadata", None)
        return series[position] if read is None else read(position)

    def __iter__(self):
        return iter(self._where)

    def __len__(self) -> int:
        return len(self._where)

    def __contains__(self, hour) -> bool:
        try:
            return int(hour) in self._where
        except (TypeError, ValueError):
            return False


def _file_bytes(path) -> int:
    try:
        return int(Path(path).stat().st_size)
    except OSError:
        return 0


def _decoded_lead_bytes(decoded: Path) -> int | None:
    """Float32 bytes of the largest lead a bridge output wrote, or ``None``."""

    try:
        lines = (Path(decoded) / "decoded-sha256.tsv").read_text(
            encoding="utf-8").splitlines()[1:]
    except OSError:
        return None
    per_hour: dict[str, int] = {}
    for raw in lines:
        columns = raw.split("\t")
        if len(columns) == 5:
            per_hour[columns[0]] = per_hour.get(columns[0], 0) + int(columns[2])
    return max(per_hour.values(), default=None)


def _first_decoded_lead_estimate(lead_bytes: int,
                                 levels: int | None = None) -> int:
    """A lead's decoded float32 bytes before any lead has been decoded.

    The whole-globe decode of the fields this route reads (``levels``
    pressure levels, the certified ladder when unknown), capped at four
    times the object: a full-file lead decodes to about its own size
    (0.56 GB of arrays from a 0.54 GB object at 23 levels), and a NOMADS
    crop's simple packing to a few times its size.
    """

    from woof.ingest.host_decode_window import gfs_decoded_lead_bytes

    if levels is None:
        levels = int(_CERTIFIED_PRESSURE_LEVELS_HPA.size)
    whole_globe = gfs_decoded_lead_bytes(
        len(_THREE_D) * int(levels) + len(_TWO_D))
    return int(min(whole_globe, 4 * max(1, int(lead_bytes))))


def _bridge_thread_request(environment) -> int:
    """Threads the bridge would take: an explicit count, else every core."""

    from woof.ingest.host_decode_window import threads_available

    value = (os.environ if environment is None else environment).get(
        "GPUWM_GFS_BRIDGE_THREADS")
    try:
        requested = int(str(value).strip()) if value is not None else 0
    except ValueError:
        requested = 0
    available = threads_available()
    return min(requested, available) if requested > 0 else available


def _say_window(window, *, total: int) -> None:
    print(f"prepare: GFS {window.sentence(total)}", file=sys.stderr,
          flush=True)


def _series_lines(series: Path) -> list[list[str]]:
    """The series file's lead rows, columns as written, paths made absolute."""

    rows = []
    for raw in Path(series).read_text(encoding="utf-8").splitlines():
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        columns = raw.split("\t")
        source = Path(columns[1])
        if not source.is_absolute():
            source = Path(series).parent / source
        columns[1] = str(source.resolve())
        rows.append(columns)
    return rows


def _decode_series_bounded(command, *, series: Path, records, decoded: Path,
                           scratch: Path, bridge: Path, cycle_time,
                           levels_pa_csv: str | None, environment):
    """Run the bridge over a whole series within this host's RAM.

    One bridge run parses every lead's object into memory before decoding
    any of them, so a 240 h whole-globe series is 81 parsed objects in one
    process (about 44 GB): the 64 GiB prepare kill of 2026-10-05T00
    (:mod:`woof.ingest.host_decode_window`).  When the series fits the
    window this is that one run, on the window's thread count.  When it
    does not, the leads are decoded in windows with ``--lead-batch``,
    ``--merge-batches`` writes the series' receipts into ``decoded`` (byte
    for byte the one decode's, the as-posted seal's own merge) and every
    lead's arrays are moved in beside them, so ``decoded`` is the tree the
    one run writes.  Returns a ``CompletedProcess`` as the one run does.
    """

    from woof.ingest.host_decode_window import (
        available_host_bytes, decode_window)

    lead_bytes = max(_file_bytes(path) for _, path in records)
    levels = None if levels_pa_csv is None else len(levels_pa_csv.split(","))
    window = decode_window(
        leads=len(records), threads=_bridge_thread_request(environment),
        lead_bytes=lead_bytes,
        decoded_lead_bytes=_first_decoded_lead_estimate(lead_bytes, levels),
        available=available_host_bytes())
    _say_window(window, total=len(records))

    def run(argv, threads):
        env = environment
        if window.available_bytes is not None:
            env = {**(os.environ if env is None else env),
                   "GPUWM_GFS_BRIDGE_THREADS": str(int(threads))}
        return subprocess.run(argv, check=False, text=True,
                              capture_output=True, env=env)

    if window.leads >= len(records):
        return run(command, window.threads)
    rows = _series_lines(series)
    if [int(columns[0]) for columns in rows] != [hour for hour, _ in records]:
        raise ValueError(f"{series} rows differ from the series read")
    batches: list[Path] = []
    start = 0
    ladder = levels_pa_csv
    decoded_lead = None
    while start < len(rows):
        if batches:
            window = decode_window(
                leads=len(rows) - start,
                threads=_bridge_thread_request(environment),
                lead_bytes=max(_file_bytes(path)
                               for _, path in records[start:]),
                decoded_lead_bytes=(decoded_lead or
                                    _first_decoded_lead_estimate(lead_bytes, levels)),
                available=available_host_bytes())
            _say_window(window, total=len(records))
        chunk = rows[start:start + window.leads]
        index = len(batches)
        table = Path(scratch) / f"series-batch-{index:03d}.tsv"
        table.write_text("".join("\t".join(columns) + "\n"
                                 for columns in chunk), encoding="utf-8")
        output = Path(scratch) / f"series-batch-{index:03d}"
        argv = [str(bridge), "--series", str(table), str(output),
                cycle_time.strftime("%Y-%m-%d %H:%M:%S")]
        if ladder is not None:
            argv += ["--pressure-levels-pa", ladder]
        argv.append("--lead-batch")
        completed = run(argv, window.threads)
        if completed.returncode != 0:
            return completed
        if ladder is None:
            # Every later batch decodes on the first batch's ladder, as the
            # one run derives its ladder from its first hour.
            ladder = _parse_gate(output / "gate.tsv")["pressure_levels_pa"]
        if decoded_lead is None:
            decoded_lead = _decoded_lead_bytes(output)
        batches.append(output)
        start += len(chunk)
    merged = subprocess.run(
        [str(bridge), "--merge-batches", str(decoded),
         cycle_time.strftime("%Y-%m-%d %H:%M:%S"),
         *(str(path) for path in batches)],
        check=False, text=True, capture_output=True, env=environment)
    if merged.returncode != 0:
        return merged
    for batch in batches:
        for hour_dir in sorted(batch.glob("f[0-9][0-9][0-9]")):
            hour_dir.rename(Path(decoded) / hour_dir.name)
        shutil.rmtree(batch, ignore_errors=True)
    return merged


class _PostedGfsSeries:
    """A GFS series decoded lead batch by lead batch as its leads post.

    Indexing a lead waits for its marker (``PostedLeads``), then decodes
    every lead whose marker is there and is not decoded yet in one bridge
    run (``--lead-batch``, on all cores), so a late launch or a past cycle
    decodes its backlog at once.  Each lead's payload is held to its
    marker's digest by the bridge's own source digests.  At the seal
    :meth:`merge` writes the whole series' decoder receipts from the
    batches (``--merge-batches``), byte for byte the one decode's.
    """

    def __init__(self, *, posted, records, cycle_time, bridge, scratch,
                 levels_pa_csv, environment, orient):
        self.posted = posted
        self.records = tuple(records)
        self.cycle_time = cycle_time
        self.bridge = Path(bridge)
        self.scratch = Path(scratch)
        self.levels_pa_csv = levels_pa_csv
        self.environment = environment
        self.orient = orient
        #: Every decoded lead, read from its batch's arrays when asked
        #: (:class:`_DecodedLeads`), never all held at once.
        self.snapshots = _DecodedLeads()
        #: The host-RAM window each batch was decoded in, in batch order.
        self.windows: list = []
        #: Float32 bytes one decoded lead wrote, measured off the first
        #: batch; ``None`` until a batch has been decoded.
        self._decoded_lead_bytes: int | None = None
        self.markers: dict[int, dict] = {}
        self.batches: list[Path] = []
        self.batch_leads: list[tuple[int, ...]] = []
        self.decode_seconds = 0.0
        self._next = 0
        self._first_grid = None

    def __len__(self) -> int:
        return len(self.records)

    def marker_objects(self, hour: int) -> dict:
        objects = [item for item in self.markers[hour].get("objects") or ()
                   if str(item.get("role", "")).startswith("gfs-")]
        return {str(item["name"]): str(item["sha256"]) for item in objects}

    def _decode_batch(self, start: int, end: int,
                      threads: int | None = None) -> None:
        batch = self.records[start:end + 1]
        expected = {}
        for hour, path in batch:
            objects = self.marker_objects(hour)
            if list(objects) != [path.name]:
                raise ValueError(
                    f"the posted object of f{hour:03d} is "
                    f"{', '.join(objects) or 'nothing'}, not the planned "
                    f"{path.name}; the input plan this preparation's head "
                    "bound names every lead's object")
            expected[hour] = objects[path.name]
        started = time.perf_counter()
        index = len(self.batches)
        table = self.scratch / f"batch-{index:03d}.tsv"
        table.write_text("".join(
            f"{hour}\t{path}\t{81 if hour == 0 else 96}\n"
            for hour, path in batch), encoding="utf-8")
        decoded = self.scratch / f"batch-{index:03d}"
        command = [str(self.bridge), "--series", str(table), str(decoded),
                   self.cycle_time.strftime("%Y-%m-%d %H:%M:%S")]
        if self.levels_pa_csv is not None:
            command += ["--pressure-levels-pa", self.levels_pa_csv]
        command.append("--lead-batch")
        environment = self.environment
        if threads is not None:
            environment = {**(os.environ if environment is None
                              else environment),
                           "GPUWM_GFS_BRIDGE_THREADS": str(int(threads))}
        completed = subprocess.run(command, check=False, text=True,
                                   capture_output=True, env=environment)
        if completed.returncode != 0:
            raise GfsRouteError(decode_failure_message(
                "GFS Rust bridge", completed.stderr))
        levels = None
        if self.levels_pa_csv is not None:
            levels = tuple(float(value) / 100.0
                           for value in self.levels_pa_csv.split(","))
        loaded = _load_bridge_snapshots(
            decoded, self.cycle_time, batch, expected,
            expected_levels_hpa=levels,
            expected_snow_convention=(self.environment or {}).get("GPUWM_GFS_SNOW_CONVENTION"))
        if self.levels_pa_csv is None:
            # Every later batch decodes on the first batch's ladder, as the
            # one decode of the series derives its ladder from its first
            # hour; the merge holds every batch's ladder equal.
            self.levels_pa_csv = _parse_gate(decoded / "gate.tsv")[
                "pressure_levels_pa"]
        oriented = _map_snapshots(loaded, self.orient)
        for position, (hour, _) in enumerate(batch):
            self.snapshots.bind(hour, oriented, position)
            self._require_first_grid(hour, self.snapshots.metadata(hour))
        if self._decoded_lead_bytes is None:
            self._decoded_lead_bytes = _decoded_lead_bytes(decoded)
        self.batches.append(decoded)
        self.batch_leads.append(tuple(hour for hour, _ in batch))
        self.decode_seconds += time.perf_counter() - started

    def _require_first_grid(self, hour: int, snapshot) -> None:
        """Hold a later lead to the first decoded lead's source grid.

        A domain tree's head certifies its donor coverage
        (``source_hierarchy._spatial_coverage_receipt``) before its later
        leads are posted, on the planned series (every lead the same
        product and crop, :meth:`_PostedGfsView.snapshot_metadata`).  The
        breakage this prevents: a lead decoded on another grid would be
        interpolated under a coverage receipt that was never true of it.
        The words are the one-shot receipt's own.
        """

        if self._first_grid is None:
            self._first_grid = snapshot
            return
        first = self._first_grid
        if (not np.array_equal(
                    np.asarray(getattr(snapshot, "latitude", ())),
                    np.asarray(getattr(first, "latitude", ())))
                or not np.array_equal(
                    np.asarray(getattr(snapshot, "longitude", ())),
                    np.asarray(getattr(first, "longitude", ())))
                or getattr(snapshot, "projection", None)
                != getattr(first, "projection", None)):
            raise ValueError(
                f"GFS hierarchy source grid changes between forcing times "
                f"(f{hour:03d})")

    def through(self, position: int) -> None:
        """Decode every lead through ``records[position]``, waiting as they post."""

        while self._next <= position:
            hour = self.records[self._next][0]
            self.markers[hour] = self.posted.wait(hour)
            end = self._next
            while end + 1 < len(self.records):
                later = self.records[end + 1][0]
                record = self.posted.marker(later)
                if record is None:
                    break
                self.markers[later] = record
                end += 1
            # Bounded by host RAM, whatever posted at once: a whole cycle
            # ready in one second used to be one bridge run over every lead.
            window = self._window(self._next, end)
            end = self._next + window.leads - 1
            self.windows.append(window)
            _say_window(window, total=len(self.records))
            self._decode_batch(self._next, end, threads=window.threads)
            self._next = end + 1

    def _window(self, start: int, end: int):
        """The lead batch from ``start`` that fits this host's RAM now."""

        from woof.ingest.host_decode_window import (
            available_host_bytes, decode_window)

        candidates = self.records[start:end + 1]
        lead_bytes = max(_file_bytes(path) for _, path in candidates)
        decoded = self._decoded_lead_bytes
        if decoded is None:
            decoded = _first_decoded_lead_estimate(
                lead_bytes, None if self.levels_pa_csv is None
                else len(self.levels_pa_csv.split(",")))
        return decode_window(
            leads=len(candidates), threads=_bridge_thread_request(self.environment),
            lead_bytes=lead_bytes, decoded_lead_bytes=decoded,
            available=available_host_bytes())

    def snapshot(self, position: int) -> Era5Snapshot:
        self.through(position)
        return self.snapshots[self.records[position][0]]

    def view(self, first: int) -> "_PostedGfsView":
        return _PostedGfsView(self, first)

    def merge(self, output: Path) -> str:
        """Write the whole series' decoder receipts; the bridge's report line."""

        self.through(len(self.records) - 1)
        command = [str(self.bridge), "--merge-batches", str(output),
                   self.cycle_time.strftime("%Y-%m-%d %H:%M:%S"),
                   *(str(path) for path in self.batches)]
        completed = subprocess.run(command, check=False, text=True,
                                   capture_output=True, env=self.environment)
        if completed.returncode != 0:
            raise GfsRouteError(decode_failure_message(
                "GFS Rust bridge", completed.stderr))
        return completed.stdout.strip()


class _PostedGfsView(_ABCSequence):
    """The run's forcing times of a :class:`_PostedGfsSeries`, from its start lead.

    A sequence: ``len`` is the planned count and indexing a time waits for
    its lead, so the build of forcing time k is where the preparation
    waits for lead k (DESIGN A136 2.4 item 2).  The preparation price
    reads only the first time and the count.

    A domain tree's head validates the whole series and prepares each
    child from the snapshot at its start time before the later leads are
    posted, so the view also declares what the plan fixes without waiting:
    ``valid_times`` (the cycle plus each planned lead, the valid time the
    bridge holds each decoded lead to) and :meth:`snapshot_metadata`.  A
    delayed nest's start lead is then the only later lead its head waits
    for (DESIGN A136 section 1).
    """

    def __init__(self, series: _PostedGfsSeries, first: int):
        self.series = series
        self.first = int(first)

    @property
    def valid_times(self) -> tuple:
        cycle = self.series.cycle_time
        return tuple(cycle + timedelta(hours=hour)
                     for hour, _ in self.series.records[self.first:])

    def snapshot_metadata(self, index: int):
        """A time's source grid, without waiting for a lead not posted yet.

        A decoded lead answers for itself; a lead not decoded yet answers
        with the first decoded lead's grid, which every lead of the plan
        shares (one product, one crop) and which each lead is held to as
        it is decoded (:meth:`_PostedGfsSeries._require_first_grid`).
        """

        from woof.ingest.source_metadata import SourceSnapshotMetadata

        index = int(index)
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        hour = self.series.records[self.first + index][0]
        snapshot = (self.series.snapshots.metadata(hour)
                    if hour in self.series.snapshots else None)
        if snapshot is None:
            if self.series._first_grid is None:
                self.series.through(self.first)
            snapshot = self.series._first_grid
        return SourceSnapshotMetadata(
            type(snapshot), getattr(snapshot, "latitude", ()),
            getattr(snapshot, "longitude", ()),
            getattr(snapshot, "projection", None))

    def __len__(self) -> int:
        return len(self.series) - self.first

    def __getitem__(self, index):
        if isinstance(index, slice):
            start, stop, step = index.indices(len(self))
            if step == 1 and stop == len(self):
                # A tail is another view: listing it would read and hold
                # every one of its leads at once.
                return _PostedGfsView(self.series, self.first + start)
            return [self[i] for i in range(start, stop, step)]
        index = int(index)
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        return self.series.snapshot(self.first + index)

    def __iter__(self):
        for index in range(len(self)):
            yield self[index]


def _seal_as_posted_inputs(series_decode, *, plan, series: Path, bridge: Path,
                           wps_namelist: Path, experiment_config: Path,
                           static_input, static_receipt,
                           input_manifest: Path, tree: Path, merged: Path,
                           portable: str) -> dict:
    """What an as-posted GFS seal writes from the lead markers (DESIGN A136 2.4 item 6).

    The whole series' decoder receipts from the lead batches, and the
    input manifest through the one function the one-shot manifest stage
    uses (``fetch.author_gfs_front_door_manifest``), so both are the
    one-shot bytes.  The manifest is refused unless it is the plan the
    head bound, every lead row is the object its posted marker named,
    and the series it binds lists exactly those rows.
    """

    from woof import fetch as fetch_module
    from woof.ingest.boundary_stream import input_plan

    decoder_stdout = series_decode.merge(merged)
    for name, target in (("gate.tsv", "decoder-gate.tsv"),
                         ("inventory.tsv", "decoder-inventory.tsv"),
                         ("decoded-sha256.tsv", "decoder-sha256.tsv")):
        shutil.copy2(merged / name, Path(tree) / target)
    path, digest = fetch_module.author_gfs_front_door_manifest(
        out=Path(series).parent, bridge=bridge, wps_namelist=wps_namelist,
        experiment_config=experiment_config, static_input=static_input,
        static_receipt=static_receipt, manifest_out=input_manifest,
        progress=lambda line: None)
    manifest_bytes = Path(path).read_bytes()
    if hashlib.sha256(manifest_bytes).hexdigest() != digest:
        raise ValueError(
            f"the input manifest at {path} changed as the seal wrote it")
    manifest = json.loads(manifest_bytes.decode("utf-8"))
    # Re-read at the seal: leads fetched under another route table than
    # the one the head planned with are another plan.
    route_table_sha256 = series_decode.posted.route_table_sha256()
    if input_plan(manifest, lead_role_prefix=_AS_POSTED_LEAD_ROLE_PREFIX,
                  route_table_sha256=route_table_sha256,
                  derived_roles=_AS_POSTED_DERIVED_ROLES) != plan:
        raise ValueError(
            "the input manifest the seal wrote is not the input plan the "
            "head bound: a lead, an object name, another input or the "
            "fetch's route table differs")
    rows = []
    for hour, planned in series_decode.records:
        spec = manifest["files"][f"{_AS_POSTED_LEAD_ROLE_PREFIX}{hour:03d}"]
        objects = series_decode.marker_objects(hour)
        if objects.get(spec["name"]) != spec["sha256"]:
            raise ValueError(
                f"the input manifest binds {spec['name']} ({spec['sha256']}) "
                f"for f{hour:03d}, not the object its posted marker named")
        rows.append(_series_row(hour, spec["name"]))
    written = Path(series).read_bytes()
    if (written != "".join(rows).encode("utf-8")
            or hashlib.sha256(written).hexdigest()
            != manifest["files"]["series"]["sha256"]):
        raise ValueError(
            f"{series} does not list exactly the posted leads the input "
            "manifest binds")
    Path(tree, portable).write_bytes(manifest_bytes)
    return {"manifest": manifest, "manifest_sha256": digest,
            "decoder_stdout": decoder_stdout,
            "route_table_sha256": route_table_sha256}


def prepare_progress_path(output_root) -> Path:
    """Where this stage publishes what it is doing, while it does it.

    A SIBLING of ``--output-root``, not a file inside it, and that is
    forced rather than chosen: the stage builds its whole product in a
    staging directory and publishes it with one ``os.replace``, so
    ``--output-root`` does not exist until the work is finished.  A
    progress file inside it could only appear after there was nothing
    left to report.

    One function, called by the publisher and by ``woof go``'s
    heartbeat, so the path cannot be spelled two ways.  It survives the
    run on purpose: the finished file is the stage's phase timeline, and
    the same thing that answered "what is it doing" answers "what did it
    spend its time on".
    """

    root = Path(output_root)
    return root.parent / f"{root.name}.progress.json"


class _PreparePhases:
    """Publish which phase this stage is in, at each boundary it times.

    THE DEFECT: prepare is the longest quiet stage in the chain (15 s
    warm, 36 s at the documented normal size) and it published nothing
    at all, so ``woof go``'s heartbeat -- which polls exactly this path
    and names whatever the running stage says about itself -- showed a
    bare stopwatch for the whole of it.  Every other long stage in this
    tree answers that question; this one is the reason the heartbeat
    looked like a hang.

    Every write is best-effort.  A progress file that cannot be written
    is not a reason to fail a preparation, and the caller must never
    have to guard a telemetry call.
    """

    def __init__(self, path: Path, *, phases=PREPARE_PHASES):
        self._path = Path(path)
        self._phases = tuple(phases)
        self._started = time.perf_counter()
        self._index = 0

    def enter(self, phase: str, **fields) -> None:
        try:
            self._index = self._phases.index(phase) + 1
        except ValueError:
            self._index = 0
        self._publish(phase, **fields)

    def _publish(self, phase: str, **fields) -> None:
        payload = {
            "schema": PREPARE_PROGRESS_SCHEMA,
            # `status` is what go's heartbeat prints, and it prints it
            # verbatim beside the runner's own SHOUTING_SNAKE statuses,
            # so it is spelled the way they are.
            "status": f"PREPARING_{phase.upper()}",
            "phase": phase,
            "phase_index": self._index,
            "phases_total": len(self._phases),
            "elapsed_seconds": round(time.perf_counter() - self._started, 6),
            **fields,
        }
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self._path.with_name(self._path.name + ".partial")
            temporary.write_text(
                json.dumps(payload, indent=2, sort_keys=True) + "\n",
                encoding="utf-8", newline="\n")
            os.replace(temporary, self._path)
        except OSError:
            pass


def _prepare_output_root_parent(output_root: Path) -> Path:
    """Make sure the directory that will hold --output-root exists.

    The bridge's scratch space is a TemporaryDirectory in the PARENT of
    ``--output-root`` -- so it can be renamed into place on the same
    filesystem -- and an absent parent surfaced as a bare
    ``FileNotFoundError`` traceback about a random ``gpuwm-gfs-bridge-*``
    name, roughly 40 s into the work, after --dry-run had passed
    cleanly.  Creating the parent is what every user did by hand
    (``mkdir -p``); doing it here keeps the create-only guarantee on
    ``--output-root`` itself, which is the directory that matters.
    """

    parent = output_root.parent
    try:
        parent.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        raise NotADirectoryError(
            f"cannot create the parent directory of --output-root "
            f"{output_root}: {error}.  The GFS bridge stages its scratch "
            f"space in {parent} so it can be renamed into place on one "
            "filesystem; choose an --output-root whose parent is writable"
        ) from error
    if not parent.is_dir():
        raise NotADirectoryError(
            f"the parent of --output-root {output_root} is not a "
            f"directory: {parent}")
    return parent


def front_door_physics_selection(
        exp,
        *,
        physics_profile: str | None = None,
        expert_acknowledgements: tuple[str, ...] = (),
) -> dict[str, object]:
    """The front door's physics selection: record, govern, never whitelist.

    One door, one contract now (owner ruling 2026-07-31: the physics
    suite is the user's choice).  The single-domain route used to
    substitute WSM6 when no profile was named and compare the config's
    switches against it for exact equality, so the product default suite
    -- and every hand-authored suite -- was refused at preparation time.
    That whitelist is gone: an UNNAMED configuration, one domain or a
    tree, is governed the way the domain-tree route has always been
    governed -- engine-valid selectors, the registry's source-neutral
    tuple governance with its expert acknowledgements, and a recorded
    blocker (never a refusal) where the registry has no spelling for an
    engine-valid tuple.  Verification status is receipt metadata.

    v1.1.0's regression (this whitelist binding at the front door for
    every configuration, refusing the wizard's default emission as a
    stack trace) and v1.1.1's scoping fix are recorded in this module's
    history; the 2026-07-31 ruling removed the single-domain half too.

    An explicitly named profile is still enforced on either route,
    because a gate the caller asked for remains binding: naming
    ``--physics-profile`` asserts the config IS that shipped suite.
    Land-surface initialization follows the selected scheme and soil
    geometry. Registry source/template membership describes evidence;
    the shared field and category checks decide whether input can run.
    """

    acknowledgements, provenance = acknowledgement_delivery(
        flag=expert_acknowledgements,
        toml=getattr(exp, "acknowledgements", ()),
    )
    if len(exp.domains) == 1:
        # THE single-domain selection spelling, shared with the
        # prepared-forecast runner's governance and recompute so the
        # two sides cannot drift (physics_compat owns it).
        selection = single_domain_physics_selection(
            exp.root.run,
            profile=physics_profile,
            expert_acknowledgements=acknowledgements,
            acknowledgement_provenance=provenance)
    else:
        selection = multi_domain_physics_selection(
            {int(domain.grid_id): domain.run for domain in exp.domains},
            profile=physics_profile,
            expert_acknowledgements=acknowledgements,
            acknowledgement_provenance=provenance)
    return selection


@preprocess_math_call
def prepare_gfs_wrf(
    *,
    series: Path,
    cycle: str,
    bridge: Path | None,
    wps_namelist: Path,
    static_input: Path | None,
    static_receipt: Path | None,
    experiment_config: Path,
    input_manifest: Path,
    input_manifest_sha256: str | None,
    output_root: Path,
    preprocess_backend: str = "auto",
    preprocess_workers: int | None = None,
    cpu_preprocess_bridge: Path | None = None,
    geog_root: Path | None = None,
    hierarchy_workers: int | None = None,
    physics_profile: str | None = None,
    expert_acknowledgements: tuple[str, ...] = (),
    stock_wrf_export: bool = True,
    statics_corridor=None,
    preprocess_backend_reason: str | None = None,
    as_posted: Path | None = None,
    physical_input_store: Path | None = None,
    physical_output_store: Path | None = None,
    physical_input_provider=None,
    physical_member_index: int | None = None,
) -> dict[str, object]:
    """Build native GFS initial/boundary files and return the proof receipt.

    ``as_posted`` is an as-posted fetch's ``posting/`` folder (DESIGN A136
    2.4): the preparation starts once the window's first lead is posted,
    decodes each lead batch as its markers appear, publishes its head from
    the start time binding the INPUT PLAN (every lead, every object name,
    every other input) instead of the input manifest, which does not
    exist yet, and at its seal writes the input manifest at
    ``input_manifest`` (``input_manifest_sha256`` is then ``None``), the
    decoder receipts and the one-shot identity, byte for byte what a
    preparation of the same bytes after the whole window writes.

    ``geog_root`` activates the internal multi-domain route when the
    experiment declares children.  The existing public single-domain CLI is
    unchanged; source-neutral public hierarchy configuration is owned by the
    higher-level integration layer.

    ``stock_wrf_export`` asks for the bonus unchanged-WRF file set beside
    the forecast this prepares, on the single-domain route and the
    domain-tree route alike.  It defaults to True, so a domain the
    exporter can represent publishes exactly what it always published.
    One it cannot represent (Kessler, Milbrandt-Yau and WDM6 have no
    stock package contract) does not fail: the export is refused, by
    name, in the proof's export slot, and the forecast preparation
    completes, because the forecast restores the prepared cache, not the
    exported files.  Which physics a forecast may run belongs to the
    registry and the acknowledgement gate, not to a downstream
    file-format contract.  Pass False to skip the export outright; the
    slot then records it as not requested.  On the single-domain route an
    export that is written must carry exactly the physics this
    preparation selected, or the preparation stops.

    ``statics_corridor`` opts the DOMAIN-TREE route into emitting sealed
    child-resolution statics corridors (``"all"`` or a sequence of child
    grid ids; :mod:`woof.static.corridor`), which is what lets the
    prepared tree runner honor ``[relocation]`` follow sources.  ``None``
    (the default) leaves every published byte exactly as before.
    """

    cycle_time = datetime.strptime(cycle, "%Y-%m-%d_%H:%M:%S")
    if (cycle_time.hour not in {0, 6, 12, 18}
            or cycle_time.minute != 0 or cycle_time.second != 0):
        raise ValueError("GFS cycle must be an exact 00/06/12/18 UTC cycle")
    physical_provider = None
    captured_decoder = None
    if (physical_input_provider is None) != (physical_member_index is None):
        raise ValueError("a physical input provider needs its original member index")
    if physical_input_provider is not None:
        if as_posted is None or physical_input_store is not None or physical_output_store is not None:
            raise ValueError("a GFS posted physical provider needs as-posted input and its own member output")
        from woof.ensemble.posted_physical import PostedPhysicalProvider
        physical_provider = (physical_input_provider if isinstance(physical_input_provider, PostedPhysicalProvider)
                             else PostedPhysicalProvider.open(physical_input_provider,
                                  cpu_bridge=cpu_preprocess_bridge, workers=preprocess_workers or 1))
        source_context = physical_provider.source_context(physical_member_index)
        if (source_context.trajectory.source != "gfs"
                or source_context.trajectory.cycle.replace(tzinfo=None) != cycle_time):
            raise ValueError("GFS member source or cycle differs from its pinned ordinary source")
        member_output = Path(output_root).resolve()
        for protected in (source_context.prepared_root, source_context.physical_stream.root, physical_provider.root):
            if (member_output == protected or member_output.is_relative_to(protected)
                    or protected.is_relative_to(member_output)):
                raise ValueError("GFS member output must be separate from its checked source and provider")
        captured_decoder = source_context.source_plan["manifest"]["files"].get("bridge")
        if not isinstance(captured_decoder, dict):
            raise ValueError("GFS source head has no captured decoder authority")
        if bridge is not None:
            bridge = Path(bridge)
            if (bridge.name != captured_decoder.get("name")
                    or bridge.exists() and (not bridge.is_file() or _sha256(bridge) != captured_decoder.get("sha256"))):
                raise ValueError("GFS member decoder differs from its pinned ordinary source authority")
        else:
            bridge = Path(captured_decoder["name"])
    elif bridge is None:
        raise ValueError("GFS source preparation requires its Rust decoder bridge")
    base_roles = {
        "series": Path(series),
        "bridge": Path(bridge),
        "wps_namelist": Path(wps_namelist),
        "experiment_config": Path(experiment_config),
    }
    static_pair = static_input is not None or static_receipt is not None
    if static_pair and (static_input is None or static_receipt is None):
        raise ValueError(
            "GFS static-input and static-receipt must be supplied together")
    if static_input is not None:
        base_roles["static_input"] = Path(static_input)
        base_roles["static_receipt"] = Path(static_receipt)
    elif geog_root is None:
        raise ValueError(
            "GFS requires either static-input/static-receipt or geog-root")
    posted = None
    if as_posted is not None:
        if input_manifest_sha256 is not None:
            raise ValueError(
                "an as-posted GFS preparation writes its input manifest at "
                "its seal, so it is given no manifest digest to verify")
        for role, path in base_roles.items():
            if role == "bridge" and captured_decoder is not None:
                continue
            if role != "series" and not path.is_file():
                raise FileNotFoundError(
                    f"missing GFS adapter input {role}: {path}")
        posted, records, posted_manifest = _as_posted_plan(
            posting=Path(as_posted), series=Path(series),
            cycle_time=cycle_time, roles=base_roles, captured_decoder=captured_decoder)
    else:
        records = _read_series(Path(series))
    grib_roles = {f"grib-f{hour:03d}": path for hour, path in records}
    roles = {**base_roles, **grib_roles}
    for role, path in roles.items():
        if role == "bridge" and captured_decoder is not None:
            continue
        if posted is not None and (role in grib_roles or role == "series"):
            continue
        if not path.is_file():
            raise FileNotFoundError(f"missing GFS adapter input {role}: {path}")
    if captured_decoder is None and not os.access(Path(bridge), os.X_OK):
        raise PermissionError("GFS Rust bridge is not executable")
    # A head published early whose producer failed, was stopped or went
    # silent is this tool's own unfinished product, not a finished run.
    remove_unfinished_tree(Path(output_root))
    if Path(output_root).exists():
        raise FileExistsError(f"refusing to overwrite {output_root}")
    # A domain tree is published through staging deeper than its
    # output root; refused here, before any GRIB is hashed or decoded.
    # The experiment is read only to count its domains, and only when
    # the limit binds: a single-domain bundle publishes no tree and is
    # not measured against it.  The load the preparation uses follows
    # the manifest verification below.
    refusal = published_path_refusal(
        Path(output_root), wrf_export=stock_wrf_export)
    if (refusal is not None
            and len(load_experiment(Path(experiment_config)).domains) > 1):
        raise ValueError(refusal)
    _prepare_output_root_parent(Path(output_root))
    # THE STAGE CLOCK STARTS HERE, not after the verification below.
    # It used to start after it, so `total` in the receipt excluded the
    # sha256 of every GRIB this stage is bound to -- work that is
    # linear in the download and is nobody's idea of free.  A receipt
    # whose `total` is not the stage's wall clock cannot be summed
    # against the chain's own stage timing, which is the whole point of
    # both numbers existing.
    total_started = time.perf_counter()
    progress = _PreparePhases(prepare_progress_path(output_root))
    progress.enter("verify_inputs", files=len(roles))
    verify_started = time.perf_counter()
    if posted is None:
        manifest = _verify_input_manifest(
            Path(input_manifest), input_manifest_sha256, roles)
        manifest_digest = input_manifest_sha256.lower()
        plan = plan_sha256 = None
    else:
        manifest = posted_manifest
        plan = input_plan(manifest,
                          lead_role_prefix=_AS_POSTED_LEAD_ROLE_PREFIX,
                          route_table_sha256=posted.route_table_sha256(),
                          derived_roles=_AS_POSTED_DERIVED_ROLES)
        plan_sha256 = input_plan_sha256(plan)
        # Where the one-shot identity binds the manifest's digest, the
        # head binds this placeholder of the plan; the seal writes the
        # digest (boundary_stream.check_as_posted_identity).
        manifest_digest = as_posted_placeholder(plan_sha256)
    source_identity = manifest.get("source")
    expected_source_identity = {
        "model": "GFS",
        "product": "pgrb2.0p25",
        "cycle": cycle_time.strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    if (not isinstance(source_identity, dict)
            or any(source_identity.get(key) != value
                   for key, value in expected_source_identity.items())):
        raise ValueError(
            "GFS input manifest source identity differs from model/product/cycle")
    implementation_sha256 = _implementation_sha256()
    git_source_identity = _git_source_identity()
    decoder_digest = manifest["files"]["bridge"]["sha256"]
    experiment_config_digest = manifest["files"]["experiment_config"]["sha256"]
    preprocess = resolve_preprocess_backend(
        preprocess_backend, workers=preprocess_workers,
        cpu_bridge=cpu_preprocess_bridge, reason=preprocess_backend_reason)
    preprocess_receipt = preprocess.receipt()
    # Everything above verified or identified an input: the manifest's
    # digest over every GRIB, this build's implementation hash, the
    # git identity, the preprocess backend's own receipt.  Named as one
    # key because it is one answer to one question -- "how long did
    # binding this stage to its inputs take" -- and because a reader
    # chasing a slow prepare needs to know whether the wall went to
    # hashing the download or to the work after it.
    verify_inputs_seconds = time.perf_counter() - verify_started

    exp = load_experiment(Path(experiment_config))
    physical_input = physical_output = physical_stream = None
    if physical_input_store is not None or physical_input_provider is not None:
        from woof.ensemble.posted_preparation import require_exclusive_native_consumer
        require_exclusive_native_consumer(physical_input_store=physical_input_store,
                                         physical_input_provider=physical_input_provider)
    if (physical_input_store is not None or physical_output_store is not None
            or physical_input_provider is not None or physical_member_index is not None):
        from woof.ensemble.physical_store import NativePhysicalStore
        if len(exp.domains) != 1:
            raise ValueError(
                "GFS physical fields require a single-domain native preparation")
        if physical_input_store is not None and as_posted is not None:
            raise ValueError("an as-posted GFS member requires a physical input provider, not a complete store")
        if physical_input_store is not None:
            physical_input = NativePhysicalStore(physical_input_store)
    # THE FLOOR, BEFORE THE DECODE (A98).  The domains alone set a lower
    # bound on the card price: an explicit cuda they cannot fit is refused
    # here, in seconds, instead of after the host decode and the statics,
    # and auto moves to the CPU here.  The decoded price below stays the
    # binding check.
    preprocess = admit_preparation(
        preprocess, lambda: price_preparation_floor("gfs", exp),
        workers=preprocess_workers)
    preprocess_receipt = preprocess.receipt()
    from woof.static.highres_production import (
        load_static_highres, apply_prepared_highres, static_highres_identity)
    static_highres = load_static_highres(experiment_config)
    from woof.case_data import (
        gfs_preparation_policy, optional_case_data_from_config,
        preparation_case_policy)
    case_data = optional_case_data_from_config(experiment_config)
    case_policy = {**preparation_case_policy(case_data),
                   **gfs_preparation_policy(case_data)}
    from woof.ingest.water_overlay import (
        load_bound_water_overlay, overlay_snapshot_sequence, verify_overlay_sequence)
    water_overlay, water_overlay_binding = load_bound_water_overlay(
        None if case_data is None else case_data.water_temperature_overlay)
    from woof.experiment import (
        deferred_initial_perturbation, refuse_unrouted_spawn,
    )
    refuse_unrouted_spawn(exp, "GFS-direct prepared-cache")
    initial_perturbation = deferred_initial_perturbation(
        exp, "GFS-direct prepared-cache")
    physics_selection = front_door_physics_selection(
        exp, physics_profile=physics_profile,
        expert_acknowledgements=expert_acknowledgements)
    if physical_provider is not None:
        if statics_corridor is not None:
            raise ValueError("--statics-corridor needs child domains; a physical member here has one domain")
        from woof.ensemble.gfs_posted_reuse import prepare_posted_gfs_member
        return prepare_posted_gfs_member(physical_provider, physical_member_index,
            input_plan=plan, experiment_config=experiment_config, wps_namelist=wps_namelist,
            output_root=output_root, preprocess=preprocess, preprocess_workers=preprocess_workers,
            physics_profile=physics_profile, expert_acknowledgements=expert_acknowledgements,
            physics_selection=physics_selection, case_policy=case_policy,
            water_overlay_binding=water_overlay_binding, stock_wrf_export=stock_wrf_export,
            implementation_sha256=implementation_sha256, git_source_identity=git_source_identity,
            input_manifest=input_manifest, progress=progress)
    # The model's time zero is the experiment's, and it may sit at any
    # source lead the series carries -- not only at the cycle.  Two hour
    # vocabularies live from here down and they are never interchanged:
    # SOURCE leads (f000, f018, f174 ...) name NOAA products, and MODEL
    # forcing offsets (0, 3, 6 ...) count from start_time.  Everything
    # downstream of this door -- the prepared cache, the WRF export, the
    # forecast runner -- speaks model offsets, which is why a run at lead
    # 0 is bit-for-bit the run it always was.
    series_hours = [hour for hour, _ in records]
    lead_hours = resolve_initial_forecast_lead(
        start_time=exp.start_time, cycle_time=cycle_time,
        source_hours=series_hours)
    provenance = initial_condition_provenance(
        cycle_time=cycle_time, lead_hours=lead_hours)
    initial_index = series_hours.index(lead_hours)
    source_hours = series_hours[initial_index:]
    if lead_hours:
        # Said BEFORE the window checks below, so a refusal about the
        # window is read in the light of the lead that shaped it.  Warn,
        # do not block: this is a legitimate initialization whose one
        # consequence the reader has to be told once.
        warn(provenance["statement"],
             "Every field this door consumes is an instantaneous GRIB2 "
             "product definition template 4.0 record, refused by "
             "gfs_grib2_bridge if it is anything else, so no consumed "
             "field carries an accumulation or averaging window whose "
             "meaning would depend on the lead.  What does depend on the "
             "lead is forecast skill: the state you are starting from "
             f"is GFS's own {lead_hours} h forecast, not an analysis.")
        if initial_index:
            warn(
                f"the GFS series carries {initial_index} forecast hour(s) "
                f"before f{lead_hours:03d} "
                + "(" + ", ".join(f"f{hour:03d}"
                                  for hour in series_hours[:initial_index])
                + "); they are decoded and bound by the input manifest but "
                "are not used by this run",
                "The series and its manifest are one hash-bound document: "
                "the door verifies every file the series names rather than "
                "silently ignoring some.  Author a manifest over the tail "
                "you want (`woof fetch --author-front-door-manifest "
                f"--forecast-start-hour {lead_hours}`) to decode only it.")
    if len(source_hours) < 2:
        raise ValueError(
            f"GFS series ends at forecast lead f{lead_hours:03d}, the lead "
            "this experiment starts from, so it carries no lateral boundary "
            "time at all; fetch at least one more forecast hour")
    hours = [hour - lead_hours for hour in source_hours]
    if hours[-1] * 3600 < exp.run_seconds:
        raise ValueError(
            f"GFS series does not cover the configured run: it reaches "
            f"f{source_hours[-1]:03d}, which is {hours[-1]} h after the "
            f"f{lead_hours:03d} this experiment starts from, and the run "
            f"is {exp.run_seconds / 3600.0:g} h long")
    boundary_interval_seconds = (hours[1] - hours[0]) * 3600
    cfg = exp.root.run
    # What the fetch actually reached, not what the default ladder would
    # have reached: the case's p_top is checked against this.
    _manifest_source_top_pa(manifest)  # Validate the original receipt unchanged.
    expected_levels_hpa = _decode_levels_above_model_top(manifest, exp.vertical.p_top)
    source_top_pressure_pa = min(expected_levels_hpa) * 100.0
    if len(exp.domains) == 1:
        grid = _validate_grid_and_vertical_contract(
            exp, Path(wps_namelist),
            source_top_pressure_pa=source_top_pressure_pa)
        grids = (grid,)
    else:
        if geog_root is None:
            raise ValueError(
                "nested GFS preparation requires an explicit geog_root")
        grids = validate_native_lambert_contracts(
            exp, Path(wps_namelist), source_name="GFS",
            source_top_pressure_pa=source_top_pressure_pa,
        )
        grid = grids[0]
    landuse_attrs = None
    # THE 14.1 SECONDS.  MEASURED, 2026-08-16: at the documented normal
    # size this block was 40% of the prepare stage and the largest
    # single item in it, and proof.json named it nowhere -- the only
    # timing that existed was `static.build_static` under the opt-in
    # WOOF_PERF_TIMING env var, which under the fixed-means-default law
    # is a deep-dive diagnostic and not instrumentation.  The number is
    # taken here, unconditionally, with a plain perf_counter pair
    # exactly like the four keys the receipt already had.  The env var
    # is untouched and still does what it always did.
    static_started = time.perf_counter()
    progress.enter("static_build",
                   cells=int(cfg.ny) * int(cfg.nx),
                   source=("geog" if static_input is None else "prebuilt"))
    if static_input is None:
        static, root_static_receipt, landuse_attrs = _static_from_geog(
            Path(wps_namelist), Path(geog_root), grid, cfg,
            static_highres=static_highres)
        root_static_provider = "native-wps-geog"
    else:
        root_static_receipt = _verify_static_receipt(
            Path(static_receipt), Path(static_input), grid, cfg)
        static = _load_static(Path(static_input), grid, cfg.ny, cfg.nx)
        root_static_provider = "prebuilt-hash-bound-cache"
        if geog_root is not None:
            # The prebuilt NPZ is numeric fields only, so the land-use
            # table has to come from the GEOG index beside it.
            from woof.hrrr_native_static import verified_static_catalog
            from woof.static.build import geog_selection_from_catalog
            from woof.static.terrain_smoothing import (
                selection_carrier_kwargs)

            catalog, _ = verified_static_catalog(
                Path(wps_namelist), Path(geog_root), (1,),
                **selection_carrier_kwargs(static_highres))
            landuse_attrs = geog_selection_from_catalog(
                catalog, 1).landuse_global_attrs()
    static, root_static_receipt = apply_prepared_highres(
        static, grid, config=static_highres, domain_id=1,
        case_date=exp.start_time.date(), landuse_attrs=landuse_attrs,
        baseline_receipt=root_static_receipt, run=cfg)
    # The land-height check reads the terrain the run will integrate: a
    # declared high-resolution terrain gives the islands the baseline
    # dataset holds at 0 m their height.
    require_land_terrain(static["HGT_M"], static["LANDMASK"])
    # Both roads, one key: "build it from the geography tree" and "load
    # and verify the prebuilt cache" are two answers to one question,
    # and a receipt that timed only the first would go quiet on exactly
    # the runs that chose the second.
    static_build_seconds = time.perf_counter() - static_started

    # THE COORDINATE, BEFORE ANYTHING IS BUILT ON IT.  The root terrain
    # exists now and every other terrain this run can touch is derivable
    # from the same geography, so this is the last moment at which the
    # vertical coordinate is still a decision rather than an assumption.
    # Past here the root coordinate, the children's shared one, the
    # exported WRF input and the prepared caches all carry whatever was
    # chosen here.
    exp, vertical_adaptation = adapt_experiment_for_statics(
        exp, grids, root_terrain=static["HGT_M"],
        static_catalog=_survey_static_catalog(
            exp, wps_namelist, geog_root, static_highres),
        static_highres=static_highres, announce=_announce_adaptation,
        statics_corridor=statics_corridor)
    cfg = exp.root.run

    # One descriptor per decoded array is held for the whole bundle
    # write, and the array count is forcing-times x fields-per-time --
    # linear in forecast length.  Ask for the budget now, while the
    # answer can still be a sentence: past here the shortfall arrives as
    # an [Errno 24] naming whichever file was next, which is how a hard
    # ceiling on forecast length looked like a random IO fault.
    refusal = ensure_descriptor_headroom(
        required_descriptors(len(records), len(_THREE_D) + len(_TWO_D)),
        what=f"preparing {len(records)} GFS forcing time(s)")
    if refusal is not None:
        raise ValueError(refusal)

    decode_started = time.perf_counter()
    progress.enter("decode", forcing_times=len(records))
    # ignore_cleanup_errors: when the body fails for a reason that also
    # stops the scratch tree being removed -- descriptor exhaustion is
    # exactly that, since rmtree needs one per directory it walks -- the
    # cleanup's exception REPLACES the body's, and the run reports the
    # rmtree's victim instead of its own fault.  Both measured EMFILE
    # failures named `decoded/f000`, the first directory the cleanup
    # reached and a path each run had already read successfully.  The
    # tree is still removed; only its failure stops overwriting the real
    # one.
    with tempfile.TemporaryDirectory(
            prefix="gpuwm-gfs-bridge-", dir=Path(output_root).parent,
            ignore_cleanup_errors=True) as temporary:
        decoded = Path(temporary) / "decoded"
        target_coordinates = tuple(
            pair for target in grids
            for pair in (target.latlon_mass(), target.latlon_u(), target.latlon_v()))
        target_longitudes = tuple(pair[1] for pair in target_coordinates)
        target_latitudes = tuple(pair[0] for pair in target_coordinates)
        bridge_environment = {
            **os.environ, "GPUWM_GFS_SNOW_CONVENTION": case_policy["gfs_snow_convention"]}
        if preprocess_workers is not None:
            bridge_environment["GPUWM_GFS_BRIDGE_THREADS"] = str(int(preprocess_workers))
        posted_series = None
        if posted is not None:
            posted_series = _PostedGfsSeries(
                posted=posted, records=records, cycle_time=cycle_time,
                bridge=Path(bridge), scratch=Path(temporary),
                levels_pa_csv=(None if expected_levels_hpa is None
                               else ",".join(format(level * 100.0, "g")
                                             for level in expected_levels_hpa)),
                environment=bridge_environment,
                orient=lambda snapshot: orient_global_source_longitudes(
                    snapshot, *target_longitudes,
                    target_latitudes=target_latitudes))
        bridge_command = [
            str(Path(bridge)), "--series", str(Path(series)), str(decoded),
            cycle_time.strftime("%Y-%m-%d %H:%M:%S")]
        if expected_levels_hpa is not None:
            # Declare the request's ladder rather than letting the
            # bridge derive one from the file.  A NOMADS subset carries
            # exactly the requested levels, so this changes nothing
            # there; a --mode full-file object carries every published
            # level up to 1 Pa, and deriving the ladder from IT would
            # decode the mesosphere nobody asked for.  The gate-versus-
            # manifest ladder comparison below still runs on the
            # bridge's own report of what it decoded.
            bridge_command += ["--pressure-levels-pa", ",".join(
                format(level * 100.0, "g") for level in expected_levels_hpa)]
        # The bridge reads and decodes the forecast hours concurrently on
        # every core it may run on; an explicit --preprocess-workers is the
        # count every host step takes, this one included.
        # Bounded by this host's RAM: a series wider than the window is
        # decoded in lead batches whose receipts merge to the one decode's
        # (_decode_series_bounded); a series that fits is the one run.
        completed = (None if posted_series is not None else
                     _decode_series_bounded(
                         bridge_command, series=Path(series), records=records,
                         decoded=decoded, scratch=Path(temporary),
                         bridge=Path(bridge), cycle_time=cycle_time,
                         levels_pa_csv=(None if expected_levels_hpa is None
                                        else ",".join(
                                            format(level * 100.0, "g")
                                            for level in expected_levels_hpa)),
                         environment=bridge_environment))
        if posted_series is not None:
            # As posted, the decode reads the leads posted so far: through
            # the start time here (one batch), then each lead as the build
            # of its forcing time asks for it.  A declared water overlay
            # binds its proof over the whole sequence, so it is decoded
            # whole first.
            posted_series.through(
                len(records) - 1 if water_overlay is not None
                else initial_index)
            decoded_snapshots = posted_series.view(0)
            head_decode_seconds = posted_series.decode_seconds
        elif completed.returncode != 0:
            # The bridge already fails closed and says why, including for
            # a GRIB whose valid time is not the one requested.  Raising
            # a bare RuntimeError meant that careful, content-based
            # diagnosis was delivered as a traceback at rc 1, because
            # main() catches only (ValueError, OSError) -- the detection
            # was right and only the delivery was wrong.
            raise GfsRouteError(decode_failure_message(
                "GFS Rust bridge", completed.stderr))
        if posted_series is None:
            expected_source_sha256 = {
                hour: manifest["files"][f"grib-f{hour:03d}"]["sha256"]
                for hour, _ in records
            }
            decoded_snapshots = _load_bridge_snapshots(
                decoded, cycle_time, records, expected_source_sha256,
                expected_levels_hpa=expected_levels_hpa,
                expected_snow_convention=case_policy["gfs_snow_convention"])
        # A NOMADS box that crosses the source's own 0/360 origin cannot be
        # asked for as one subregion, so `Area.as_nomads` widens it to the
        # whole band (fetch.py:348-362) and the decode returns a ring stored
        # with an arbitrary cut at longitude 0.  Re-cut it opposite the
        # target before anything indexes it; a regional crop is untouched.
        if posted_series is None:
            decoded_snapshots = _map_snapshots(
                decoded_snapshots,
                lambda snapshot: orient_global_source_longitudes(
                    snapshot, *target_longitudes,
                    target_latitudes=target_latitudes))
        # From here on the run sees only the window that starts at the
        # experiment's lead.  Each snapshot still carries the SOURCE
        # valid time the bridge decoded (cycle + its own lead), which is
        # exactly start_time + the model offset.
        if posted_series is not None and water_overlay is None:
            snapshots = posted_series.view(initial_index)
        else:
            snapshots = overlay_snapshot_sequence(
                decoded_snapshots[initial_index:], water_overlay,
                binding=water_overlay_binding,
                workers=host_step_workers(preprocess))
        decode_seconds = time.perf_counter() - decode_started
        # The land-use table's own ISLAKE, never a hard-coded 21: the
        # category number is a property of the selected table, and a table
        # without an inland-water class is a state the assembly declares
        # rather than a silently empty mask.  lake_override=True because
        # the soil call below hands the router a lake_mask/lake_skin pair,
        # which makes every one of these cells water inside the router.
        if (landuse_attrs is None and case_data is not None
                and (case_data.water_temperature_policy is not None or water_overlay is not None)
                and case_policy["water_temperature_policy"] != "wrf_compat"):
            raise ValueError("declared water-temperature policy needs the selected land-use "
                             "table's ISLAKE metadata; provide --geog-root with this static cache")
        water_statics = (
            None if landuse_attrs is None
            else WaterTemperatureStatics.for_route(
                route=_WATER_ROUTE, policy=case_policy["water_temperature_policy"],
                landmask=static["LANDMASK"], lu_index=static["LU_INDEX"],
                landuse_attrs=landuse_attrs, lake_override=True))
        if water_statics is None:
            # LOUD, and no coherent receipt is printed below: with a
            # prebuilt static cache and no geog root there is no
            # land-use table to read ISLAKE from, so the shipped MODIS
            # category is ASSUMED rather than resolved.  Say which
            # number is being assumed and how to stop assuming it.
            print(
                "lake mask: the prebuilt static cache carries no "
                "land-use metadata, so this run assumes the MODIS "
                f"inland-water category {MODIS_LAKE_CATEGORY} instead "
                "of reading ISLAKE, and keeps the historical per-cell "
                "water temperature.  Pass --geog-root, which is read "
                "only for the land-use index, to resolve it.",
                file=sys.stderr)
            lake_mask = static["LU_INDEX"] == MODIS_LAKE_CATEGORY
        else:
            lake_mask = water_statics.lake
        def coverage_of(hours_and_snapshots):
            return {
                f"f{hour:03d}": _source_coverage_receipt(
                    snapshot, grid, lake_mask)
                for hour, snapshot in hours_and_snapshots
            }

        if posted_series is None:
            coverage_receipt = coverage_of(zip(source_hours, snapshots))
        else:
            # The leads decoded so far; the seal writes every lead's.
            coverage_receipt = coverage_of(
                (hour, posted_series.snapshots[hour])
                for hour in source_hours if hour in posted_series.snapshots)

        from woof.core.grid import make_vertical_coord

        # THE FIT, BEFORE THE FIRST DEVICE ALLOCATION.  The decode above
        # ran on the host; everything below builds on the chosen backend.
        # auto moves a preparation the card cannot hold to the CPU, and an
        # explicit cuda that cannot fit is refused here, by name, instead
        # of stopping in DomainState.__init__ minutes later (A65).  A
        # backend already on the CPU is not priced.
        preprocess = admit_preparation(
            preprocess,
            lambda: price_forcing_preparation("gfs", exp, snapshots),
            workers=preprocess_workers)
        preprocess_receipt = preprocess.receipt()
        initialize_started = time.perf_counter()
        progress.enter("initialize_all_times",
                       forcing_times=len(records))
        # CHAINED TREES.  A tree's children need only the start time and
        # the root's start state, so a tree is chained exactly like a
        # single domain, on either backend: the start time first, the
        # children into the head, one root interval per segment, the
        # one-shot tree at the seal (_prepare_chained_gfs_tree).
        # An as-posted tree takes the same head and seal whether or not its
        # head is published early: its children bind the input plan at the
        # head and its seal writes the one-shot tree, which the start-last
        # hierarchy (every time before any artifact) cannot do.
        chain_tree = len(exp.domains) > 1 and (
            chained_enabled() or posted_series is not None)
        if getattr(exp, 'nudging', None) is not None:
            if posted_series is not None:
                raise ValueError('NUDGING_UNSEALED_ANALYSIS_IDENTITY: an as-posted input plan changes its source digest at seal; prepare the complete source window before nudging')
            # The complete hierarchy join also builds the selected nests'
            # full-domain analyses. A start-only child head cannot supply them.
            chain_tree = False
        # ONE forcing time is ever resident.  A single domain builds the
        # start time FIRST, publishes it in the prepared head and releases
        # it, then writes each boundary interval as soon as its two times
        # exist (woof.ingest.boundary_stream).  A chained tree does the
        # same, and its seal re-reads every start state from the head
        # (boundary_stream.TreeStartStates).  An unchained hierarchy builds
        # the start time LAST (start_last_forcing_order) and retains only
        # that met/state; every other time contributes its perimeter frames
        # against its own position and is released before the next one is
        # interpolated.  Walking the times in order instead meant holding
        # the start time -- which nothing reads until the boundaries are
        # complete -- while each later time was built underneath it.  At
        # 800x800x49 with mp=10 and three GFS times that second resident
        # time is 14.67 GiB of device residency against 7.66, a priced
        # peak envelope of 23.92 GiB against 15.86: the difference
        # between preparing the domain on a 16 GiB card and OOMing after
        # the whole forcing chain had already been fetched.
        initial_result = None
        initial_met = None
        forcing = StateBoundaryFrames(
            spec_bdy_width=cfg.spec_bdy_width,
            spec_zone=cfg.spec_zone, relax_zone=cfg.relax_zone)
        interpolation_landmask = np.asarray(
            static["LANDMASK"] >= 0.5, dtype=np.bool_).copy()
        # WPS metgrid treats lakes as water for skin temperature and ice,
        # and excludes them from the land snow/soil donor pass.
        interpolation_landmask[lake_mask] = False
        # Lake skin temperature: by default (gfs_lake_skin = "wps") a lake
        # keeps metgrid's whole-array water search, so a lake GFS does not
        # resolve takes the nearest GFS water exactly as WPS and real.exe
        # give it (TSK from SKINTEMP, no use_tavg_for_tsk) -- the answer
        # the 2.8.x route gave its lakes through its own nearest-water
        # override.  The inland-water reach would otherwise send every such
        # lake to the 2 m air temperature: 10,410 cells and a 3.1 K
        # lake-only RMS against the WPS 4.6.0 / real.exe 4.7.1 witness of
        # 2026-03-02 12Z.  "air_temperature" keeps that reach for lakes
        # too.  Non-lake inland water keeps the reach either way.
        lake_skin_search = (
            np.asarray(lake_mask, dtype=np.bool_)
            if case_policy["gfs_lake_skin"] == "wps" else None)
        # WRF's smooth_cg_topo (woof.ingest.cg_topo): the root terrain is
        # blended toward GFS's once, before the first initialization reads
        # it.  Off, this does nothing.
        terrain_blend = RootTerrainBlend(exp, static, route="gfs")

        native_source_identity = {
            "adapter": "gfs-pgrb2-0p25-direct-v1",
            **({"static_highres": static_highres_identity(static_highres)}
               if static_highres is not None else {}),
            "preparation_case_policy": case_policy,
            "water_temperature_overlay": water_overlay_binding,
            "input_manifest_schema": manifest["schema"],
            "input_manifest_sha256": manifest_digest,
            # Cycle and lead remain distinct source authorities.
            "initial_condition": provenance,
            "source_forecast_hours": list(source_hours),
            "decoded_forecast_hours": list(series_hours),
            "decoder": {
                "name": Path(bridge).name,
                "sha256": decoder_digest,
                "implementation": "gpuwm-all-rust-gfs-grib2-bridge",
            },
            "relative_humidity_convention": "GFS mixed ice/water converted by WPS fix_gfs_rh",
            "lake_skin_temperature": (
                "WPS masked water interpolation, whole-source search"
                if case_policy["gfs_lake_skin"] == "wps" else
                "WPS masked water interpolation within the inland-water "
                "reach, else the 2 m air temperature"),
            "soil_mapping": "exact GFS Noah 4-layer copy",
            "initial_hydrometeors": (
                "explicit zero (WRF Vtable.GFS parity)"),
            "implementation_sha256": implementation_sha256,
            "git_source_identity": git_source_identity,
            "preprocessing": preprocess_identity(preprocess_receipt),
            **({"initial_perturbation": initial_perturbation}
               if initial_perturbation is not None else {}),
        }
        if physical_input is not None or physical_output_store is not None:
            from woof.ensemble.physical_store import (
                NativePhysicalStore, physical_input_binding, physical_static_identity)
            from woof.ensemble import gfs_physical_contract
            from woof.native_wrf_contract import native_landuse_identity
            if physical_input is not None:
                if physical_input.document["grid"] != _geometry_contract(grid, cfg):
                    raise ValueError("GFS physical input geometry differs from native preparation geometry")
                gfs_physical_contract.require_native_gfs_field_contract(
                    physical_input.require_field_contract(), _geometry_contract(grid, cfg))
            if physical_output_store is not None:
                field_contract = gfs_physical_contract.native_gfs_field_contract(
                    _geometry_contract(grid, cfg), evidence={
                        ("input_manifest" if posted_series is None else "input_plan"):
                            manifest_digest if posted_series is None else plan_sha256,
                        "native_decoder": decoder_digest,
                        "decoder_gate": _sha256((decoded if posted_series is None
                                                  else posted_series.batches[0]) / "gate.tsv"),
                        "gfs_adapter": _sha256(Path(__file__)),
                        "horizontal_mapping": _sha256(Path(__file__).parent / "ingest" / "horiz.py"),
                        # The contract is a packaged document now: bind the
                        # document that defines it, not the import path.
                        "field_contract": gfs_physical_contract.contract_sha256(),
                    })
                if posted_series is None:
                    physical_output = NativePhysicalStore(
                        physical_output_store, grid_identity=_geometry_contract(grid, cfg),
                        source_identity=dict(native_source_identity), field_contract=field_contract)

        def build_forcing_time(index):
            # One forcing time's build, unchanged.  A single domain and a
            # chained tree call it start first
            # (woof.ingest.boundary_stream); an unchained hierarchy keeps
            # the start time last.
            nonlocal physical_stream
            source = snapshots[index]
            if physical_input is None:
                met = interpolate_era5_to_lambert(
                    source, grid,
                    target_landmask=interpolation_landmask,
                    relative_humidity_convention="gfs_mixed",
                    backend=preprocess,
                    inland_water_search=lake_skin_search,
                )
            else:
                met = physical_input.read(index)
            terrain_blend.before_initialize(
                met.fields.get("SOURCE_OROGRAPHY"))
            if index == 0 and (physical_input is not None or physical_output_store is not None):
                static_identity = physical_static_identity(
                    native_static_export_fields(static, grid),
                    native_landuse_identity(static))
                if physical_input is not None:
                    native_source_identity["ensemble_physical_input"] = physical_input_binding(
                        physical_input, grid, cfg, native_source_identity,
                        input_manifest_sha256=manifest_digest, static_identity=static_identity)
                if physical_output is not None:
                    physical_output.document["source"] = {
                        **native_source_identity, "static_identity": static_identity}
                if physical_output_store is not None and posted_series is not None:
                    from woof.ensemble.posted_physical import PostedPhysicalStream, posted_source_identity
                    from woof.ensemble.recipes import SourceTrajectory
                    from datetime import timezone
                    trajectory = SourceTrajectory(posted.source, cycle_time.replace(tzinfo=timezone.utc),
                                                  posted_series.markers[source_hours[0]].get("member"))
                    physical_stream = PostedPhysicalStream.create(
                        physical_output_store, trajectory=trajectory,
                        valid_times=[value.replace(tzinfo=timezone.utc) for value in times],
                        grid_identity=_geometry_contract(grid, cfg),
                        source_identity=posted_source_identity(
                            {**native_source_identity, "static_identity": static_identity}, input_plan=plan),
                        field_contract=field_contract, input_plan_sha256=plan_sha256)
            if physical_stream is not None:
                from woof.ingest.boundary_stream import posted_lead_marker_sha256
                hour = source_hours[index]
                batch = next(path for path, leads in zip(posted_series.batches, posted_series.batch_leads)
                             if hour in leads)
                physical_stream.publish(met,
                    posted_leads={str(hour): posted_lead_marker_sha256(posted_series.markers[hour])},
                    decoded_leads={str(hour): _sha256(batch / "decoded-sha256.tsv")})
            if physical_output is not None:
                physical_output.write(met)
            from woof.ensemble.posted_preparation import (
                current_posted_preparation, replace_current_native_snapshot)
            if current_posted_preparation() is not None:
                met = replace_current_native_snapshot(met, grid=grid, cfg=cfg,
                    static_fields=static, landuse_attrs=landuse_attrs,
                    domain_id=exp.root.grid_id, metadata={
                        "input_manifest": manifest, "input_manifest_sha256": manifest_digest,
                        "input_plan": plan, "input_plan_sha256": plan_sha256,
                        "source_snapshot": source, "source_forecast_hours": tuple(source_hours),
                        "decoded_forecast_hours": tuple(series_hours), "decoder_sha256": decoder_digest,
                        "initial_condition": provenance, "preprocessing": preprocess_identity(preprocess_receipt),
                        "preparation_case_policy": case_policy,
                        "water_temperature_overlay": water_overlay_binding,
                        "implementation_sha256": implementation_sha256,
                        "git_source_identity": git_source_identity})
            coord = make_vertical_coord(
                cfg.nz, hybrid_opt=cfg.hybrid_opt, etac=cfg.etac,
                eta_levels=exp.vertical.eta_levels)
            initialized = initialize_real(
                met, cfg, coord, static["HGT_M"], grid=grid,
                landmask=static["LANDMASK"],
                p_top=exp.vertical.p_top, sfcp_to_sfcp=case_policy["sfcp_to_sfcp"],
                preprocess_backend=preprocess,
                state_backend="preprocess", boundary_only=index != 0)
            initialized.state.set_map_coriolis(
                static["MAPFAC_M"], static["MAPFAC_U"], static["MAPFAC_V"],
                static["F"], static["E"], sina=static["SINALPHA"],
                cosa=static["COSALPHA"])
            if nudging_writer is not None:
                nudging_writer.write(index, initialized)
            return met, initialized

        # The plan's times as posted (each snapshot's valid time is the
        # cycle plus its lead, gfs_direct._load_bridge_snapshots).
        times = (_sequence_valid_times(snapshots)
                 if posted_series is None else
                 tuple(cycle_time + timedelta(hours=hour)
                       for hour in source_hours))
        if physical_input is not None and tuple(physical_input.times) != times:
            raise ValueError("GFS physical input valid times differ from the complete initial/boundary window")
        nudging_writer = None
        if getattr(exp, "nudging", None) is not None:
            from woof.spectral_nudging.prepared import analysis_writer
            nudging_writer = analysis_writer(output_root, exp, times, manifest_digest)
        if len(exp.domains) > 1 and not chain_tree:
            for index in start_last_forcing_order(len(snapshots)):
                met, initialized = build_forcing_time(index)
                forcing.add_state(initialized.state, index=index)
                if index == 0:
                    initial_met = met
                    initial_result = initialized
                else:
                    del met, initialized
                    release_backend_memory(preprocess)
            boundaries = forcing.build(times)
            attach_lateral_boundaries(initial_result.state, boundaries)
        else:
            # START FIRST: the start time makes the head, the later times
            # are built after it is published, one resident at a time
            # (a single domain, or a chained tree).
            initial_met, initial_result = build_forcing_time(0)
            forcing.add_state(initial_result.state, index=0)
            boundaries = None
        # The chem processes' preparation-time arrays (dust statics, sulfur
        # lat/lon), before the head is published; nothing on a chem-off
        # state (woof/core/chem_statics.py).
        from woof.core.chem_statics import attach_chem_statics
        attach_chem_statics(initial_result.state, grid, geog_root, None)
        # Use the same WPS water interpolation that supplied lake ice.  The
        # old nearest-water override was a workaround for mapping lakes as
        # land and could replace a valid water-side interpolated temperature.
        from woof.ingest.horiz import _as_host_float64
        lake_skin = _as_host_float64(initial_met.fields["SKINTEMP"]).copy()
        if case_policy["gfs_lake_skin"] == "wps":
            _announce_lake_source_water(
                max((receipt.get("lake_cells_nearest_water_past_crop", 0)
                     for receipt in coverage_receipt.values()), default=0),
                lake_cells=int(np.count_nonzero(lake_mask)))
        water_temperature = None
        if water_statics is not None:
            from woof.ingest.horiz import _as_host_float64

            assembly_skin = _as_host_float64(
                initial_met.fields["SKINTEMP"]).copy()
            # Reuse the already mapped water temperature where finite.
            # Every lake value is finite wherever the source carries a
            # skin temperature at all (see
            # masked surface mapping), and only finite ones
            # are substituted.
            usable_lake = lake_mask & np.isfinite(lake_skin)
            assembly_skin[usable_lake] = lake_skin[usable_lake]
            assembly = assemble_for_route(
                water_statics, mapped_sst=None, mapped_skin=assembly_skin,
                workers=host_step_workers(preprocess))
            water_temperature = assembly.values
            # The receipt is printed only now that the field it describes
            # is the one soil consumes, two statements below.
            announce_water_temperature(assembly.receipt)
        # WRF's `adjust_soil_temp_new` lapse, which this route was not
        # arming.  The elevation pair is all-or-none in
        # `preprocess_noah_soil`, and passing NEITHER satisfies that guard
        # silently -- so TSK and every GFS_ST* level came through with no
        # elevation correction at all and nothing said so.  Measured, the
        # regression of TSK on terrain was -0.0003 K/m against WRF's
        # -0.0065.  `SOURCE_OROGRAPHY` was decoded all along: the bridge
        # gate above refuses any decode whose invariant_fields is not
        # "SOURCE_OROGRAPHY,LANDSEA".  Resolved through the one shared
        # resolver, with no declared artifact on this route.
        soil_orography = soil_source_orography(None, initial_met.fields)
        # THE RECONCILED CATEGORY, never the raw geogrid SCT_DOM: the one
        # rulebook the ERA5 door and the nested child already follow
        # (woof/ingest/soil.py: door_reconciled_soil_category).  Raw, a
        # shoreline land column with the water soil category killed a RUC
        # run on its first surface call (`mavail must be finite`) after a
        # full preparation -- the death the retired GFS+RUC route refusal
        # used to pre-empt (ENG-009).
        # OBSERVED LAKE ICE (woof.ingest.lake_ice).  GFS carries no ice for
        # a lake it does not resolve, and the nearest-water search above
        # hands such a lake the nearest GFS water's ice (none) and skin, so a
        # frozen lake started as open water far below freezing.  The daily
        # observed analysis freezes the lake cells it sees frozen; the
        # provider rides the child catalog so every nest is frozen from the
        # same analysis.  gfs_lake_ice = "wps" keeps the forcing's ice (the
        # WPS / real.exe identity arm); an unreadable analysis falls back to
        # it loudly, for this run only.
        from woof.ingest.lake_ice import resolve_lake_ice_provider
        lake_ice_provider = resolve_lake_ice_provider(
            case_policy["gfs_lake_ice"], times[0],
            where=f"the GFS start {times[0]:%Y-%m-%d %H} UTC")
        _lake_lat, _lake_lon = grid.latlon_mass()
        soil_fields, lake_skin, lake_ice_receipt = lake_ice_provider.freeze(
            initial_met.fields, lake_skin, lat=_lake_lat, lon=_lake_lon,
            lake_mask=lake_mask, dx_m=float(cfg.dx),
            where=f"domain {exp.root.grid_id}")
        native_source_identity = {**native_source_identity,
                                  "lake_ice": lake_ice_receipt}
        from woof.core.landuse import (
            ruc_fractional_seaice as _ruc_fractional_seaice,
            seaice_threshold_of as _seaice_threshold_of)
        soil = preprocess_land_surface_soil(
            soil_fields,
            # real.exe's adjust_for_seaice_pre/post keep the fraction under
            # fractional_seaice = 1 (threshold 0.02) and snap to 0/1 at 0.5
            # otherwise (module_soil_pre.F:216-219, :337-343, :392-393 of the HRRR
            # v4.1.21 fork).
            fractional_seaice=_ruc_fractional_seaice(cfg),
            seaice_threshold=_seaice_threshold_of(cfg),
            sf_surface_physics=int(cfg.sf_surface_physics),
            num_soil_layers=int(cfg.num_soil_layers),
            soil_type=door_reconciled_soil_category(
                static, initial_met.fields, landuse_attrs, route="GFS"),
            deep_soil_temperature=static["TMN"], lake_mask=lake_mask,
            lake_skin_temperature=lake_skin,
            landmask=static["LANDMASK"],
            # Land the source holds no land for takes the column the
            # router builds (woof/ingest/soil.py: island_soil_columns).
            soil_no_source_land=getattr(
                initial_met, "soil_no_source_land", None),
            terrain=static["HGT_M"] if soil_orography is not None else None,
            source_orography=soil_orography,
            water_temperature=water_temperature,
            water_temperature_policy=case_policy["water_temperature_policy"],
            # GFS's 0.25 degree soil state is 5.5 by 9.1 cells wide on a
            # 3 km European domain, which is exactly the regime the
            # sub-source-cell reconstitution exists for.
            soil_mesh=soil_mesh_plan_from_case(
                snapshots[0], grid, experiment_config),
            # Where and when, for a same-model soil donor (woof/ingest/
            # land_state.py: HRRR's RUC soil for a RUC run over CONUS).
            land_state=land_state_context(grid, times[0]),
            route=_WATER_ROUTE)
        if len(exp.domains) > 1 and not chain_tree:
            # Every forcing time has been consumed; a single domain and a
            # chained tree prove this after their remaining times, below.
            verify_overlay_sequence(snapshots)
        initialize_seconds = time.perf_counter() - initialize_started

        from woof.ensemble.posted_preparation import bind_current_source_identity
        native_source_identity = bind_current_source_identity(native_source_identity,
            domain_id=exp.root.grid_id)
        staging = _atomic_staging_sibling(Path(output_root))
        if staging.exists():
            raise FileExistsError(f"stale GFS staging directory exists: {staging}")
        staging.mkdir(parents=True)
        writer = None
        try:
            if posted_series is None:
                # As posted, the seal writes these three from the batches
                # and the input manifest the seal authors.
                shutil.copy2(decoded / "gate.tsv",
                             staging / "decoder-gate.tsv")
                shutil.copy2(decoded / "inventory.tsv",
                             staging / "decoder-inventory.tsv")
                shutil.copy2(decoded / "decoded-sha256.tsv",
                             staging / "decoder-sha256.tsv")
            static_cache = staging / "native-static.npz"
            geometry_receipt = staging / "geometry-receipt.json"
            portable_source_manifest = staging / "source-input-manifest.json"
            prepared_cache = staging / "prepared-cache"
            wrf_output = staging / "wrf-native-input"
            if posted_series is None:
                shutil.copy2(Path(input_manifest), portable_source_manifest)
            _write_static_cache(static_cache, static)
            _write_geometry_receipt(geometry_receipt, grid, cfg, static_cache)
            if statics_corridor is not None and len(exp.domains) < 2:
                raise ValueError(
                    "--statics-corridor prepares child-resolution statics "
                    "over the ground a child domain can reach, and this "
                    "experiment has no "
                    "child domain; remove the flag or prepare a domain "
                    "tree")
            if len(exp.domains) > 1:
                selected_workers = hierarchy_workers
                if selected_workers is None:
                    selected_workers = (
                        8 if preprocess_receipt["backend"] == "cpu" else 1)
                hierarchy_inputs = {
                    "input_manifest_sha256": manifest_digest,
                    "decoder_sha256": decoder_digest,
                    "preprocessing": preprocess_receipt,
                }
                stock_wrf_export_mode = (
                    "optional" if stock_wrf_export else "off")

                def tree_proof_head(*, static_catalog, source_coverage,
                                    topology, moisture_floors,
                                    statics_corridor=None):
                    # The tree's proof without its seal-only keys
                    # (boundary_stream.SEAL_ONLY_PROOF_KEYS): the whole
                    # proof of a chained tree's head, and the one-shot
                    # tree's proof less its artifact records, export and
                    # wall times, so the two arms cannot drift apart.
                    return {
                        "schema": HIERARCHY_PROOF_SCHEMA,
                        "status": "READY_NOT_YET_STOCK_WRF_GATED",
                        "domain_count": len(exp.domains),
                        "physics": physics_selection,
                        "initial_condition": provenance,
                        # The coordinate this bundle was built on and why,
                        # on every run: the forecast adopts it from these
                        # artifacts, so a reader must be able to see it
                        # here without the configuration in hand.
                        "vertical_coordinate": _vertical_coordinate_receipt(
                            exp, vertical_adaptation),
                        "source_forecast_hours": list(source_hours),
                        "forcing_times": [
                            value.isoformat() for value in times],
                        "forcing_hours": hours,
                        "boundary_interval_seconds": (
                            boundary_interval_seconds),
                        "input_manifest_sha256": manifest_digest,
                        "decoder_sha256": decoder_digest,
                        "implementation_sha256": implementation_sha256,
                        "git_source_identity": git_source_identity,
                        "preprocessing": preprocess_receipt,
                        "root_static_provider": root_static_provider,
                        "root_static_receipt": root_static_receipt,
                        "hierarchy_workers": selected_workers,
                        "source_coverage": coverage_receipt,
                        # The soil-state SOURCE resolution, on every run: a
                        # reader of this forecast must be able to answer
                        # "how coarse was the soil I started from" without
                        # the config, and whether the sub-source-cell
                        # reconstitution ran on it.
                        "soil_texture_downscale": dict(
                            getattr(soil, "soil_texture_downscale", {})
                            or {}),
                        "decoder_stdout": (None if completed is None
                                           else completed.stdout.strip()),
                        "static_catalog": dict(static_catalog),
                        "hierarchy_source_coverage": dict(source_coverage),
                        "hierarchy_topology": dict(topology),
                        # WHETHER EACH DOMAIN'S INITIALIZATION MODIFIED
                        # VAPOUR ON THE WAY IN, root and children alike.
                        # Unconditional, and stated even when no floor
                        # fired: an absent key would read as "prepared
                        # before the receipt existed", a different claim
                        # and one no reader of the bundle could check.
                        **dict(moisture_floors),
                        **({"initial_perturbation": initial_perturbation}
                           if initial_perturbation is not None else {}),
                        # Present only when the preparation opted in: the
                        # statics-corridor set, digest-bound here the way
                        # every other sealed artifact is.  Absent, the
                        # bundle is byte-for-byte what it always was.  A
                        # chained tree builds it into its head, so a
                        # moving nest's forecast starts there.
                        **({"statics_corridor": dict(statics_corridor)}
                           if statics_corridor is not None else {}),
                    }

                def inputs_unchanged(sealed_inputs=None):
                    # As posted, the manifest the seal wrote.
                    bound, digest = (
                        (manifest, manifest_digest) if sealed_inputs is None
                        else (sealed_inputs["manifest"],
                              sealed_inputs["manifest_sha256"]))
                    final_manifest = _verify_input_manifest(
                        Path(input_manifest), digest, roles)
                    if final_manifest != bound:
                        raise ValueError(
                            "GFS input manifest content changed during "
                            "preparation")
                    if _implementation_sha256() != implementation_sha256:
                        raise ValueError(
                            "GFS adapter implementation changed during "
                            "preparation")

                if chain_tree:
                    posted_tree = None
                    if posted_series is not None:
                        def seal_posted_inputs(tree_root):
                            return _seal_as_posted_inputs(
                                posted_series, plan=plan,
                                series=Path(series), bridge=Path(bridge),
                                wps_namelist=Path(wps_namelist),
                                experiment_config=Path(experiment_config),
                                static_input=static_input,
                                static_receipt=static_receipt,
                                input_manifest=Path(input_manifest),
                                tree=tree_root,
                                merged=Path(temporary) / "decoded",
                                portable=portable_source_manifest.name)

                        posted_tree = SimpleNamespace(
                            posted=posted, series=posted_series, plan=plan,
                            source_hours=tuple(source_hours),
                            coverage_of=coverage_of,
                            head_decode_seconds=head_decode_seconds,
                            seal_inputs=seal_posted_inputs,
                            manifest_path=portable_source_manifest.name)
                    chained_tree = SimpleNamespace(
                        posted_tree=posted_tree,
                        exp=exp, grids=grids, cfg=cfg, snapshots=snapshots,
                        times=times, hours=hours,
                        build_forcing_time=build_forcing_time,
                        forcing=forcing, preprocess=preprocess,
                        preprocess_receipt=preprocess_receipt,
                        initial_result=initial_result,
                        initial_met=initial_met, soil=soil, static=static,
                        static_highres=static_highres,
                        statics_corridor=statics_corridor,
                        case_policy=case_policy,
                        lake_ice_provider=lake_ice_provider,
                        experiment_config=experiment_config,
                        wps_namelist=wps_namelist, geog_root=geog_root,
                        cpu_bridge=cpu_preprocess_bridge,
                        selected_workers=selected_workers,
                        manifest_digest=manifest_digest,
                        namelist_sha256=experiment_config_digest,
                        source_identity=native_source_identity,
                        input_provenance=hierarchy_inputs,
                        stock_wrf_export=stock_wrf_export_mode,
                        staging=staging, output_root=Path(output_root),
                        progress=progress,
                        tree_proof_head=tree_proof_head,
                        verify_overlay_sequence=verify_overlay_sequence,
                        inputs_unchanged=inputs_unchanged,
                        timings={
                            "verify_inputs": verify_inputs_seconds,
                            "static_build": static_build_seconds,
                            "decode": decode_seconds,
                        },
                        initialize_seconds=initialize_seconds,
                        total_started=total_started,
                    )
                    # The chained tree owns the start state from here and
                    # releases it at its head; these names would keep it
                    # resident.
                    del initial_result, initial_met
                    return _prepare_chained_gfs_tree(chained_tree)
                hierarchy_started = time.perf_counter()
                hierarchy = initialize_and_export_regular_source_hierarchy(
                    nudging_context=((output_root, build_forcing_time)
                        if getattr(exp, 'nudging', None) is not None else None),
                    exp=exp, grids=grids, snapshots=snapshots,
                    forcing_hours=hours,
                    wps_namelist=Path(wps_namelist),
                    geog_root=Path(geog_root), source_name="GFS",
                    artifact_output=staging / "hierarchy-artifacts",
                    wrf_output=staging / "wrf-native-input",
                    root_initial_result=initial_result, root_met=initial_met,
                    root_soil=soil, root_static_fields=static,
                    root_boundaries=boundaries,
                    bridge_manifest_sha256=manifest_digest,
                    source_manifest_sha256=manifest_digest,
                    namelist_sha256=experiment_config_digest,
                    source_identity=native_source_identity,
                    source_inventory=tuple(snapshots[0].fields),
                    workers=selected_workers,
                    preprocess_backend=preprocess_receipt["backend"],
                    cpu_bridge=cpu_preprocess_bridge,
                    input_provenance=hierarchy_inputs,
                    artifact_manifest_reference=(
                        "../hierarchy-artifacts/domain-artifacts.json"),
                    stock_wrf_export=stock_wrf_export_mode,
                    statics_corridor=statics_corridor,
                    static_highres=static_highres,
                    sfcp_to_sfcp=case_policy["sfcp_to_sfcp"],
                    water_temperature_policy=case_policy["water_temperature_policy"],
                    # The root's mapping conventions, so a child's RH2 and
                    # lakes are mapped as the root's are.
                    relative_humidity_convention="gfs_mixed",
                    lake_whole_source_search=(
                        case_policy["gfs_lake_skin"] == "wps"),
                    # The root's observed lake-ice analysis, sampled on each
                    # child's own grid (woof.ingest.lake_ice).
                    lake_ice_provider=lake_ice_provider,
                    # A child sees the catalog, not the config.  Without
                    # this a WRF-comparison reproduction would disable the
                    # soil reconstitution on the parent and silently keep
                    # it on every nest.
                    soil_texture_downscale=declared_soil_texture_downscale(
                        experiment_config),
                )
                hierarchy_seconds = time.perf_counter() - hierarchy_started
                verify_overlay_sequence(snapshots)
                inputs_unchanged()
                proof = {
                    **tree_proof_head(
                        static_catalog=hierarchy.static_catalog_receipt,
                        source_coverage=hierarchy.source_coverage_receipt,
                        topology=hierarchy.topology_receipt,
                        moisture_floors=(
                            hierarchy.hierarchy.moisture_floor_receipts),
                        statics_corridor=hierarchy.statics_corridor_receipt),
                    "artifact_receipt": dict(
                        hierarchy.hierarchy.artifacts.receipt),
                    "wrf_manifest": dict(
                        hierarchy.hierarchy.wrf_manifest),
                    "timing_seconds": {
                        "verify_inputs": verify_inputs_seconds,
                        "static_build": static_build_seconds,
                        "decode": decode_seconds,
                        "initialize_all_root_times": initialize_seconds,
                        **dict(hierarchy.hierarchy.timings_seconds),
                        "hierarchy_call_wall": hierarchy_seconds,
                        "total": time.perf_counter() - total_started,
                    },
                }
                (staging / "proof.json").write_text(
                    json.dumps(proof, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8")
                os.replace(staging, Path(output_root))
                return proof
            static_cache_sha256 = _sha256(static_cache)
            identity = prepared_cache_identity(
                bridge_manifest_sha256=manifest_digest,
                source_manifest_sha256=manifest_digest,
                static_cache_sha256=static_cache_sha256,
                namelist_sha256=experiment_config_digest,
                domain_config=exp.root,
                forcing_hours=hours,
                source_identity=native_source_identity,
            )
            writer = PreparedTreeWriter(
                staging=staging, output_root=Path(output_root),
                identity=identity)
            preprocessing_receipt_sha256 = hashlib.sha256(json.dumps(
                preprocess_receipt, sort_keys=True, separators=(",", ":"),
                allow_nan=False).encode("utf-8")).hexdigest()
            # Everything the proof says that the start time already knows.
            # The seal adds the artifact records, the cache receipt, the
            # export and the wall times (SEAL_ONLY_PROOF_KEYS) and refuses
            # a proof that differs from this head anywhere else.
            proof_head = {
                "schema": PROOF_SCHEMA,
                "status": "READY_NOT_YET_STOCK_WRF_GATED",
                # Cycle, lead and start, all three, in the document a
                # forecast run is pinned to.  ``forcing_hours`` below
                # are MODEL offsets from start_time;
                # ``source_forecast_hours`` are the NOAA leads they came
                # from, and the two are equal exactly when the lead is 0.
                "initial_condition": provenance,
                # The coordinate this bundle was built on and why, on
                # every run: the forecast adopts it from these artifacts,
                # so a reader must be able to see it here without the
                # configuration in hand.
                "vertical_coordinate": _vertical_coordinate_receipt(
                    exp, vertical_adaptation),
                "source_forecast_hours": list(source_hours),
                "forcing_times": [value.isoformat() for value in times],
                "forcing_hours": hours,
                "boundary_interval_seconds": boundary_interval_seconds,
                "input_manifest_sha256": manifest_digest,
                "decoder_sha256": decoder_digest,
                "implementation_sha256": implementation_sha256,
                "git_source_identity": git_source_identity,
                "preprocessing": preprocess_receipt,
                "preprocessing_receipt_sha256": preprocessing_receipt_sha256,
                "source_inputs": {
                    "manifest_schema": manifest["schema"],
                    "manifest_sha256": manifest_digest,
                    "files": manifest["files"],
                },
                # WHETHER THIS INITIALIZATION MODIFIED VAPOUR ON THE WAY
                # IN.  Unconditional, and stated even when no floor fired.
                **moisture_floor_proof_entry(
                    initial_result,
                    when_unrecorded=(
                        "this preparation's initialization result carries "
                        "no moisture-floor field, so it came from an "
                        "ingest predating the receipt; re-prepare the "
                        "case to record whether its vapour was floored "
                        "on the way in")),
                "source_coverage": coverage_receipt,
                # The soil-state SOURCE resolution, on every run: a reader of
                # this forecast must be able to answer "how coarse was the
                # soil I started from" without the config, and whether the
                # sub-source-cell reconstitution ran on it.
                "soil_texture_downscale": dict(
                    getattr(soil, "soil_texture_downscale", {}) or {}),
                "decoder_stdout": (None if completed is None
                                   else completed.stdout.strip()),
                "physics": physics_selection,
                # Whether the unchanged-WRF files were asked for: the
                # forecast reads this beside the export slot, which says
                # whether they were written (READY), declined
                # (NOT_REQUESTED) or refused by name (REFUSED).
                "stock_wrf_export": ("optional" if stock_wrf_export
                                     else "off"),
            }
            as_posted_head = None
            if posted_series is not None:
                # The keys that read every lead are the seal's; the head
                # binds the plan and the markers of the leads its own
                # decode read, and each segment the markers of its two
                # times (source_hours from the start lead).
                for key in _AS_POSTED_SEAL_KEYS:
                    proof_head.pop(key, None)
                as_posted_head = {
                    "input_plan": plan,
                    "start_markers": {
                        hour: posted_series.markers[hour]
                        for hour in posted_series.snapshots},
                    "forcing_leads": list(source_hours),
                    "seal_authored_proof_keys": _AS_POSTED_SEAL_KEYS,
                    "manifest_path": portable_source_manifest.name,
                    "lead_role_prefix": _AS_POSTED_LEAD_ROLE_PREFIX,
                    "derived_roles": _AS_POSTED_DERIVED_ROLES,
                    "manifest_bound_identity_keys": (
                        "bridge_manifest_sha256", "source_manifest_sha256",
                        "input_manifest_sha256"),
                }
            # One machine, one card: the forecast may start beside this
            # producer only when both fit (boundary_stream.chained_admission).
            writer.admit(
                experiment=exp, backend=str(preprocess_receipt["backend"]),
                device_bytes=producer_device_bytes(str(preprocess_receipt["backend"])),
                urban_columns=prepared_head_urban_columns(exp, static))
            progress.enter("write_prepared_cache")
            cache_started = time.perf_counter()
            writer.write_head(
                initial_result=initial_result, met=initial_met,
                surface=_canonical_surface(soil),
                metadata={
                    "source_adapter": "gfs",
                    "initial_valid_time": times[0].isoformat(),
                    "last_valid_time": times[-1].isoformat(),
                    "forcing_hours": hours,
                    "boundary_interval_seconds": boundary_interval_seconds,
                    "preprocessing": preprocess_identity(preprocess_receipt),
                },
                lbc={
                    "spec_bdy_width": cfg.spec_bdy_width,
                    "spec_zone": cfg.spec_zone,
                    "relax_zone": cfg.relax_zone,
                    "schedule": [
                        [float(earlier * 3600), float(later * 3600)]
                        for earlier, later in zip(hours, hours[1:])],
                    "fields": forcing.inventory,
                },
                proof_head=proof_head,
                input_manifest_sha256=(None if posted_series is not None
                                       else manifest_digest),
                forcing=forcing,
                as_posted=as_posted_head,
            )
            if posted is not None:
                # The lead waits from here on say so on the producer
                # heartbeat, so a forecast at a seam says it waits on the
                # source (PostedLeads.wait), and each segment binds the
                # markers its two times were decoded from.
                posted.writer = writer
                writer.bind_posted_leads(posted_series.markers)
            cache_seconds = time.perf_counter() - cache_started
            # The start state has done its work: it is in the head.
            del initial_result, initial_met
            release_backend_memory(preprocess)
            progress.enter("initialize_all_times",
                           forcing_times=len(records))
            boundaries_started = time.perf_counter()
            writer.stream_forcing_times(
                count=len(snapshots), build_forcing_time=build_forcing_time,
                forcing=forcing, times=times,
                release=lambda: release_backend_memory(preprocess))
            verify_overlay_sequence(snapshots)
            initialize_seconds += time.perf_counter() - boundaries_started
            if physical_output is not None:
                physical_output.document["source"]["preprocessing"] = preprocess.receipt()
                physical_output.seal()
            if physical_stream is not None:
                physical_stream.seal()
            progress.enter("write_prepared_cache")
            cache_started = time.perf_counter()
            sealed = None
            if posted_series is not None:
                # Every lead is decoded now (the last build read it), so
                # the decode time spent inside the builds moves back to
                # the decode.
                later_decode = posted_series.decode_seconds - head_decode_seconds
                decode_seconds += later_decode
                initialize_seconds -= later_decode
                sealed = _seal_as_posted_inputs(
                    posted_series, plan=plan, series=Path(series),
                    bridge=Path(bridge), wps_namelist=Path(wps_namelist),
                    experiment_config=Path(experiment_config),
                    static_input=static_input, static_receipt=static_receipt,
                    input_manifest=Path(input_manifest), tree=writer.root,
                    merged=Path(temporary) / "decoded",
                    portable=portable_source_manifest.name)
                manifest = sealed["manifest"]
                manifest_digest = sealed["manifest_sha256"]
                native_source_identity = {
                    **native_source_identity,
                    "input_manifest_sha256": manifest_digest}
                coverage_receipt = coverage_of(
                    (hour, posted_series.snapshots[hour])
                    for hour in source_hours)
                writer.write_posted_leads(
                    posted_series.markers,
                    route_table_sha256=sealed["route_table_sha256"])
                cache_receipt = writer.seal_cache(
                    manifest_sha256=manifest_digest,
                    identity=prepared_cache_identity(
                        bridge_manifest_sha256=manifest_digest,
                        source_manifest_sha256=manifest_digest,
                        static_cache_sha256=static_cache_sha256,
                        namelist_sha256=experiment_config_digest,
                        domain_config=exp.root,
                        forcing_hours=hours,
                        source_identity=native_source_identity,
                    ))
            else:
                cache_receipt = writer.seal_cache()
            cache_seconds += time.perf_counter() - cache_started
            portable_cache_receipt = dict(cache_receipt)
            portable_cache_receipt["path"] = prepared_cache.name
            # Where the tree lives now: published at the head when
            # chained, still staged otherwise.
            static_cache = writer.root / static_cache.name
            geometry_receipt = writer.root / geometry_receipt.name
            portable_source_manifest = (
                writer.root / portable_source_manifest.name)
            prepared_cache = writer.cache_path
            wrf_output = writer.root / wrf_output.name
            progress.enter("direct_wrf_export")
            export_started = time.perf_counter()
            export_receipt, _refusal = _single_domain_stock_export(
                prepared_cache, static_cache, geometry_receipt, wrf_output,
                valid_time=times[0],
                boundary_interval_seconds=boundary_interval_seconds,
                physics_selection=physics_selection,
                stock_wrf_export=stock_wrf_export)
            export_seconds = time.perf_counter() - export_started
            verify_overlay_sequence(snapshots)
            final_manifest = _verify_input_manifest(
                Path(input_manifest), manifest_digest, roles)
            if final_manifest != manifest:
                raise ValueError(
                    "GFS input manifest content changed during preparation")
            if _implementation_sha256() != implementation_sha256:
                raise ValueError(
                    "GFS adapter implementation changed during preparation")
            initialization_artifacts = {
                "source_manifest": {
                    "path": portable_source_manifest.name,
                    "bytes": portable_source_manifest.stat().st_size,
                    "sha256": _sha256(portable_source_manifest),
                },
                "static_cache": {
                    "path": static_cache.name,
                    "bytes": static_cache.stat().st_size,
                    "sha256": _sha256(static_cache),
                },
                "geometry_receipt": {
                    "path": geometry_receipt.name,
                    "bytes": geometry_receipt.stat().st_size,
                    "sha256": _sha256(geometry_receipt),
                },
                "prepared_cache": {
                    "path": prepared_cache.name,
                    "content_sha256": portable_cache_receipt["content_sha256"],
                    "payload_bytes": portable_cache_receipt["payload_bytes"],
                },
                "wrf_files": {
                    name: {
                        "path": f"{wrf_output.name}/{name}",
                        **details,
                    }
                    for name, details in (
                        export_receipt.get("files") or {}).items()
                },
            }
            if sealed is not None:
                # The keys the head left to the seal, as the one-shot
                # proof carries them, and the record of the waits.
                proof_head = {
                    **proof_head,
                    "input_manifest_sha256": manifest_digest,
                    "source_inputs": {
                        "manifest_schema": manifest["schema"],
                        "manifest_sha256": manifest_digest,
                        "files": manifest["files"],
                    },
                    "source_coverage": coverage_receipt,
                    "decoder_stdout": sealed["decoder_stdout"],
                    "posting": {
                        "as_posted": True,
                        "waits": list(posted.waits),
                        "leads_late": [],
                    },
                }
            proof = {
                **proof_head,
                "initialization_artifacts": initialization_artifacts,
                "prepared_cache": portable_cache_receipt,
                "export": export_receipt,
                "boundary_stream": writer.boundary_stream_proof(),
                # Every phase this stage passes through, and a `total`
                # its children account for.  `woof.stage_timing` states
                # the rule and the acceptance test holds this document
                # to it: an unnamed phase is time no reader can find.
                "timing_seconds": {
                    "verify_inputs": verify_inputs_seconds,
                    "static_build": static_build_seconds,
                    "decode": decode_seconds,
                    "initialize_all_times": initialize_seconds,
                    "write_prepared_cache": cache_seconds,
                    "direct_wrf_export": export_seconds,
                    "total": time.perf_counter() - total_started,
                },
            }
            progress.enter("publish")
            writer.publish(proof)
            return proof
        except BaseException as error:
            shutil.rmtree(staging, ignore_errors=True)
            if writer is not None:
                # A head already published stays, marked failed, so a
                # waiting forecast ends with this reason and the next
                # preparation of this output root rebuilds it.
                writer.fail(error)
            raise


def _prepare_chained_gfs_tree(c) -> dict[str, object]:
    """A GFS domain tree, chained: head, one root interval per segment, seal.

    The head holds the root's static files and start state (streamed cache
    under ``hierarchy-head/domains/d01/prepared-cache``, its header written
    at the seal) and every child's complete artifact set under
    ``hierarchy-head/domains/dNN``: a child, a delayed one included, needs
    only its start-time snapshot and the root's start state
    (:func:`woof.native_hierarchy.initialize_native_hierarchy_children`).
    A tree whose nest moves has its statics corridor built into
    ``hierarchy-head/statics-corridor`` and bound by the head's proof: it
    needs no boundary time, and a forecast started on the head moves the
    nest over it.  Segment k is the root's boundary interval k.  No start state stays
    resident between head and seal, on either backend: the seal re-reads
    the root (with its whole boundary set) and every child from the head
    (:class:`woof.ingest.boundary_stream.TreeStartStates`) and writes the
    one-shot ``hierarchy-artifacts/`` tree from them, through the same
    writer and the same arguments as the unchained tree, then the
    companion WRF files and the corridor (copied from the head), then
    ``proof.json``.  The top-level static cache, geometry receipt, source manifest and decoder receipts
    were staged before the head and are published with it.

    As posted (``c.posted_tree``, DESIGN A136 2.4 and the L3 design
    ruling), the head is written from the start lead (and a delayed
    nest's start lead, which its child's preparation waits for) before
    the input manifest exists: the root's and every child's identity bind
    the input plan's placeholder where the manifest digest goes, the head
    binds the plan and the markers its own decode read, and each segment
    the markers of its two times.  The seal writes the input manifest and
    the decoder receipts from the lead batches, seals the root cache under
    the one-shot identity, writes the one-shot ``hierarchy-artifacts/``
    tree with the manifest's digest, holds every sealed child to its head
    twin (:func:`woof.ingest.boundary_stream.verify_as_posted_tree_children`)
    and adds the proof keys that read every lead.  A moving or cyclone
    nest adds no start need and no lead: its corridor needs no boundary
    time and is built into the head as before.
    """

    exp = c.exp
    cfg = c.cfg
    backend = str(c.preprocess_receipt["backend"])
    manifest_digest = c.manifest_digest
    initialize_seconds = c.initialize_seconds
    posted = c.posted_tree
    writer = None
    try:
        c.progress.enter("write_prepared_cache")
        hierarchy_started = time.perf_counter()
        # The same head arguments the one-shot tree hands
        # initialize_and_export_regular_source_hierarchy.
        tree_head = prepare_regular_source_hierarchy_head(
            exp=exp, grids=c.grids, snapshots=c.snapshots,
            forcing_hours=c.hours,
            wps_namelist=Path(c.wps_namelist), geog_root=Path(c.geog_root),
            source_name="GFS", root_initial_result=c.initial_result,
            source_inventory=tuple(c.snapshots[0].fields),
            workers=c.selected_workers, preprocess_backend=backend,
            cpu_bridge=c.cpu_bridge,
            preprocess_selection=c.preprocess_receipt.get("selection"),
            source_manifest_sha256=manifest_digest,
            statics_corridor=c.statics_corridor,
            static_highres=c.static_highres,
            sfcp_to_sfcp=c.case_policy["sfcp_to_sfcp"],
            water_temperature_policy=c.case_policy[
                "water_temperature_policy"],
            relative_humidity_convention="gfs_mixed",
            lake_whole_source_search=(
                c.case_policy["gfs_lake_skin"] == "wps"),
            # The root's observed lake-ice analysis, sampled on each
            # child's own grid (woof.ingest.lake_ice).
            lake_ice_provider=c.lake_ice_provider,
            # A child sees the catalog, not the config.
            soil_texture_downscale=declared_soil_texture_downscale(
                c.experiment_config),
            # The statics corridor goes into the head, so a nest that
            # moves from the first step has its ground there.
            head_artifacts=c.staging / HIERARCHY_HEAD_DIRNAME,
        )
        bound_identity = tree_head.bound_source_identity(c.source_identity)
        head_domains = c.staging / HIERARCHY_HEAD_DIRNAME / "domains"
        root_directory = head_domains / "d01"
        root_directory.mkdir(parents=True)
        root_static_receipt, _root_geometry = write_domain_static_files(
            root_directory, domain=exp.domains[0], grid=c.grids[0],
            static_fields=c.static)
        child_builds = write_child_domain_artifacts(
            head_domains, exp=exp, child_results=tree_head.child_results,
            bridge_manifest_sha256=manifest_digest,
            source_manifest_sha256=manifest_digest,
            namelist_sha256=c.namelist_sha256, **tree_head.forcing_identity,
            source_identity=bound_identity, valid_time=exp.start_time)
        binding = root_domain_artifact_binding(
            exp=exp, static_cache_sha256=root_static_receipt["sha256"],
            bridge_manifest_sha256=manifest_digest,
            source_manifest_sha256=manifest_digest,
            namelist_sha256=c.namelist_sha256, **tree_head.forcing_identity,
            source_identity=bound_identity, valid_time=exp.start_time)
        head_hierarchy_seconds = time.perf_counter() - hierarchy_started
        cache_name = f"{HIERARCHY_HEAD_DIRNAME}/domains/d01/prepared-cache"
        writer = PreparedTreeWriter(
            staging=c.staging, output_root=c.output_root,
            identity=binding.identity, cache_name=cache_name)
        head_moisture_floor_receipts = hierarchy_moisture_floor_receipts(
            exp, c.initial_result, tree_head.child_results)
        proof_head = c.tree_proof_head(
            static_catalog=tree_head.static_receipt,
            source_coverage=tree_head.source_coverage_receipt,
            topology=tree_head.topology_receipt,
            moisture_floors=head_moisture_floor_receipts,
            statics_corridor=tree_head.statics_corridor_receipt)
        as_posted_head = None
        if posted is not None:
            # The keys that read every lead are the seal's; the head binds
            # the plan and the markers of the leads its own decode read
            # (the start lead, and a delayed nest's), and each segment the
            # markers of its two times.
            for key in _AS_POSTED_TREE_SEAL_KEYS:
                proof_head.pop(key, None)
            as_posted_head = {
                "input_plan": posted.plan,
                "start_markers": {
                    hour: posted.series.markers[hour]
                    for hour in posted.series.snapshots},
                "forcing_leads": list(posted.source_hours),
                "seal_authored_proof_keys": _AS_POSTED_TREE_SEAL_KEYS,
                "manifest_path": posted.manifest_path,
                "lead_role_prefix": _AS_POSTED_LEAD_ROLE_PREFIX,
                "derived_roles": _AS_POSTED_DERIVED_ROLES,
                "manifest_bound_identity_keys": (
                    "bridge_manifest_sha256", "source_manifest_sha256",
                    "input_manifest_sha256"),
            }
        # One machine, one card: the forecast may start beside this
        # producer only when both fit (boundary_stream.chained_admission).
        writer.admit(
            experiment=exp, backend=backend,
            device_bytes=producer_device_bytes(backend),
            urban_columns=prepared_head_urban_columns(
                exp, c.static, child_results=tree_head.child_results))
        head_started = time.perf_counter()
        writer.write_head(
            initial_result=c.initial_result, met=c.initial_met,
            surface=_canonical_surface(c.soil),
            metadata=dict(binding.metadata),
            lbc={
                "spec_bdy_width": cfg.spec_bdy_width,
                "spec_zone": cfg.spec_zone,
                "relax_zone": cfg.relax_zone,
                "schedule": [
                    [float(earlier * 3600), float(later * 3600)]
                    for earlier, later in zip(c.hours, c.hours[1:])],
                "fields": c.forcing.inventory,
            },
            proof_head=proof_head,
            input_manifest_sha256=(None if posted is not None
                                   else manifest_digest),
            forcing=c.forcing,
            as_posted=as_posted_head,
            tree=domain_tree_head_fields(
                [f"d{int(domain.grid_id):02d}" for domain in exp.domains],
                root_cache=cache_name,
                children_receipts={
                    f"d{int(build.receipt['grid_id']):02d}": _sha256(
                        head_domains
                        / f"d{int(build.receipt['grid_id']):02d}"
                        / "receipt.json")
                    for build in child_builds}),
            extra_head_payload_bytes=sum(
                int(build.receipt["artifacts"]["prepared_cache"][
                    "payload_bytes"]) for build in child_builds),
        )
        head_seconds = time.perf_counter() - head_started
        if posted is not None:
            # The lead waits from here on say so on the producer heartbeat,
            # and each segment binds the markers its two times were decoded
            # from.
            posted.posted.writer = writer
            writer.bind_posted_leads(posted.series.markers)
        # Every start state is in the head now: the root's in its streamed
        # cache, each child's in hierarchy-head/domains/dNN.  None stays
        # resident while the later times are built (on the card that is the
        # residency start_last_forcing_order avoids); the seal re-reads them.
        start_states = TreeStartStates.release(
            root_result=c.initial_result, root_met=c.initial_met,
            child_results=tree_head.child_results,
            child_content_sha256={
                f"d{int(build.receipt['grid_id']):02d}": build.receipt[
                    "artifacts"]["prepared_cache"]["content_sha256"]
                for build in child_builds})
        c.initial_result = c.initial_met = None
        tree_head.child_results = ()
        release_backend_memory(c.preprocess)
        c.progress.enter("initialize_all_times",
                         forcing_times=len(c.snapshots))
        boundaries_started = time.perf_counter()
        writer.stream_forcing_times(
            count=len(c.snapshots), build_forcing_time=c.build_forcing_time,
            forcing=c.forcing, times=c.times,
            release=lambda: release_backend_memory(c.preprocess))
        # Every forcing time has been consumed.
        c.verify_overlay_sequence(c.snapshots)
        initialize_seconds += time.perf_counter() - boundaries_started
        c.progress.enter("write_prepared_cache")
        timings = dict(c.timings)
        source_identity = c.source_identity
        input_provenance = c.input_provenance
        root_identity = binding.identity
        sealed = None
        if posted is None:
            root_content_sha256 = str(writer.seal_cache()["content_sha256"])
        else:
            # Every lead is decoded now (the last build read it), so the
            # decode time spent inside the builds moves back to the decode.
            later_decode = (posted.series.decode_seconds
                            - posted.head_decode_seconds)
            timings["decode"] = timings["decode"] + later_decode
            initialize_seconds -= later_decode
            sealed = posted.seal_inputs(writer.root)
            manifest_digest = sealed["manifest_sha256"]
            source_identity = {**c.source_identity,
                               "input_manifest_sha256": manifest_digest}
            input_provenance = {**c.input_provenance,
                                "input_manifest_sha256": manifest_digest}
            writer.write_posted_leads(
                posted.series.markers,
                route_table_sha256=sealed["route_table_sha256"])
            # The root's one-shot identity: the head's binding with the
            # manifest's digest where the plan's placeholder stood.
            root_identity = root_domain_artifact_binding(
                exp=exp, static_cache_sha256=root_static_receipt["sha256"],
                bridge_manifest_sha256=manifest_digest,
                source_manifest_sha256=manifest_digest,
                namelist_sha256=c.namelist_sha256,
                **tree_head.forcing_identity,
                source_identity=tree_head.bound_source_identity(
                    source_identity),
                valid_time=exp.start_time).identity
            root_content_sha256 = str(writer.seal_cache(
                identity=root_identity,
                manifest_sha256=manifest_digest)["content_sha256"])
        c.progress.enter("direct_wrf_export")
        hierarchy_seal_started = time.perf_counter()
        (root_result, root_met, boundaries,
         tree_head.child_results) = start_states.reread(
            writer.root, exp=exp, grids=c.grids,
            root_identity=root_identity,
            root_content_sha256=root_content_sha256)
        hierarchy_result = seal_regular_source_hierarchy(
            tree_head,
            artifact_output=writer.root / "hierarchy-artifacts",
            wrf_output=writer.root / "wrf-native-input",
            root_initial_result=root_result, root_met=root_met,
            root_soil=c.soil, root_static_fields=c.static,
            root_boundaries=boundaries,
            bridge_manifest_sha256=manifest_digest,
            source_manifest_sha256=manifest_digest,
            namelist_sha256=c.namelist_sha256,
            source_identity=source_identity,
            input_provenance=input_provenance,
            artifact_manifest_reference=(
                "../hierarchy-artifacts/domain-artifacts.json"),
            stock_wrf_export=c.stock_wrf_export,
            head_artifacts=writer.root / HIERARCHY_HEAD_DIRNAME,
        )
        hierarchy_seconds = (head_hierarchy_seconds
                             + time.perf_counter() - hierarchy_seal_started)
        if dict(hierarchy_result.hierarchy.moisture_floor_receipts) \
                != head_moisture_floor_receipts:
            raise RuntimeError(
                "the sealed tree's moisture-floor receipts differ from the "
                "head's, which were computed from the same initial states")
        # Named only as posted, so every other tree seals with the call it
        # always made.
        start_states.require_sealed_is_head(
            hierarchy_result.hierarchy.artifacts.receipt,
            root_content_sha256=root_content_sha256,
            **({} if posted is None else {"as_posted": {
                "root": writer.root, "head": writer.head,
                "manifest_sha256": manifest_digest}}))
        c.verify_overlay_sequence(c.snapshots)
        c.inputs_unchanged(sealed)
        if sealed is not None:
            # The keys the head left to the seal, as the one-shot proof
            # carries them, and the record of the waits.
            proof_head = {
                **proof_head,
                "input_manifest_sha256": manifest_digest,
                "source_coverage": posted.coverage_of(
                    (hour, posted.series.snapshots[hour])
                    for hour in posted.source_hours),
                "decoder_stdout": sealed["decoder_stdout"],
                "posting": {
                    "as_posted": True,
                    "waits": list(posted.posted.waits),
                    "leads_late": [],
                },
            }
        proof = {
            **proof_head,
            "artifact_receipt": dict(
                hierarchy_result.hierarchy.artifacts.receipt),
            "wrf_manifest": dict(hierarchy_result.hierarchy.wrf_manifest),
            # Present only when the preparation opted in, as in the
            # unchained tree; the head's copy, which publish holds equal
            # to the one the head's proof binds.
            **({"statics_corridor": dict(
                hierarchy_result.statics_corridor_receipt)}
               if hierarchy_result.statics_corridor_receipt is not None
               else {}),
            "boundary_stream": writer.boundary_stream_proof(),
            "timing_seconds": {
                **timings,
                "initialize_all_root_times": initialize_seconds,
                **dict(hierarchy_result.hierarchy.timings_seconds),
                "hierarchy_call_wall": hierarchy_seconds,
                "prepared_head": head_seconds,
                "head_published_after": writer.head_seconds,
                "total": time.perf_counter() - c.total_started,
            },
        }
        c.progress.enter("publish")
        writer.publish(proof)
    except BaseException as error:
        if writer is not None:
            # A head already published stays, marked failed, so a waiting
            # forecast ends with this reason and the next preparation of
            # this output root rebuilds it.  The caller removes a staging
            # tree that was never published.
            writer.fail(error)
        raise
    return proof


#: The export receipt schema of the single-domain route.
SINGLE_DOMAIN_EXPORT_SCHEMA = "gpuwm-native-direct-wrf-export-v3"


def _single_domain_stock_export(prepared_cache, static_cache,
                                geometry_receipt, wrf_output, *, valid_time,
                                boundary_interval_seconds: int,
                                physics_selection: Mapping[str, object],
                                stock_wrf_export: bool = True,
                                ) -> tuple[dict[str, object],
                                           StockWrfExportUnsupported | None]:
    """The single-domain route's stock-WRF export slot and any refusal.

    Not requested (``stock_wrf_export`` False), nothing is attempted and
    the slot says NOT_REQUESTED.  Requested, the export runs, but a
    wrfinput for an unchanged WRF cannot represent every preparation
    (WDM6, Kessler and Milbrandt-Yau have no stock package contract),
    which says nothing about running it here: the forecast restores the
    prepared cache, not these files.  Such a slot records the refusal by
    name, as the mapped and HRRR routes write it for the same case, and
    the proof's ``stock_wrf_export = "optional"`` beside it is what the
    forecast reader admits.  An export that runs keeps its READY receipt
    byte for byte, and must carry exactly the selected physics.
    """

    if not stock_wrf_export:
        return (stock_wrf_export_not_requested(
            schema=SINGLE_DOMAIN_EXPORT_SCHEMA), None)
    try:
        receipt = export_prepared_wrf(
            prepared_cache, static_cache, geometry_receipt, wrf_output,
            valid_time=valid_time,
            boundary_interval_seconds=boundary_interval_seconds,
            # The RESOLVED selection, not the caller's argument.  Named,
            # the exporter re-validates the same profile the front door
            # checked.  Unnamed (owner ruling 2026-07-31), the exporter
            # recomputes the experiment-config suite selection from its
            # OWN cache-bound config -- the profileless contract -- and
            # the equality check below proves both spellings agree byte
            # for byte.
            physics_profile=physics_selection["profile"],
            experiment_config_suite=physics_selection["profile"] is None,
            expert_acknowledgements=tuple(
                physics_selection["acknowledgements"]),
            acknowledgement_provenance=physics_selection[
                "acknowledgement_provenance"])
    except StockWrfExportUnsupported as error:
        shutil.rmtree(wrf_output, ignore_errors=True)
        return (stock_wrf_export_refused(
            error, schema=SINGLE_DOMAIN_EXPORT_SCHEMA), error)
    if (receipt.get("schema") != SINGLE_DOMAIN_EXPORT_SCHEMA
            or receipt.get("physics") != physics_selection):
        raise RuntimeError(
            "direct-WRF export physics provenance differs from the "
            "selected preparation profile")
    return receipt, None


def _arg(value) -> str:
    """One printed argument, quoted if a shell would split it.

    A perfectly valid ``--outdir`` or config path with a space in it was
    printed bare, so the command whose entire purpose is to be pasted
    broke the moment it was.  POSIX display form, exactly as
    ``source_cli._quote_command`` renders ``--dry-run`` argv, because
    the certified runtime is Linux/CUDA and forward slashes are accepted
    by every path API on Windows too.  Ordinary paths come back
    unquoted, so this is invisible except where it matters.
    """

    return shlex.quote(str(value).replace("\\", "/"))


def stock_wrf_export_notice(proof: Mapping[str, object]) -> list[str]:
    """Say plainly when the bonus stock-WRF export did not happen.

    The forecast preparation succeeded either way -- that is the point of
    the layering -- but a user who expected ``wrf-native-input/`` has to
    be told it is not there and why, in the same sentence form every
    other refusal on this door uses.  A READY export prints nothing.
    """

    # A domain tree's slot is its hierarchy manifest; a single domain's
    # is the export receipt.  Both carry the same three statuses.
    export = proof.get("wrf_manifest")
    if not isinstance(export, Mapping):
        export = proof.get("export")
    if not isinstance(export, Mapping):
        return []
    status = export.get("status")
    if status in (None, "READY"):
        return []
    if status == "NOT_REQUESTED":
        return ["", "rw-wps --source gfs: no stock-WRF export was requested; "
                    "the prepared forecast is complete."]
    return [
        "",
        "rw-wps --source gfs: the prepared forecast is complete, but the "
        f"bonus stock-WRF export was refused: {export.get('reason')}.",
        "  The prepared forecast itself is unaffected -- run the "
        "forecast command below.",
    ]


def prepared_forecast_next_command(
        proof: Mapping[str, object], *, output_root, experiment_config,
        wps_namelist, source: str = "gfs") -> list[str]:
    """The ready-to-run forecast command, hashes filled in.

    ``woof fetch --author-front-door-manifest`` already prints the
    complete ``rw-wps`` line with its digest filled in, and it is the
    single most-praised thing in the pilots.  Nothing did the equivalent
    here: the front door finished by dumping 42 KB of proof.json to
    stdout and printing no next step at all, so every user hand-extracted
    three SHA-256 values -- one of them only findable by grepping the
    JSON for ``content_sha256`` -- before they could start a forecast.

    Everything below is already known to this process.  Emitted on
    stderr so the proof document on stdout stays machine-readable.

    **Every line printed as a command must run when pasted, exactly as
    printed.**  v1.0.0 printed ``--physics-profile <the profile this
    config was materialized for>`` and ``--outdir OUTPUT_DIR``: two
    placeholders in a command whose entire value was that a user did not
    have to reconstruct it, and the second one is a shell metacharacter
    error rather than an accurate gap.  Both are resolved here -- the
    profile by asking the same table the runner's guard asks, the outdir
    by naming a real directory beside the preparation.  Where a value
    genuinely cannot be resolved, this prints prose saying so instead of
    a command that cannot be run.

    When the proof carries a ``gpuwm-front-door-physics-selection-v1``
    receipt (proof v3 and later), that receipt IS what was materialized,
    so it names the profile.  Re-deriving it from the config would be a
    second opinion about a decision already recorded.  The config-table
    lookup stays as the fallback for the historical v2 proof, which
    carries no physics receipt.

    ``source`` names the prepared-forecast runner's own ``--source``
    value.  Every native front door that reaches a prepared bundle can
    print this, and one of them printing a complete hash-bound command
    while another printed nothing at all was a test host H finding in its own
    right: the 20CRv3 route ran end to end and then said nothing about
    how to run the forecast.
    """

    from woof.experiment import load_experiment
    from woof.physics_compat import identify_single_domain_profile

    output_root = Path(output_root)
    proof_path = output_root / "proof.json"
    lines = [
        "",
        "rw-wps: preparation complete.  Run the forecast with:",
    ]
    try:
        proof_digest = _sha256(proof_path)
    except OSError:
        return lines + [
            f"  (cannot read {proof_path}; no next command to print)"]
    # A real, pasteable directory: a sibling of the preparation, so the
    # command works from anywhere and does not overwrite the inputs.
    #
    # It has to be a genuine SIBLING, not a child.  Both runners declare
    # --prepared-root a protected input and refuse any --outdir that
    # overlaps it in either direction, so the <root>/forecast this used
    # to print was a command the runner rejected -- which is how a test host H
    # pilot got a traceback out of the front door's own suggestion.
    outdir = output_root.parent / f"{output_root.name}-forecast"
    cache = proof.get("prepared_cache")
    content = (cache.get("content_sha256")
               if isinstance(cache, Mapping) else None)
    manifest_digest = proof.get("input_manifest_sha256")
    if not isinstance(manifest_digest, str):
        # The mapped single-domain proof records the published manifest's
        # digest inside its composition receipt instead of at the top
        # level, and it is the SAME digest: the copy this command points
        # the runner at, `source-evidence/input-manifest.json`, is
        # written from those bytes and verified against this value at
        # publication.  Without the fallback a perfectly ordinary
        # single-domain mapped bundle -- every packaged source runs on
        # one -- looked manifest-less here and got the multi-domain
        # hierarchy command printed for it, which is the wrong runner.
        composition = proof.get("source_composition")
        record = (composition.get("input_manifest")
                  if isinstance(composition, Mapping) else None)
        if isinstance(record, Mapping) and isinstance(
                record.get("sha256"), str):
            manifest_digest = record["sha256"]
    if not isinstance(content, str) or not isinstance(manifest_digest, str):
        # The multi-domain hierarchy product.  Its runner binds the
        # experiment config by digest too, so the full command is
        # printable here rather than a fragment the user completes.
        config_path = Path(experiment_config)
        try:
            config_digest = _sha256(config_path)
        except OSError:
            return lines + [
                f"  (cannot read {config_path}; no next command to print)"]
        return lines + [
            "  python -m woof.prepared_domain_tree_forecast \\",
            f"      --prepared-root {_arg(output_root)} \\",
            f"      --preparation-receipt-sha256 {proof_digest} \\",
            f"      --experiment-config {_arg(config_path)} \\",
            f"      --experiment-config-sha256 {config_digest} \\",
            f"      --io-mode history --outdir {_arg(outdir)}",
            "  (this proof carries no single prepared-cache identity "
            "because it is a multi-domain hierarchy product.)",
        ]
    physics = proof.get("physics")
    profile = physics.get("profile") if isinstance(physics, Mapping) else None
    if not isinstance(profile, str):
        try:
            profile = identify_single_domain_profile(
                load_experiment(Path(experiment_config)).root.run)
        except Exception:  # a config this process cannot load is not a command
            profile = None
    profile_line = (
        [] if profile is None else [f"      --physics-profile {profile} \\"])
    lines += [
        "  python -m woof.prepared_single_domain_forecast \\",
        f"      --source {source} \\",
        f"      --prepared-root {_arg(output_root)} \\",
        f"      --proof-sha256 {proof_digest} \\",
        f"      --source-manifest-sha256 {manifest_digest} \\",
        f"      --prepared-content-sha256 {content} \\",
        f"      --experiment-config {_arg(experiment_config)} \\",
        f"      --wps-namelist {_arg(wps_namelist)} \\",
        *profile_line,
        f"      --io-mode history --outdir {_arg(outdir)}",
        "  (--run-seconds and --history-interval-seconds default to the "
        "hash-bound experiment; the runner refuses any other value.)",
    ]
    if profile is None:
        # One sentence, and the command above runs (owner posture
        # 2026-07-31: verification status is stated, never gating).
        lines.append(
            "  (physics: this config's suite is supported, not yet "
            "WRF-verified; the runner executes it as written.)")
    return lines


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--series", type=Path, required=True)
    parser.add_argument("--cycle", required=True)
    parser.add_argument("--bridge", type=Path,
                        help="source decoder; posted members reuse their pinned producer decoder authority")
    parser.add_argument("--wps-namelist", type=Path, required=True)
    parser.add_argument("--static-input", type=Path)
    parser.add_argument("--static-receipt", type=Path)
    parser.add_argument("--experiment-config", type=Path, required=True)
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--input-manifest-sha256")
    parser.add_argument(
        "--as-posted", type=Path, default=None, metavar="POSTING_DIR",
        help="prepare as an as-posted fetch publishes the window's leads: "
             "POSTING_DIR is its posting/ folder; the seal writes "
             "--input-manifest, which then takes no digest")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--physical-input-store", type=Path,
                        help="sealed native physical snapshots for the complete forcing window")
    parser.add_argument("--physical-output-store", type=Path,
                        help="capture mapped native physical snapshots before real initialization")
    parser.add_argument("--physical-input-provider", type=Path,
                        help="posted native physical member provider directory")
    parser.add_argument("--physical-member-index", type=int,
                        help="original recipe member index for a posted physical provider")
    parser.add_argument(
        "--preprocess-backend", choices=("cuda", "cpu", "auto"),
        default="auto",
        help="where the source-grid/WRF-real setup transforms run "
             "(default auto: CUDA when the certified runtime is usable, "
             "otherwise the deterministic parallel CPU backend, "
             "announced in one line)")
    # Why a caller's configuration policy named the backend; recorded in
    # the receipt's selection block in place of "named by the caller".
    parser.add_argument("--preprocess-backend-reason",
                        help=argparse.SUPPRESS)
    parser.add_argument("--preprocess-workers", type=int)
    parser.add_argument("--cpu-preprocess-bridge", type=Path)
    parser.add_argument("--geog-root", type=Path)
    parser.add_argument("--hierarchy-workers", type=int)
    parser.add_argument(
        "--physics-profile", default=None,
        help="optional assertion that the experiment IS this registered "
             "fixed physics template, refused on any switch drift; "
             "omitted, the suite the config selects runs as written on "
             "both routes and its WRF-verification status is reported, "
             "never gating")
    parser.add_argument(
        "--ack", action="append", default=[],
        help="registry-owned expert acknowledgement id; repeat as needed")
    parser.add_argument(
        "--no-stock-wrf-export", dest="stock_wrf_export",
        action="store_false", default=True,
        help="prepare the forecast only, and do not attempt the bonus "
             "unchanged-WRF wrfinput/wrfbdy export")
    parser.add_argument(
        "--statics-corridor", nargs="?", const="all", default=None,
        metavar="GRID_IDS",
        help="also seal child-resolution statics over the ground each "
             "child can reach (the moving-nest corridor); bare flag covers "
             "every child domain, or pass comma-separated child grid ids "
             "(e.g. 2,3).  Required before the prepared tree runner will "
             "honor a [relocation] follow source; omitted, the bundle is "
             "byte-for-byte unchanged")
    return parser


@owns_source_coverage_refusal
def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if (args.as_posted is None) == (args.input_manifest_sha256 is None):
        print("rw-wps --source gfs: pass --input-manifest-sha256 for a "
              "fetched window, or --as-posted POSTING_DIR for one the seal "
              "binds, not both and not neither.", file=sys.stderr)
        return 2
    statics_corridor = args.statics_corridor
    if statics_corridor is not None and statics_corridor != "all":
        try:
            statics_corridor = tuple(
                int(part) for part in statics_corridor.split(",") if part)
        except ValueError:
            print("rw-wps --source gfs: --statics-corridor accepts 'all' "
                  f"or comma-separated child grid ids, got "
                  f"{args.statics_corridor!r}.", file=sys.stderr)
            return 2
    try:
        proof = prepare_gfs_wrf(
            series=args.series, cycle=args.cycle, bridge=args.bridge,
            wps_namelist=args.wps_namelist, static_input=args.static_input,
            static_receipt=args.static_receipt,
            experiment_config=args.experiment_config,
            input_manifest=args.input_manifest,
            input_manifest_sha256=args.input_manifest_sha256,
            output_root=args.output_root,
            preprocess_backend=args.preprocess_backend,
            preprocess_backend_reason=args.preprocess_backend_reason,
            preprocess_workers=args.preprocess_workers,
            cpu_preprocess_bridge=args.cpu_preprocess_bridge,
            geog_root=args.geog_root,
            hierarchy_workers=args.hierarchy_workers,
            physics_profile=args.physics_profile,
            expert_acknowledgements=tuple(args.ack),
            stock_wrf_export=args.stock_wrf_export,
            statics_corridor=statics_corridor,
            as_posted=args.as_posted,
            physical_input_store=args.physical_input_store,
            physical_output_store=args.physical_output_store,
            physical_input_provider=args.physical_input_provider,
            physical_member_index=args.physical_member_index,
        )
    except PreparationRefusal:
        # The decorator on this main owns the whole refusal family: two
        # lines, the message and ITS remedy.  Catching it here as a
        # plain ValueError flattened the remedy away -- the eta-ladder
        # refusal reached the walk as one sentence with no door named
        # (UX finding R2) -- so it is re-raised to the owner.
        raise
    except InitializationMemoryRefused as error:
        # A preparation the card cannot hold under an explicit
        # --preprocess-backend cuda (PreparationDeviceRefused), or any
        # other measured initialization budget that refuses the case, is
        # a refusal with its own remedy line: one message at exit 2, as
        # `woof` itself answers it, not a traceback that buries the
        # remedy at the bottom of a stack.
        print(f"rw-wps --source gfs: {error}", file=sys.stderr)
        return 2
    except (ValueError, OSError) as error:
        # Every gate this door applies is a refusal, and a refusal is a
        # sentence.  A pilot met three of these as stack traces -- a
        # missing input, a manifest bound to a different namelist, and
        # the mis-scoped physics whitelist this release fixes -- and a
        # traceback tells nobody which of their inputs to change.  The
        # gates themselves are untouched: only how they reach the reader
        # is.  RC 2 is what the 20CRv3 door already costs for the same
        # class of refusal, and `rw-wps` passes it straight through.
        print(f"rw-wps --source gfs: {error}.", file=sys.stderr)
        return 2
    print(json.dumps(proof, indent=2, sort_keys=True))
    corridor = proof.get("statics_corridor")
    if isinstance(corridor, dict):
        # Size accuracy at the door: the corridor's cost, and the ground
        # it covers, are stated where they are paid.
        from woof.static.corridor import corridor_summary_line
        for label, entry in sorted(corridor.get("domains", {}).items()):
            print(corridor_summary_line(label, entry), file=sys.stderr)
    for line in stock_wrf_export_notice(proof):
        print(line, file=sys.stderr)
    for line in prepared_forecast_next_command(
            proof, output_root=args.output_root,
            experiment_config=args.experiment_config,
            wps_namelist=args.wps_namelist):
        print(line, file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
