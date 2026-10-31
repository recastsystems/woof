"""ERA5 GRIB1 -> prepared cache -> stock-WRF direct-export proof adapter.

This adapter uses woof's existing all-Rust GRIB1 decoder and GPU real-data
initialization.  WPS and ``real.exe`` are never invoked.  It remains an
uncertified proof lane until an unchanged stock ``wrf.exe`` accepts the
products emitted by this module.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import sys
import time
import tomllib
from types import SimpleNamespace
from typing import Mapping

import numpy as np

from woof import netcdf_bridge

from woof.case_data import (
    CaseDataConfig,
    PerDomainSourceOrography,
    SourceOrography,
    SourceOrographyDeclaration,
    load_experiment_case_bytes,
    resolve_source_orography,
)
from woof.config import soil_layer_count
from woof.experiment import ExperimentConfig, load_experiment
from woof.explain import warn
from woof.ingest.grib import (
    cached_era5_snapshots,
    canonical_units,
    parse_vtable,
)
from woof.ingest.horiz import interpolate_era5_to_lambert
from woof.ingest.source_coverage import (ForcingSeriesRefusal,
                                          owns_source_coverage_refusal)
from woof.ingest.lateral_bc import (
    StateBoundaryFrames,
    attach_lateral_boundaries,
    start_last_forcing_order,
)
from woof.ingest.prepared_cache import (
    CONDITIONAL_PREPARATION_RECEIPTS,
    prepared_cache_identity,
)
from woof.ingest.boundary_stream import (
    SEALED_REASONS,
    PreparedTreeWriter,
    prepared_head_urban_columns,
    producer_device_bytes,
    remove_unfinished_tree,
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
from woof.ingest.cg_topo import RootTerrainBlend
from woof.ingest.soil import soil_source_orography
from woof.ingest.soil_downscale import (
    declared_soil_texture_downscale, soil_mesh_plan_from_case)
from woof.ingest.water_overlay import (
    load_bound_water_overlay, overlay_snapshot_sequence, verify_overlay_sequence,
)
from woof.moisture_floor_receipt import moisture_floor_proof_entry
from woof.native_domain_artifacts import (
    _atomic_staging_sibling, published_path_refusal)
from woof.native_wrf_contract import (
    NATIVE_STATIC_REQUIRED,
    canonical_noah_surface,
    load_native_static_cache,
    require_land_terrain,
    validate_native_lambert_contract,
    validate_native_lambert_contracts,
    validate_native_static_fields,
    verify_native_static_receipt,
    write_native_geometry_receipt,
    write_native_static_cache,
)
from woof.source_hierarchy import (
    initialize_and_export_regular_source_hierarchy,
)
from woof.hrrr_native_static import verified_static_catalog
from woof.static.build import (build_static_for_domain,
                                geog_selection_from_catalog)
from woof.static.terrain_smoothing import selection_carrier_kwargs
from woof.wrf_direct import export_prepared_wrf
from woof.vertical_contract import validate_explicit_eta_grid
from woof.vertical_adaptation import (
    adapt_experiment_for_statics,
    vertical_coordinate_receipt as _vertical_coordinate_receipt,
)


INPUT_MANIFEST_SCHEMA = "gpuwm-era5-direct-input-manifest-v1"
_STATIC_REQUIRED = NATIVE_STATIC_REQUIRED


def _domain_source_orography(
        bindings: list[str] | None, variable: str,
) -> PerDomainSourceOrography | None:
    """Parse repeatable ``dNN=PATH`` hierarchy bindings fail-closed."""

    if not bindings:
        return None
    if not variable:
        raise ValueError("source-orography variable must be non-empty")
    parsed: list[tuple[int, SourceOrography]] = []
    seen: set[int] = set()
    for binding in bindings:
        label, separator, raw_path = binding.partition("=")
        if (not separator or len(label) < 2 or label[0].lower() != "d"
                or not label[1:].isdigit() or not raw_path):
            raise ValueError(
                "domain source-orography bindings must use dNN=PATH")
        domain_id = int(label[1:])
        if domain_id < 1:
            raise ValueError("domain source-orography ids must be positive")
        if domain_id in seen:
            raise ValueError(
                f"duplicate domain source-orography binding for d{domain_id:02d}")
        seen.add(domain_id)
        parsed.append((domain_id, SourceOrography(Path(raw_path), variable)))
    parsed.sort(key=lambda item: item[0])
    return PerDomainSourceOrography(tuple(parsed))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical(value: object) -> str:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"),
        ensure_ascii=True, allow_nan=False)


def _verify_input_manifest(
    manifest_path: Path,
    expected_manifest_sha256: str,
    role_paths: Mapping[str, Path],
) -> dict[str, object]:
    observed_manifest_sha256 = _sha256(manifest_path)
    if observed_manifest_sha256 != expected_manifest_sha256.lower():
        raise ValueError(
            "ERA5 input manifest SHA mismatch: expected "
            f"{expected_manifest_sha256}, got {observed_manifest_sha256}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != INPUT_MANIFEST_SCHEMA:
        raise ValueError("unsupported ERA5 direct-input manifest schema")
    declared = manifest.get("files")
    if not isinstance(declared, dict) or set(declared) != set(role_paths):
        raise ValueError("ERA5 input manifest role inventory differs from request")
    for role, path in role_paths.items():
        spec = declared[role]
        if not isinstance(spec, dict) or spec.get("name") != path.name:
            raise ValueError(f"ERA5 input manifest path mismatch for {role}")
        observed = _sha256(path)
        if spec.get("sha256") != observed:
            raise ValueError(
                f"ERA5 input digest mismatch for {role}: "
                f"expected {spec.get('sha256')}, got {observed}")
    return manifest


def _validated_static(
        fields: Mapping[str, object], grid, ny: int, nx: int, *,
        land_terrain: bool = True,
) -> dict[str, np.ndarray]:
    return validate_native_static_fields(
        fields, grid, ny, nx, land_terrain=land_terrain)


from woof.ingest.water_temperature import (
    WaterTemperatureStatics,
    validate_water_temperature_policy,
)


#: The ERA5 adapter's name in every water-temperature refusal and
#: receipt.  Not a case name and not a file name: the ROUTE.
_WATER_ROUTE = "the ERA5 direct route"

#: Where each CONDITIONAL_PREPARATION_RECEIPTS metadata key is read from
#: on the soil result.  The two names differ for the moisture floor --
#: the metadata says ``soil_moisture_floor``, the soil object says
#: ``moisture_floor`` -- and only the METADATA spelling is shared with the
#: validator.  A key with no entry here is read under its own name.
_SOIL_RECEIPT_ATTRIBUTE = {
    "soil_moisture_floor": "moisture_floor",
}


def _soil_receipt_bindings(soil) -> dict[str, object]:
    """Every fired CONDITIONAL_PREPARATION_RECEIPTS receipt on ``soil``.

    Read under the Noah state's attribute name first
    (:data:`_SOIL_RECEIPT_ATTRIBUTE`), then under the metadata spelling,
    which is the RUC state's own (:class:`woof.ingest.ruc_soil.RucSoilState`
    ``soil_moisture_floor``, real.exe's layer-source ``MAX(SMOIS, 0.005)``).
    Reading the Noah name alone dropped a fired RUC floor from both the
    cache and the proof of an ERA5 run (2.8.8 acceptance D-02 sweep).
    """

    bindings = {}
    for key in CONDITIONAL_PREPARATION_RECEIPTS:
        attribute = _SOIL_RECEIPT_ATTRIBUTE.get(key, key)
        receipt = dict(getattr(soil, attribute, None)
                       or getattr(soil, key, None) or {})
        if receipt:
            bindings[key] = receipt
    return bindings


def _water_temperature_statics(static, landuse_attrs, policy):
    """The surface this route assembles water temperature over.

    Lakes come from the selected land-use table's own ``ISLAKE`` rather
    than a hard-coded MODIS 21, because the category number is a
    property of the table: USGS 24-category has no inland-water class at
    all.  ``WaterTemperatureStatics`` carries whether the table could
    name one, and the advisory says so when it could not.

    ``None`` when the land-use metadata is not reachable at all, which
    is the prebuilt-static-cache path with no geog root: the NPZ is
    numeric fields only.  The caller announces that and the assembly is
    skipped rather than run on a silently all-False lake mask.
    """
    if landuse_attrs is None:
        return None
    return WaterTemperatureStatics.for_route(
        route=_WATER_ROUTE, policy=policy,
        landmask=static["LANDMASK"], lu_index=static["LU_INDEX"],
        landuse_attrs=landuse_attrs)


def _announce_adaptation(sentence: str) -> None:
    """Say, once, that the run is not on the configured vertical coordinate."""

    warn(sentence,
         "WRF v4.6.1 dyn_em/nest_init_utils.F:1158-1182 calls a column "
         "the coordinate cannot order fatal and names reducing etac as "
         "the remedy, and a column it only just orders keeps one layer "
         "too thin to integrate, which a lower etac thickens; the etac is "
         "derived here from the terrain this run can actually touch and "
         "applied, so the prepared inputs, their receipt and the forecast "
         "all carry the same coordinate.  p_top is untouched.")


def _survey_static_catalog(exp, wps_namelist, geog_root, static_highres=None):
    """The WPS_GEOG catalog the terrain survey needs, or None."""

    if geog_root is None or len(exp.domains) < 2:
        return None
    catalog, _ = verified_static_catalog(
        Path(wps_namelist), Path(geog_root),
        [domain.grid_id for domain in exp.domains],
        **selection_carrier_kwargs(static_highres))
    return catalog


def _load_static(path: Path, grid, ny: int, nx: int) -> dict[str, np.ndarray]:
    """A prebuilt root static cache, validated but for its land height.

    The land-height check (:func:`woof.native_wrf_contract
    .require_land_terrain`) is the caller's: the ERA5 and GFS doors lay a
    declared ``[static.highres]`` terrain over these fields first, and it
    reads the terrain the run will integrate.
    """
    return load_native_static_cache(
        path, grid, ny, nx, land_terrain=False)


def _static_from_geog(
        wps_namelist: Path, geog_root: Path, grid, cfg, *, static_highres=None,
) -> tuple[dict[str, np.ndarray], dict[str, object], dict[str, object]]:
    """The root statics from WPS GEOG, validated but for their land height.

    The land-height check is the caller's, as for :func:`_load_static`.
    """
    # Only a carrier the selection reads (a non-default terrain smoothing,
    # or a land cover its block builds) rides on the catalog, so any other
    # root is built through the call it always made.
    catalog, receipt = verified_static_catalog(
        Path(wps_namelist), Path(geog_root), (1,),
        **selection_carrier_kwargs(static_highres))
    fields = build_static_for_domain(
        grid, catalog, 1,
        **({"cfg": cfg} if any(getattr(cfg, option, 0) for option in
            ("sf_lake_physics", "topo_wind", "gwd_opt")) else {}))
    # The land-use table's own ISLAKE/ISWATER, from the same GEOG index
    # the statics were built from, so the water-temperature assembly
    # and the statics cannot disagree about what a lake is.
    landuse_attrs = geog_selection_from_catalog(
        catalog, 1).landuse_global_attrs()
    return (_validated_static(fields, grid, cfg.ny, cfg.nx,
                              land_terrain=False),
            receipt, landuse_attrs)


def _write_static_cache(path: Path, fields: Mapping[str, np.ndarray]) -> None:
    write_native_static_cache(path, fields)


def _write_geometry_receipt(path: Path, grid, cfg, static_path: Path) -> None:
    write_native_geometry_receipt(path, grid, cfg, static_path)


def _canonical_surface(soil) -> dict[str, np.ndarray]:
    return canonical_noah_surface(soil)


def load_era5_adapter_config(
        experiment_config: Path,
) -> tuple[ExperimentConfig, CaseDataConfig | None]:
    """The ERA5 adapter's one config door (task #204).

    Both config shapes this source accurately has are accepted: the
    ONE-FILE config ``woof domain --source era5`` writes (experiment
    tables plus ``[case_data]``/``[fetch]``) and the bare experiment
    config.  Before this door existed the adapter loaded through
    :func:`woof.experiment.load_experiment` alone, which split off
    ``[fetch]`` but refused ``[case_data]`` -- so the wizard's own
    emission was refused by the wizard's own source adapter, with a
    message claiming the table it carried was absent.

    A present ``[case_data]`` is validated by its owning schema and
    RETURNED, so its declarations (source orography, geog root, water
    temperature) can default the matching adapter arguments the caller
    left unset.  Input existence is deliberately not required here
    (``require_inputs=False``): the adapter's binding inputs arrive as
    explicit manifest-bound arguments and are checked by the adapter
    itself, and a declaration pointing at data fetched to another path
    must not veto the run the explicit arguments fully describe.
    """
    from woof.config_authority import read_config_authority

    authority = read_config_authority(experiment_config)
    raw = tomllib.load(io.BytesIO(authority.payload))
    if "case_data" in raw:
        return load_experiment_case_bytes(
            authority.payload, source=str(authority.source),
            base_dir=Path(authority.source).parent, require_inputs=False)
    return load_experiment(experiment_config), None


def _load_source_orography(path: Path, variable: str) -> np.ndarray:
    # Decoded by the Rust bridge: this is a meteorological field read out
    # of a third-party NetCDF, which is decode work, not plumbing.
    with netcdf_bridge.open_dataset(path) as dataset:
        if variable not in dataset.variables:
            raise KeyError(f"source-orography file lacks {variable!r}")
        value = np.asarray(dataset.variables[variable][0], dtype=np.float64)
    if not np.isfinite(value).all():
        raise ValueError("source orography contains non-finite values")
    return value


def _soil_source_orography(declared, fields):
    """The terrain the SOURCE model's soil fields were defined on.

    WRF's ``adjust_soil_temp_new`` shifts land skin and soil temperature by
    ``-0.0065 * (HGT_M - toposoil)``, where ``toposoil`` is the source
    model's own orography -- for ERA5 that is ERA5's terrain, never the
    target ``HGT_M``.  This route accepts it two ways, and refuses both at
    once well upstream: an explicit ``--source-orography`` artifact, or the
    invariant SOILGEO record the combined GRIB carries, which
    :func:`woof.ingest.horiz.interpolate_era5_to_lambert` bilinearly remaps
    onto the mass grid and divides by WPS gravity as ``SOURCE_OROGRAPHY``.

    ``initialize_real`` already makes exactly this choice for itself
    (woof/ingest/real.py:2478-2484).  The soil hand-off never did, so a
    wizard case whose orography rides inside the GRIB -- which is what
    ``woof fetch --source era5`` writes and what the domain wizard emits --
    reached ``preprocess_noah_soil`` with ``terrain`` set and its partner
    ``None`` and died on that function's all-or-none guard.  The guard was
    right; the argument was missing.
    """
    resolved = soil_source_orography(declared, fields)
    if resolved is None:
        raise ValueError(
            "ERA5 soil preparation has no source orography: neither a "
            "declared source-orography artifact nor the SOURCE_OROGRAPHY "
            "field the invariant SOILGEO record resolves to is present, so "
            "the soil-temperature elevation adjustment has no source "
            "terrain to lapse from")
    return resolved


@preprocess_math_call
def prepare_era5_wrf(
    *,
    grib: Path,
    vtable: Path,
    bridge: Path,
    wps_namelist: Path,
    static_input: Path | None,
    static_receipt: Path | None,
    source_orography: Path | None,
    source_orography_variable: str,
    experiment_config: Path,
    input_manifest: Path,
    input_manifest_sha256: str,
    output_root: Path,
    preprocess_backend: str = "auto",
    preprocess_workers: int | None = None,
    cpu_preprocess_bridge: Path | None = None,
    geog_root: Path | None = None,
    hierarchy_workers: int | None = None,
    statics_corridor=None,
    hierarchy_source_orography: SourceOrographyDeclaration | None = None,
    water_temperature_overlay: Path | None = None,
    water_temperature_policy_declared: str | None = None,
    experiment_config_suite: bool = False,
) -> dict[str, object]:
    """Build native ERA5 initial/boundary files and return the proof receipt.

    Multi-domain internal callers must supply ``geog_root`` and a complete
    per-domain source-orography declaration.  The legacy d01 field is never
    reused on a child.
    """

    paths = {
        "grib": Path(grib),
        "vtable": Path(vtable),
        "bridge": Path(bridge),
        "wps_namelist": Path(wps_namelist),
        "experiment_config": Path(experiment_config),
    }
    if source_orography is not None:
        paths["source_orography"] = Path(source_orography)
    if water_temperature_overlay is not None:
        paths["water_temperature_overlay"] = Path(water_temperature_overlay)
    static_pair = static_input is not None or static_receipt is not None
    if static_pair and (static_input is None or static_receipt is None):
        raise ValueError(
            "ERA5 static-input and static-receipt must be supplied together")
    if static_input is not None:
        paths["static_input"] = Path(static_input)
        if not Path(static_receipt).is_file():
            raise FileNotFoundError(
                f"missing ERA5 adapter input static_receipt: {static_receipt}")
    # geog_root may still arrive from the config's own [case_data] below,
    # so its absence is judged AFTER the config is loaded, not here.
    for role, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"missing ERA5 adapter input {role}: {path}")
    if not os.access(paths["bridge"], os.X_OK):
        raise PermissionError("ERA5 Rust bridge is not executable")
    # A head published early whose producer failed, was stopped or went
    # silent is this tool's own unfinished product, not a finished run.
    remove_unfinished_tree(output_root)
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite {output_root}")
    # A domain tree is published through staging deeper than its
    # output root; refused here, before any GRIB is hashed or decoded.
    # The experiment is read only to count its domains, and only when
    # the limit binds: a single-domain bundle publishes no tree and is
    # not measured against it.  The load the preparation uses follows
    # the manifest verification below.
    refusal = published_path_refusal(output_root)
    if (refusal is not None and len(load_era5_adapter_config(
            paths["experiment_config"])[0].domains) > 1):
        raise ValueError(refusal)
    manifest = _verify_input_manifest(
        Path(input_manifest), input_manifest_sha256, paths)
    preprocess = resolve_preprocess_backend(
        preprocess_backend, workers=preprocess_workers,
        cpu_bridge=cpu_preprocess_bridge)
    preprocess_receipt = preprocess.receipt()

    total_started = time.perf_counter()
    exp, declared = load_era5_adapter_config(paths["experiment_config"])
    # THE FLOOR, BEFORE THE DECODE (A98): the domains alone bound the card
    # price from below, so an explicit cuda they cannot fit is refused
    # before the host decode, and auto moves to the CPU here.  The decoded
    # price below stays the binding check.
    preprocess = admit_preparation(
        preprocess, lambda: price_preparation_floor("era5", exp),
        workers=preprocess_workers)
    preprocess_receipt = preprocess.receipt()
    from woof.case_data import preparation_case_policy
    case_policy = preparation_case_policy(declared)
    from woof.static.highres_production import (
        load_static_highres, apply_prepared_highres, static_highres_identity)
    static_highres = load_static_highres(paths["experiment_config"])
    from woof.experiment import (
        deferred_initial_perturbation, refuse_unrouted_spawn,
    )
    refuse_unrouted_spawn(exp, "ERA5-direct prepared-cache")
    initial_perturbation = deferred_initial_perturbation(
        exp, "ERA5-direct prepared-cache")
    # A present [case_data] defaults the adapter inputs the caller left
    # unset, so the wizard's one-file config is self-sufficient.  An
    # explicit argument always wins, and an explicitly passed input stays
    # manifest-bound (it sits in ``paths``); a config-declared default is
    # bound through the experiment config's own manifest entry instead,
    # exactly like geog_root always was.
    if declared is not None:
        if source_orography is None and declared.source_orography is not None:
            orography_declared = declared.source_orography
            if isinstance(orography_declared, PerDomainSourceOrography):
                if hierarchy_source_orography is None:
                    hierarchy_source_orography = orography_declared
                root_orography = resolve_source_orography(
                    orography_declared, 1)
                source_orography = root_orography.path
                source_orography_variable = root_orography.variable
            else:
                source_orography = orography_declared.path
                source_orography_variable = orography_declared.variable
            if not Path(source_orography).is_file():
                raise FileNotFoundError(
                    f"source_orography {source_orography} declared in "
                    f"[case_data] of {paths['experiment_config']} does not "
                    "exist; stage the artifact at that path or pass "
                    "--source-orography explicitly.")
        if geog_root is None and declared.geog_root is not None:
            geog_root = declared.geog_root
        if (water_temperature_overlay is None
                and declared.water_temperature_overlay is not None):
            water_temperature_overlay = declared.water_temperature_overlay
            if not Path(water_temperature_overlay).is_file():
                raise FileNotFoundError(
                    f"water_temperature_overlay {water_temperature_overlay} "
                    f"declared in [case_data] of {paths['experiment_config']} "
                    "does not exist; stage the analysis at that path or pass "
                    "--water-temperature-overlay explicitly.")
        if (water_temperature_policy_declared is None
                and declared.water_temperature_policy is not None):
            water_temperature_policy_declared = (
                declared.water_temperature_policy)
    if static_input is None and geog_root is None:
        raise ValueError(
            "ERA5 requires prebuilt statics or a geography root: pass "
            "--static-input WITH --static-receipt, or --geog-root pointing "
            "at a WPS_GEOG tree.  The one-file config written by `woof "
            "domain --source era5` declares geog_root in [case_data] and "
            "needs no flag; this config carries no such declaration.")
    cfg = exp.root.run
    if len(exp.domains) == 1:
        grid = validate_native_lambert_contract(
            exp, paths["wps_namelist"], source_name="ERA5")
        grids = (grid,)
    else:
        if geog_root is None:
            raise ValueError(
                "nested ERA5 preparation requires an explicit geog_root")
        if hierarchy_source_orography is not None:
            if not isinstance(
                    hierarchy_source_orography, PerDomainSourceOrography):
                raise ValueError(
                    "nested ERA5 explicit source_orography requires one "
                    "artifact per domain")
            if source_orography is None:
                raise ValueError(
                    "per-domain ERA5 source_orography requires the matching "
                    "root source-orography input")
            root_orography = resolve_source_orography(
                hierarchy_source_orography, 1)
            if (Path(root_orography.path).resolve()
                    != Path(source_orography).resolve()
                    or root_orography.variable != source_orography_variable):
                raise ValueError(
                    "nested ERA5 d01 source_orography differs from the root "
                    "preparation input")
        grids = validate_native_lambert_contracts(
            exp, paths["wps_namelist"], source_name="ERA5")
        grid = grids[0]
    landuse_attrs = None
    if static_input is None:
        static, root_static_receipt, landuse_attrs = _static_from_geog(
            paths["wps_namelist"], Path(geog_root), grid, cfg,
            static_highres=static_highres)
        root_static_provider = "native-wps-geog"
    else:
        root_static_receipt = verify_native_static_receipt(
            Path(static_receipt), paths["static_input"], grid, cfg)
        static = _load_static(paths["static_input"], grid, cfg.ny, cfg.nx)
        root_static_provider = "prebuilt-hash-bound-cache"
        if geog_root is not None:
            catalog, _ = verified_static_catalog(
                paths["wps_namelist"], Path(geog_root), (1,),
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
    # THE COORDINATE, BEFORE ANYTHING IS BUILT ON IT.  The same call the
    # other source doors make, in the same place: root terrain in hand,
    # nothing yet built on a vertical coordinate.
    exp, vertical_adaptation = adapt_experiment_for_statics(
        exp, grids, root_terrain=static["HGT_M"],
        static_catalog=_survey_static_catalog(
            exp, paths["wps_namelist"], geog_root, static_highres),
        static_highres=static_highres, announce=_announce_adaptation,
        statics_corridor=statics_corridor)
    cfg = exp.root.run
    source_terrain = (
        None if source_orography is None
        else _load_source_orography(
            Path(source_orography), source_orography_variable)
    )

    water_temperature_policy = validate_water_temperature_policy(
        water_temperature_policy_declared
        if water_temperature_policy_declared is not None
        else ("external_overlay" if water_temperature_overlay is not None
              else None))
    water_statics = _water_temperature_statics(
        static, landuse_attrs, water_temperature_policy)
    if (water_statics is None and water_temperature_policy != "wrf_compat"
            and (water_temperature_policy_declared is not None
                 or water_temperature_overlay is not None)):
        raise ValueError("declared water-temperature policy needs the selected land-use "
                         "table's ISLAKE metadata; provide --geog-root with this static cache")
    if water_statics is None:
        # LOUD, and the coherent receipt is NOT printed: a prebuilt static
        # NPZ is numeric fields only, so with no geog root there is no
        # land-use table to read ISLAKE from, and the assembly would have to
        # guess a category number.  Run the historical selector, say so, and
        # name the one argument that restores the default treatment.
        water_temperature_policy = "wrf_compat"
        print(
            "water temperature: the prebuilt static cache carries no "
            "land-use metadata, so this run cannot tell an inland body from "
            "the ocean and falls back to the historical per-cell selector "
            "(lakes stay blocky).  Pass --geog-root, which is read only for "
            "the land-use index, to get the default class-coherent "
            "treatment.",
            file=sys.stderr)

    decode_started = time.perf_counter()
    snapshots = cached_era5_snapshots(
        paths["grib"], paths["vtable"], bridge=paths["bridge"])
    decode_seconds = time.perf_counter() - decode_started
    snapshots = tuple(sorted(snapshots, key=lambda value: value.valid_time))
    if not snapshots:
        raise ValueError(
            f"ERA5 decode produced no forcing snapshots from {paths['grib']}")
    # The run initializes at start_time EXACTLY.  A series that begins
    # earlier is data, not a defect: the leading snapshots are dropped
    # loudly and the run proceeds (a fetch window wider than the run used
    # to be refused as "must begin at").  A series that does not contain
    # the start hour at all refuses by naming both facts and both fixes.
    start_index = next(
        (index for index, snapshot in enumerate(snapshots)
         if snapshot.valid_time == exp.start_time), None)
    if start_index is None:
        first = snapshots[0].valid_time
        last = snapshots[-1].valid_time
        raise ValueError(
            "ERA5 forcing must contain the experiment start_time "
            f"{exp.start_time.isoformat()}, and the decoded series "
            f"[{first.isoformat()} .. {last.isoformat()}] does not.  "
            "Either fetch a window that covers the start hour (`woof "
            f"fetch --source era5 --cycle {exp.start_time:%Y-%m-%dT%H} "
            "--retrieve` downloads one that begins there), or set "
            "[experiment].start_time in "
            f"{paths['experiment_config']} to an hour the series carries.")
    if start_index:
        dropped = ", ".join(
            snapshot.valid_time.isoformat()
            for snapshot in snapshots[:start_index])
        print(
            "ERA5 forcing begins before the experiment start_time; "
            f"dropping {start_index} leading snapshot(s) ({dropped}) and "
            f"initializing at {exp.start_time.isoformat()}.",
            file=sys.stderr)
        snapshots = snapshots[start_index:]
    # Optional hi-res water-temperature overlay (task #71): applied to
    # the snapshots that feed BOTH the single-domain loop and the nested
    # hierarchy, before any horizontal interpolation.  Absent, this is
    # the identity: the same tuple object flows onward.
    water_overlay, water_file_binding = load_bound_water_overlay(water_temperature_overlay)
    snapshots = overlay_snapshot_sequence(
        snapshots, water_overlay, binding=water_file_binding,
        workers=host_step_workers(preprocess))
    source_inventory = tuple(snapshots[0].fields) if snapshots else ()
    source_units = dict(canonical_units(parse_vtable(paths["vtable"])))
    has_invariant_orography = (
        bool(snapshots)
        and all("SOILGEO" in snapshot.fields for snapshot in snapshots)
    )
    if source_terrain is None and not has_invariant_orography:
        raise ValueError(
            "ERA5 initialization needs the source model's own terrain, and "
            f"this run has none: the decoded series from {paths['grib']} "
            "carries no invariant SOILGEO record in every snapshot, no "
            "--source-orography artifact was passed, and the config "
            "declares no [case_data] source_orography.  Fix ONE of these: "
            "fetch the combined GRIB with the invariant geopotential "
            "included (`woof fetch --source era5 --retrieve` writes it into "
            "era5-combined.grib), pass --source-orography PATH "
            "(--source-orography-variable NAME, default SOILHGT), or "
            "declare source_orography + source_orography_variable in "
            "[case_data] of the experiment config.")
    if source_terrain is not None and any(
            "SOILGEO" in snapshot.fields for snapshot in snapshots):
        raise ValueError(
            "ERA5 input declares both invariant SOILGEO and explicit "
            "source-orography")
    source_orography_catalog = (
        SimpleNamespace(
            snapshots=snapshots,
            inventory=source_inventory,
            units=source_units,
        )
        if has_invariant_orography else None
    )
    times = tuple(value.valid_time for value in snapshots)
    if not times or times[0] != exp.start_time:
        raise ValueError(
            f"ERA5 forcing must begin at {exp.start_time}, got {times[:1]}")
    forcing_hours = [
        int((value - times[0]).total_seconds() // 3600) for value in times]
    if any(
            (value - times[0]).total_seconds() != hour * 3600
            for value, hour in zip(times, forcing_hours)):
        raise ValueError("ERA5 direct forcing offsets must be whole hours")
    if forcing_hours[-1] * 3600 < exp.run_seconds:
        raise ValueError("ERA5 forcing does not cover the configured run")
    if len(forcing_hours) < 2:
        raise ForcingSeriesRefusal(
            "ERA5 direct forcing requires at least two times")
    deltas = {
        later - earlier
        for earlier, later in zip(forcing_hours, forcing_hours[1:])}
    if len(deltas) != 1:
        raise ValueError("ERA5 direct forcing cadence must be uniform")
    boundary_interval_seconds = deltas.pop() * 3600

    source_top_pressure_pa = max(
        float(np.min(snapshot.levels_hpa) * 100.0)
        for snapshot in snapshots
    )
    if len(exp.domains) == 1:
        validate_explicit_eta_grid(
            exp.vertical.eta_levels,
            nz=cfg.nz,
            p_top=exp.vertical.p_top,
            source_top_pressure_pa=source_top_pressure_pa,
            context="ERA5 direct adapter",
        )
    else:
        grids = validate_native_lambert_contracts(
            exp, paths["wps_namelist"], source_name="ERA5",
            source_top_pressure_pa=source_top_pressure_pa)
        grid = grids[0]

    from woof.core.grid import make_vertical_coord

    # THE FIT, BEFORE THE FIRST DEVICE ALLOCATION: the decode above ran on
    # the host.  ERA5's humidity conversion also runs on the whole SOURCE
    # grid in FP64, which the price carries as its own phase.  A backend
    # already on the CPU is not priced.
    preprocess = admit_preparation(
        preprocess, lambda: price_forcing_preparation(
            "era5", exp, snapshots, fp64_humidity_transform=True),
        workers=preprocess_workers)
    preprocess_receipt = preprocess.receipt()
    initialize_started = time.perf_counter()
    # ONE forcing time is ever resident.  A single domain builds the start
    # time FIRST, publishes it in the prepared head and releases it, then
    # writes each boundary interval as soon as its two times exist
    # (woof.ingest.boundary_stream).  Otherwise the start time is built
    # LAST (start_last_forcing_order) and is the only met/state this loop
    # retains; every other time contributes its perimeter frames against
    # its own position and is released before the next one is
    # interpolated.  Walking the times in order instead meant holding the
    # start time -- which nothing reads until the boundaries are complete
    # -- while each later time was built underneath it.  At 800x800x49
    # with mp=10 and three GFS times that second resident time is 14.67
    # GiB of device residency against 7.66, a priced peak envelope of
    # 23.92 GiB against 15.86: the difference between preparing the
    # domain on a 16 GiB card and OOMing after the whole forcing chain
    # had already been fetched.
    initial_result = None
    initial_met = None
    forcing = StateBoundaryFrames(
        spec_bdy_width=cfg.spec_bdy_width,
        spec_zone=cfg.spec_zone, relax_zone=cfg.relax_zone)
    # WRF's smooth_cg_topo (woof.ingest.cg_topo): the root terrain is
    # blended toward ERA5's once, before the first initialization reads
    # it: the declared source-orography artifact when there is one, else
    # the invariant SOILGEO the snapshot carries.  Off, this does nothing.
    terrain_blend = RootTerrainBlend(exp, static, route="era5")
    # Inland water the source does not resolve takes the forcing's
    # daily-mean 2 m air temperature (WRF use_tavg_for_tsk).
    from woof.ingest.horiz import with_inland_air_temperature
    water_statics = with_inland_air_temperature(water_statics, snapshots)

    nudging_writer = None
    if getattr(exp, "nudging", None) is not None:
        from woof.spectral_nudging.prepared import analysis_writer
        nudging_writer = analysis_writer(output_root, exp, times, _sha256(Path(input_manifest)))

    def build_forcing_time(index):
        # One forcing time's build, unchanged.  A single domain calls it
        # start first (woof.ingest.boundary_stream).
        source = snapshots[index]
        met = interpolate_era5_to_lambert(
            source, grid,
            source_orography_catalog=source_orography_catalog,
            backend=preprocess,
            target_landmask=np.asarray(static["LANDMASK"]) >= 0.5,
            water_temperature_statics=water_statics)
        terrain_blend.before_initialize(
            source_terrain if source_terrain is not None
            else met.fields.get("SOURCE_OROGRAPHY"))
        coord = make_vertical_coord(
            cfg.nz, hybrid_opt=cfg.hybrid_opt, etac=cfg.etac,
            eta_levels=exp.vertical.eta_levels)
        initialized = initialize_real(
            met, cfg, coord, static["HGT_M"], grid=grid,
            landmask=static["LANDMASK"],
            source_orography=source_terrain,
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

    # START FIRST when the head can be published before the last time
    # exists.  A hierarchy needs every root interval before its
    # children, and a water-temperature overlay binds its receipt over
    # every forcing time into the cache identity: both build every time
    # first, start time last, as before.
    start_first = (len(exp.domains) == 1
                   and water_temperature_overlay is None)
    if not start_first:
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
        initial_met, initial_result = build_forcing_time(0)
        forcing.add_state(initial_result.state, index=0)
        boundaries = None
    # The chem processes' preparation-time arrays (dust statics, sulfur
    # lat/lon), before the head is published; nothing on a chem-off state
    # (woof/core/chem_statics.py).
    from woof.core.chem_statics import attach_chem_statics
    attach_chem_statics(initial_result.state, grid, geog_root, None)

    # No lake skin override: metgrid's masked=both SKINTEMP chain with
    # static-landmask targets already yields water-source skin at lakes,
    # matching real.exe's no-TAVGSFC behavior (SST only where valid), except
    # inland water with no source water within INLAND_WATER_SOURCE_REACH_M,
    # which takes the daily-mean 2 m air temperature (use_tavg_for_tsk).  The
    # router forwards this exact argument list to preprocess_noah_soil for
    # Noah-geometry schemes, so their soil state is unchanged by the LSM
    # dispatch seam.
    from woof.core.landuse import (
        ruc_fractional_seaice as _ruc_fractional_seaice,
        seaice_threshold_of as _seaice_threshold_of)
    soil = preprocess_land_surface_soil(
        initial_met.fields,
        # real.exe's adjust_for_seaice_pre/post keep the fraction under
        # fractional_seaice = 1 (threshold 0.02) and snap to 0/1 at 0.5
        # otherwise (module_soil_pre.F:216-219, :337-343, :392-393 of the HRRR
        # v4.1.21 fork).
        fractional_seaice=_ruc_fractional_seaice(cfg),
        seaice_threshold=_seaice_threshold_of(cfg),
        sf_surface_physics=int(cfg.sf_surface_physics),
        # Resolved, not defaulted: see woof/ingest/hrrr_physics.py for the
        # failure this closes.  Inert at every geometry but RUC's six.
        num_soil_layers=soil_layer_count(cfg),
        soil_type=static["SCT_DOM"],
        deep_soil_temperature=static["TMN"],
        landmask=static["LANDMASK"],
        # Land the source holds no land for takes the column the
        # router builds (woof/ingest/soil.py: island_soil_columns).
        soil_no_source_land=getattr(
            initial_met, "soil_no_source_land", None),
        terrain=static["HGT_M"],
        # Declared artifact XOR the GRIB's own invariant SOILGEO, resolved
        # the same way initialize_real resolves it a few lines above.
        source_orography=_soil_source_orography(
            source_terrain, initial_met.fields),
        # getattr on every route, so a stand-in snapshot is refused by
        # the router on its merits rather than by an AttributeError.
        water_temperature=getattr(
            initial_met, "water_temperature", None),
        water_temperature_policy=water_temperature_policy,
        # The moisture and deep-temperature analogue of the elevation
        # lapse two arguments up: the target grid knows things about the
        # land surface a 0.25 degree forcing mesh cannot carry.
        soil_mesh=soil_mesh_plan_from_case(snapshots[0], grid, declared),
        # Where and when, for a same-model soil donor (woof/ingest/
        # land_state.py: HRRR's RUC soil for a RUC run over CONUS).
        land_state=land_state_context(grid, times[0]),
        route=_WATER_ROUTE)
    # LOUD when it happened, silent when it did not: the SMCDRY floor
    # receipt is only bound into the prepared metadata/proof when at
    # least one land cell was floored (getattr: the RUC state carries
    # its own moisture and has no Noah floor receipt).
    #
    # Same discipline for the deep-soil repair: bound into the prepared
    # metadata only when at least one LAND cell took TSK for its TMN.
    # The water half of that substitution is every run's ordinary path
    # and never appears here.
    #
    # Both are built from CONDITIONAL_PREPARATION_RECEIPTS, the same tuple
    # the prepared-cache validator binds from the proof.  Keeping one list
    # is the point: when these two sides disagreed, the front door refused
    # the cache it had just written.
    soil_floor_binding = _soil_receipt_bindings(soil)
    initialize_seconds = time.perf_counter() - initialize_started
    input_manifest_digest = _sha256(Path(input_manifest))
    # LOUD when configured, absent when not: the overlay binding joins
    # the source identity only when an overlay was applied, so the
    # prepared cache identity moves exactly when the data does.
    water_overlay_receipt = verify_overlay_sequence(snapshots)
    water_overlay_binding = ({} if water_overlay_receipt is None else
                             {"water_temperature_overlay": water_overlay_receipt})
    native_source_identity = {
        "adapter": "era5-grib1-direct-v1",
        **({"static_highres": static_highres_identity(static_highres)}
           if static_highres is not None else {}),
        "preparation_case_policy": {**case_policy,
            "water_temperature_policy": water_temperature_policy,
            "water_temperature_overlay": (None if water_temperature_overlay is None
                else str(Path(water_temperature_overlay).resolve()))},
        "input_manifest_schema": manifest["schema"],
        "input_manifest_sha256": input_manifest_digest,
        "decoder": {
            "name": paths["bridge"].name,
            "sha256": _sha256(paths["bridge"]),
            "implementation": ("gpuwm-all-rust-netcdf-bridge"
                if paths["grib"].suffix.lower() == ".nc"
                else "gpuwm-all-rust-grib1-bridge"),
        },
        # What ran, without what was measured (A138): the proof below
        # keeps the whole receipt.
        "preprocessing": preprocess_identity(preprocess_receipt),
        **water_overlay_binding,
    }

    staging = _atomic_staging_sibling(output_root)
    if staging.exists():
        raise FileExistsError(f"refusing stale ERA5 staging path {staging}")
    staging.mkdir(parents=True)
    writer = None
    try:
        static_cache = staging / "native-static.npz"
        geometry_receipt = staging / "geometry-receipt.json"
        portable_source_manifest = staging / "source-input-manifest.json"
        prepared_cache = staging / "prepared-cache"
        wrf_output = staging / "wrf-native-input"
        shutil.copy2(Path(input_manifest), portable_source_manifest)
        _write_static_cache(static_cache, static)
        _write_geometry_receipt(geometry_receipt, grid, cfg, static_cache)
        # Refused HERE rather than left to the hierarchy call, because the
        # hierarchy call is exactly what does not happen on a single-domain
        # preparation: without this the flag would be accepted, silently
        # emit nothing, and the missing corridor would surface much later
        # as the forecast runner's relocation refusal.
        if statics_corridor is not None and len(exp.domains) < 2:
            raise ValueError(
                "--statics-corridor prepares child-resolution statics over "
                "the ground a child domain can reach, and this experiment "
                "has no child domain; "
                "remove the flag or prepare a domain tree")
        if len(exp.domains) > 1:
            selected_workers = hierarchy_workers
            if selected_workers is None:
                selected_workers = (
                    8 if preprocess_receipt["backend"] == "cpu" else 1)
            hierarchy_started = time.perf_counter()
            hierarchy = initialize_and_export_regular_source_hierarchy(
                nudging_context=((output_root, build_forcing_time)
                    if getattr(exp, 'nudging', None) is not None else None),
                exp=exp, grids=grids, snapshots=snapshots,
                forcing_hours=forcing_hours,
                wps_namelist=paths["wps_namelist"],
                geog_root=Path(geog_root), source_name="ERA5",
                artifact_output=staging / "hierarchy-artifacts",
                wrf_output=staging / "wrf-native-input",
                root_initial_result=initial_result, root_met=initial_met,
                root_soil=soil, root_static_fields=static,
                root_boundaries=boundaries,
                bridge_manifest_sha256=input_manifest_digest,
                source_manifest_sha256=input_manifest_digest,
                namelist_sha256=_sha256(paths["experiment_config"]),
                source_identity=native_source_identity,
                source_orography=hierarchy_source_orography,
                source_inventory=source_inventory,
                source_units=source_units,
                workers=selected_workers,
                preprocess_backend=preprocess_receipt["backend"],
                cpu_bridge=cpu_preprocess_bridge,
                input_provenance={
                    "input_manifest_sha256": input_manifest_digest,
                    "decoder_sha256": _sha256(paths["bridge"]),
                    "preprocessing": preprocess_receipt,
                },
                artifact_manifest_reference=(
                    "../hierarchy-artifacts/domain-artifacts.json"),
                # A child sees the catalog, not the case data.
                soil_texture_downscale=declared_soil_texture_downscale(
                    declared),
                statics_corridor=statics_corridor,
                static_highres=static_highres,
                sfcp_to_sfcp=case_policy["sfcp_to_sfcp"],
                water_temperature_policy=water_temperature_policy,
                # The GPU runner consumes hierarchy-artifacts/; the
                # unchanged-WRF file set beside it is a companion, as on
                # the GFS, mapped and HRRR tree routes.  Required, a
                # state the WRF format cannot represent (a deferred
                # [perturbation] bubble, which the prepared runner
                # applies) failed this preparation after every domain
                # was built; optional, it is recorded as a REFUSED
                # export manifest in the proof instead.
                stock_wrf_export="optional",
            )
            hierarchy_seconds = time.perf_counter() - hierarchy_started
            verify_overlay_sequence(snapshots)
            final_manifest = _verify_input_manifest(
                Path(input_manifest), input_manifest_digest, paths)
            if final_manifest != manifest:
                raise ValueError(
                    "ERA5 input manifest content changed during preparation")
            proof = {
                "schema": "gpuwm-era5-native-hierarchy-proof-v1",
                "status": "READY_NOT_YET_STOCK_WRF_GATED",
                # The mode the companion export ran in, which is what
                # lets a reader accept a REFUSED export manifest below.
                "stock_wrf_export": "optional",
                "domain_count": len(exp.domains),
                "vertical_coordinate": _vertical_coordinate_receipt(
                    exp, vertical_adaptation),
                "forcing_times": [value.isoformat() for value in times],
                "forcing_hours": forcing_hours,
                "boundary_interval_seconds": boundary_interval_seconds,
                "input_manifest_sha256": input_manifest_digest,
                "decoder_sha256": _sha256(paths["bridge"]),
                "preprocessing": preprocess_receipt,
                # Present only when the preparation opted in: the sealed
                # statics-corridor set, digest-bound here the way every
                # other sealed artifact is.  Absent, the proof is
                # byte-for-byte what it always was.
                **(
                    {"statics_corridor": dict(
                        hierarchy.statics_corridor_receipt)}
                    if hierarchy.statics_corridor_receipt is not None
                    else {}),
                **soil_floor_binding,
                # The soil-state SOURCE resolution, unconditional: a reader
                # of this forecast must be able to answer "how coarse was
                # the soil I started from" without the config, and whether
                # the sub-source-cell reconstitution ran on it.  Bound
                # outside CONDITIONAL_PREPARATION_RECEIPTS on purpose: the
                # prepared-cache validator compares that tuple EXACTLY on
                # both sides, and this receipt is proof-only.
                "soil_texture_downscale": dict(
                    getattr(soil, "soil_texture_downscale", {}) or {}),
                **water_overlay_binding,
                "root_static_provider": root_static_provider,
                "root_static_receipt": root_static_receipt,
                # WHETHER EACH DOMAIN'S INITIALIZATION MODIFIED VAPOUR ON
                # THE WAY IN, root and children alike.  Unconditional, and
                # stated even when no floor fired: an absent key would read
                # as "prepared before the receipt existed", a different
                # claim and one no reader of the bundle could check.
                **dict(hierarchy.hierarchy.moisture_floor_receipts),
                "hierarchy_workers": selected_workers,
                "static_catalog": dict(hierarchy.static_catalog_receipt),
                "source_coverage": dict(
                    hierarchy.source_coverage_receipt),
                "hierarchy_topology": dict(hierarchy.topology_receipt),
                "artifact_receipt": dict(
                    hierarchy.hierarchy.artifacts.receipt),
                "wrf_manifest": dict(hierarchy.hierarchy.wrf_manifest),
                # Present only when the experiment carries a
                # [perturbation] block, which this tree's forecast
                # applies at start; the prepared arrays stay unperturbed.
                **({"initial_perturbation": initial_perturbation}
                   if initial_perturbation is not None else {}),
                "timing_seconds": {
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
            os.replace(staging, output_root)
            return proof
        identity = prepared_cache_identity(
            bridge_manifest_sha256=input_manifest_digest,
            source_manifest_sha256=input_manifest_digest,
            static_cache_sha256=_sha256(static_cache),
            namelist_sha256=_sha256(paths["experiment_config"]),
            domain_config=exp.root,
            forcing_hours=forcing_hours,
            source_identity=native_source_identity,
        )
        writer = PreparedTreeWriter(
            staging=staging, output_root=output_root, identity=identity)
        if not start_first:
            writer.decline_chaining(SEALED_REASONS["water_overlay"])
        preprocessing_receipt_sha256 = hashlib.sha256(
            _canonical(preprocess_receipt).encode("utf-8")).hexdigest()
        # Everything the proof says that the start time already knows; the
        # seal adds the artifact records, the cache receipt, the export and
        # the wall times (SEAL_ONLY_PROOF_KEYS).
        proof_head = {
            "schema": "gpuwm-era5-direct-wrf-proof-v2",
            "status": "READY_NOT_YET_STOCK_WRF_GATED",
            "vertical_coordinate": _vertical_coordinate_receipt(
                exp, vertical_adaptation),
            "forcing_times": [value.isoformat() for value in times],
            "forcing_hours": forcing_hours,
            "boundary_interval_seconds": boundary_interval_seconds,
            "input_manifest_sha256": input_manifest_digest,
            "decoder_sha256": _sha256(paths["bridge"]),
            "preprocessing": preprocess_receipt,
            "preprocessing_receipt_sha256": preprocessing_receipt_sha256,
            **soil_floor_binding,
            # The soil-state SOURCE resolution, unconditional: a reader
            # of this forecast must be able to answer "how coarse was
            # the soil I started from" without the config, and whether
            # the sub-source-cell reconstitution ran on it.  Bound
            # outside CONDITIONAL_PREPARATION_RECEIPTS on purpose: the
            # prepared-cache validator compares that tuple EXACTLY on
            # both sides, and this receipt is proof-only.
            "soil_texture_downscale": dict(
                getattr(soil, "soil_texture_downscale", {}) or {}),
            **water_overlay_binding,
            "source_inputs": {
                "manifest_schema": manifest["schema"],
                "manifest_sha256": input_manifest_digest,
                "files": manifest["files"],
            },
            # WHETHER THIS INITIALIZATION MODIFIED VAPOUR ON THE WAY IN.
            # Unconditional, and stated even when no floor fired.
            **moisture_floor_proof_entry(
                initial_result,
                when_unrecorded=(
                    "this preparation's initialization result carries "
                    "no moisture-floor field, so it came from an ingest "
                    "predating the receipt; re-prepare the case to "
                    "record whether its vapour was floored on the way "
                    "in")),
        }
        # One machine, one card: the forecast may start beside this
        # producer only when both fit (boundary_stream.chained_admission).
        writer.admit(
            experiment=exp, backend=str(preprocess_receipt["backend"]),
            device_bytes=producer_device_bytes(str(preprocess_receipt["backend"])),
            urban_columns=prepared_head_urban_columns(exp, static))
        cache_started = time.perf_counter()
        writer.write_head(
            initial_result=initial_result, met=initial_met,
            surface=_canonical_surface(soil),
            metadata={
                "source_adapter": "era5",
                "initial_valid_time": times[0].isoformat(),
                "last_valid_time": times[-1].isoformat(),
                "forcing_hours": forcing_hours,
                "boundary_interval_seconds": boundary_interval_seconds,
                "preprocessing": preprocess_identity(preprocess_receipt),
                **soil_floor_binding,
            },
            lbc={
                "spec_bdy_width": cfg.spec_bdy_width,
                "spec_zone": cfg.spec_zone,
                "relax_zone": cfg.relax_zone,
                "schedule": [
                    [float(earlier * 3600), float(later * 3600)]
                    for earlier, later in zip(forcing_hours,
                                              forcing_hours[1:])],
                "fields": forcing.inventory,
            },
            proof_head=proof_head,
            input_manifest_sha256=input_manifest_digest,
            forcing=forcing,
        )
        cache_seconds = time.perf_counter() - cache_started
        # The start state has done its work: it is in the head.
        del initial_result, initial_met
        release_backend_memory(preprocess)
        boundaries_started = time.perf_counter()
        if boundaries is not None:
            writer.write_intervals(boundaries.intervals)
        else:
            writer.stream_forcing_times(
                count=len(snapshots), build_forcing_time=build_forcing_time,
                forcing=forcing, times=times,
                release=lambda: release_backend_memory(preprocess))
            initialize_seconds += time.perf_counter() - boundaries_started
        cache_started = time.perf_counter()
        cache_receipt = writer.seal_cache()
        cache_seconds += time.perf_counter() - cache_started
        portable_cache_receipt = dict(cache_receipt)
        portable_cache_receipt["path"] = prepared_cache.name
        # Where the tree lives now: published at the head when chained,
        # still staged otherwise.
        static_cache = writer.root / static_cache.name
        geometry_receipt = writer.root / geometry_receipt.name
        portable_source_manifest = writer.root / portable_source_manifest.name
        prepared_cache = writer.cache_path
        wrf_output = writer.root / wrf_output.name
        export_started = time.perf_counter()
        export_receipt = export_prepared_wrf(
            prepared_cache, static_cache, geometry_receipt, wrf_output,
            valid_time=times[0],
            boundary_interval_seconds=boundary_interval_seconds,
            experiment_config_suite=experiment_config_suite)
        export_seconds = time.perf_counter() - export_started
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
                for name, details in export_receipt["files"].items()
            },
        }
        verify_overlay_sequence(snapshots)
        proof = {
            **proof_head,
            "initialization_artifacts": initialization_artifacts,
            "prepared_cache": portable_cache_receipt,
            "export": export_receipt,
            "boundary_stream": writer.boundary_stream_proof(),
            "timing_seconds": {
                "decode": decode_seconds,
                "initialize_all_times": initialize_seconds,
                "write_prepared_cache": cache_seconds,
                "direct_wrf_export": export_seconds,
                "total": time.perf_counter() - total_started,
            },
        }
        writer.publish(proof)
        return proof
    except BaseException as error:
        shutil.rmtree(staging, ignore_errors=True)
        if writer is not None:
            # A head already published stays, marked failed, so a waiting
            # forecast ends with this reason and the next preparation of
            # this output root rebuilds it.
            writer.fail(error)
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grib", type=Path, required=True)
    parser.add_argument("--vtable", type=Path, required=True)
    parser.add_argument(
        "--bridge", type=Path, required=True,
        help="prebuilt executable all-Rust GRIB1 bridge bound by the manifest",
    )
    parser.add_argument("--wps-namelist", type=Path, required=True)
    parser.add_argument("--static-input", type=Path)
    parser.add_argument("--static-receipt", type=Path)
    parser.add_argument("--source-orography", type=Path)
    parser.add_argument("--source-orography-variable", default="SOILHGT")
    parser.add_argument(
        "--domain-source-orography", action="append", metavar="DNN=PATH",
        help=(
            "repeat once per ERA5 hierarchy domain; paths are target-grid "
            "orography artifacts using --source-orography-variable"),
    )
    parser.add_argument(
        "--water-temperature-overlay", type=Path, metavar="ANALYSIS",
        help=(
            "optional high-resolution water-temperature analysis (netCDF "
            "or GRIB2) replacing ERA5 SST/SKINTEMP over water source "
            "cells; off by default"),
    )
    parser.add_argument("--experiment-config", type=Path, required=True)
    parser.add_argument("--experiment-config-suite", action="store_true")
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--input-manifest-sha256", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--preprocess-backend", choices=("cuda", "cpu", "auto"),
        default="auto",
        help="where the source-grid/WRF-real setup transforms run "
             "(default auto: CUDA when the certified runtime is usable, "
             "otherwise the deterministic parallel CPU backend, "
             "announced in one line)")
    parser.add_argument("--preprocess-workers", type=int)
    parser.add_argument("--cpu-preprocess-bridge", type=Path)
    parser.add_argument("--geog-root", type=Path)
    parser.add_argument("--hierarchy-workers", type=int)
    parser.add_argument(
        "--statics-corridor", nargs="?", const="all", default=None,
        metavar="GRID_IDS",
        help="seal child-resolution statics over the ground each child "
             "can reach, for every child (bare flag) or for a "
             "comma-separated list of child grid ids.  REQUIRED when "
             "the experiment configures "
             "[relocation]: a prepared bundle runs without WPS_GEOG, so a "
             "nest that moves onto ground it has not visited has no other "
             "source of terrain, landuse and soil category.  Costs host "
             "RAM and disk only, never VRAM")
    return parser


@owns_source_coverage_refusal
def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    # `all` or an explicit child list, resolved at the door so a typo is an
    # argv complaint rather than a failure deep in the hierarchy.  Spelled
    # exactly as the GFS and mapped front doors spell it, so all three take
    # the same string.
    statics_corridor = args.statics_corridor
    if statics_corridor is not None and statics_corridor != "all":
        try:
            statics_corridor = tuple(
                int(part) for part in statics_corridor.split(",") if part)
        except ValueError:
            parser.error(
                "--statics-corridor accepts 'all' or a comma-separated "
                f"list of child grid ids, not {args.statics_corridor!r}")
        if not statics_corridor:
            parser.error(
                "--statics-corridor was given an empty grid-id list; pass "
                "the bare flag for every child, or name ids")
    try:
        hierarchy_source_orography = _domain_source_orography(
            args.domain_source_orography,
            args.source_orography_variable,
        )
    except ValueError as error:
        parser.error(str(error))
    try:
        proof = prepare_era5_wrf(
            grib=args.grib, vtable=args.vtable, bridge=args.bridge,
            wps_namelist=args.wps_namelist,
            static_input=args.static_input,
            static_receipt=args.static_receipt,
            source_orography=args.source_orography,
            source_orography_variable=args.source_orography_variable,
            experiment_config=args.experiment_config,
            experiment_config_suite=args.experiment_config_suite,
            input_manifest=args.input_manifest,
            input_manifest_sha256=args.input_manifest_sha256,
            output_root=args.output_root,
            preprocess_backend=args.preprocess_backend,
            preprocess_workers=args.preprocess_workers,
            cpu_preprocess_bridge=args.cpu_preprocess_bridge,
            geog_root=args.geog_root,
            hierarchy_workers=args.hierarchy_workers,
            statics_corridor=statics_corridor,
            hierarchy_source_orography=hierarchy_source_orography,
            water_temperature_overlay=args.water_temperature_overlay,
        )
    except InitializationMemoryRefused as error:
        # A preparation the card cannot hold under an explicit
        # --preprocess-backend cuda (PreparationDeviceRefused), or any
        # other measured initialization budget that refuses the case, is
        # a refusal with its own remedy line: one message at exit 2, as
        # `woof` itself answers it, not a traceback that buries the
        # remedy at the bottom of a stack.
        print(f"rw-wps --source era5: {error}", file=sys.stderr)
        return 2
    print(json.dumps(proof, indent=2, sort_keys=True))
    # Parity with the GFS and 20CRv3 front doors, and the last silent
    # one of the three the prepared-forecast runners actually support:
    # a complete, hash-bound run command on stderr, so the proof
    # document on stdout stays machine-readable.
    from woof.gfs_direct import prepared_forecast_next_command

    for line in prepared_forecast_next_command(
            proof, output_root=args.output_root,
            experiment_config=args.experiment_config,
            wps_namelist=args.wps_namelist, source="era5"):
        print(line, file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
