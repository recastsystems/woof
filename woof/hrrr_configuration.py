"""Bind native analyzed-input preparation to the configuration it will run."""
from __future__ import annotations

import copy
from dataclasses import asdict
from pathlib import Path
import tempfile
import tomllib


#: The route name a native HRRR root preparation gives the shared
#: [perturbation] owner, in its refusal and its deferral alike.
NATIVE_ROOT_PREPARATION_ROUTE = "native HRRR root preparation"


def root_perturbation_deferral(experiment_config, *, announce=True):
    """The [perturbation] deferral a native root preparation records.

    This route prepares d01 alone and builds the children in a later
    stage from namelists, so the configuration is read here or nowhere:
    the root seals the receipt into its source identity and the
    hierarchy stage relays it (woof.hrrr_hierarchy_direct).  The tree
    is the configuration's, not the root's, which is why the whole
    configuration is loaded rather than the d01 slice this route
    prepares.  WRF namelists spell no [perturbation], so a namelist-only
    preparation has nothing to defer.
    """
    if experiment_config is None:
        return None
    from woof.experiment import deferred_initial_perturbation, load_experiment
    return deferred_initial_perturbation(
        load_experiment(experiment_config), NATIVE_ROOT_PREPARATION_ROUTE,
        announce=announce)


def resolved_run_settings(cfg):
    """Wire representation of actual RunConfig with resolved radiation selectors."""
    from woof.config import radiation_scheme_ids
    resolved = asdict(cfg)
    resolved["ra_lw_physics"], resolved["ra_sw_physics"] = radiation_scheme_ids(cfg)
    resolved["radiation_scheme_ids"] = list(radiation_scheme_ids(cfg))
    # Wire form: tuples as JSON arrays.  The preparation report is JSON, so a
    # tuple field (the ocean profiles became tuples to keep RunConfig hashable,
    # lane/cap-fix-a 1e5d77fbd3e) came back as a list and () != [] refused every
    # HRRR preparation with "resolved physics differs from the request"
    # (V58 whole HRRR gate run on lane/cap-base, 2026-10-21).
    def wire(value):
        if isinstance(value, (tuple, list)):
            return [wire(item) for item in value]
        if isinstance(value, dict):
            return {key: wire(item) for key, item in value.items()}
        return value
    return {key: wire(value) for key, value in resolved.items()}


def resolve_root_experiment(*, target, vertical, namelist_input, start_time,
                            run_seconds, experiment_config=None, wps_namelist=None,
                            physics_profile=None, acknowledgements=(),
                            history_interval_seconds=None):
    """Use an explicit experiment or the common WRF importer; retain d01 settings.

    The CLI's forcing window may be a subwindow of the configured hierarchy
    (sealed extensions use this). Geometry and vertical values remain asserted
    against the same target that the decoder and static preparation consume.
    """
    from woof.experiment import build_experiment_from_config_tables
    from woof.config_authority import read_config_authority
    from woof.namelist_import import import_namelists, parse_namelist
    from woof.physics_compat import single_domain_runtime_switches

    if experiment_config is not None:
        # Validates the companion tables through their owners, and refuses
        # a single-domain [perturbation] block here, before any source is
        # decoded: the prepared single-domain runner applies no bubble.
        root_perturbation_deferral(experiment_config, announce=False)
        raw = tomllib.loads(read_config_authority(experiment_config).payload.decode("utf-8"))
        authority = str(read_config_authority(experiment_config).source)
        # The scheme generations the declared [fetch] source carries are
        # filled BEFORE the table is split off, exactly as the config-table
        # loader behind `woof sim` fills them
        # (woof.physics_source_defaults.fill_omitted_generation_selectors),
        # so the physics this preparation resolves, records in its receipt
        # and holds its benchmark to is the physics the forecast integrates
        # from the same file.  Breakage this prevents: a bare
        # operational-fork-source configuration (no thompson_version,
        # thompson_fork_snow_fall or mynn_sfclay_variant written) prepared
        # through this route resolved and recorded WRF v4.6.1's generations
        # while `woof sim` ran the fork pair (lane review, 2026-10-06).
        from woof.physics_source_defaults import fill_omitted_generation_selectors
        fill_omitted_generation_selectors(raw, raw.get("fetch"))
        # Prepared meteorological identity is independent of execution capacity
        # and fetch hints. The caller keeps these in its unchanged launch config.
        raw.pop("fetch", None)
        raw.pop("tiles", None)
        # [radar_heating] is an execution control too: one preparation
        # serves the unheated control and every heated arm, and the launch
        # config hands the table to the door (--radar-heating-table).
        raw.pop("radar_heating", None)
        for domain in raw.get("domain", ()):
            domain.pop("tiles", None)
    else:
        # WHAT THE NAMED SUITE CARRIES THAT A WRF NAMELIST CANNOT SPELL.
        # Two facts, both woof's own, both read off the profile the
        # caller named rather than guessed from the selectors:
        #
        #   ra_rrtmg_variant -- which 4/4 implementation serves the pair;
        #   wrf_rrtmg_compatibility -- which WRF RRTMG lineage the run
        #   reproduces, which the RTE+RRTMGP arm READS to choose its snow
        #   treatment and stamps into its restart identity.
        #
        # And the governance declaration the suite makes about itself: a
        # shortwave-only suite integrates the land surface against a
        # declared constant downward longwave, which the configuration
        # route states in [experiment].acknowledgements and a WRF namelist
        # has no field for.  Naming the profile IS that declaration, so it
        # travels with the profile here and the same load guard reads it
        # from the same array on both routes.
        profile_switches = (single_domain_runtime_switches(physics_profile)
                            if physics_profile is not None else {})
        variant = profile_switches.get("ra_rrtmg_variant")
        compatibility = profile_switches.get("wrf_rrtmg_compatibility")
        from woof.physics_compat import profile_declared_acknowledgements
        declared = profile_declared_acknowledgements(physics_profile)
        # A SEPARATE NAME from the caller's own list: `acknowledgements`
        # stays what the operator stated, which is what the expert-tuple
        # check below is asking about.
        imported_acknowledgements = tuple(acknowledgements) + tuple(
            token for token in declared if token not in acknowledgements)
        with tempfile.TemporaryDirectory(prefix="gpuwm-namelist-root-") as scratch:
            if wps_namelist is None:
                # The native source door already owns the projection/target.
                # WRF input owns its domain columns; keep every column so the
                # common importer can validate their hierarchy before extracting d01.
                sections = parse_namelist(namelist_input)
                dm = sections.get("domains", {})
                count = int(dm.get("max_dom", [1])[0])
                def row(key, default):
                    values = list(dm.get(key, [default] * count))
                    if key == "parent_id":
                        values[0] = 1
                    return ", ".join(repr(v) for v in values)
                wps = ("&share\n wrf_core='ARW',\n max_dom=" + str(count) +
                       ",\n interval_seconds=3600,\n io_form_geogrid=2,\n/\n&geogrid\n" +
                       "\n".join(f" {key}={row(key, value)}," for key, value in (
                           ("parent_id", 1), ("parent_grid_ratio", 1),
                           ("i_parent_start", 1), ("j_parent_start", 1),
                           ("e_we", target.nx + 1), ("e_sn", target.ny + 1))) +
                       f"\n dx={target.dx_m!r},\n dy={target.dy_m!r},\n map_proj='lambert',\n" +
                       f" ref_lat={target.ref_lat!r},\n ref_lon={target.ref_lon!r},\n" +
                       f" truelat1={target.truelat1!r},\n truelat2={target.truelat2!r},\n" +
                       f" stand_lon={target.stand_lon!r},\n geog_data_res='default',\n/\n")
                wps_namelist = Path(scratch) / "namelist.wps"
                wps_namelist.write_text(wps, encoding="utf-8")
            text, _ = import_namelists(wps_namelist, namelist_input,
                name=target.name, rrtmg_variant=variant,
                rrtmg_compatibility=compatibility,
                acknowledgements=imported_acknowledgements)
        raw = tomllib.loads(text)
        authority = str(namelist_input)
    from woof.static.source_defaults import with_source_static_defaults
    raw = with_source_static_defaults(raw, "hrrr")
    full = build_experiment_from_config_tables(
        raw, source=authority, base_dir=Path(authority).parent)
    from woof.hrrr_route_inputs import target_domain
    observed = asdict(target_domain(full))
    expected = asdict(target)
    # A descriptive experiment name is not geometry.
    observed.pop("name", None)
    expected.pop("name", None)
    # Nor is the soil donor radius: the configuration writes its own, and
    # the native target document is the one the preparation reads it from.
    observed.pop("surface_fallback_radius_cells", None)
    expected.pop("surface_fallback_radius_cells", None)
    if observed != expected:
        drift = {key: (observed.get(key), value) for key, value in expected.items()
                 if observed.get(key) != value}
        raise ValueError(f"configured d01 differs from the native target: {drift}")
    for name in ("p_top", "hybrid_opt", "etac", "eta_levels"):
        if getattr(full.vertical, name) != getattr(vertical, name):
            raise ValueError(f"configured d01 vertical {name} differs from namelist.input")
    raw = copy.deepcopy(raw)
    if raw.get("static") is None or "highres" not in raw["static"]:
        # (A [static] table naming only a static source keeps its source
        # and takes this default beside it.)
        # The root's own configuration keeps only d01, so the grid-spacing
        # default is settled on the whole tree here and written into the
        # table the preparation binds: a 2 km root over a 667 m child
        # carries the block its child needs, and applies it to itself only
        # if its own spacing reaches the row.
        from woof.static.highres_production import (
            default_static_highres, static_highres_identity)
        default = default_static_highres(
            [float(dc.run.dx) for dc in full.domains])
        if default is not None:
            raw["static"] = {**(raw.get("static") or {}),
                             "highres": static_highres_identity(default)}
    raw["domain"] = raw["domain"][:1]
    for key in ('nudging','spectral_nudging'):
        if isinstance(raw.get(key),dict) and 'domain' in raw[key]:
            raw[key]['domain'] = {gid:control for gid,control in raw[key]['domain'].items()
                                  if int(gid) == full.root.grid_id}
    from woof.experiment import (drop_unreached_grell_selectors,
                                  drop_unreached_relocation,
                                  drop_unreached_sase_selectors)
    drop_unreached_sase_selectors(raw, [full.root.run])
    drop_unreached_grell_selectors(raw, [full.root.run])
    drop_unreached_relocation(raw, [full.root.grid_id])
    raw["experiment"].update(name=target.name, feedback=0, smooth_option=0)
    if start_time is not None:
        raw["experiment"]["start_time"] = start_time
    if run_seconds is not None:
        raw["experiment"]["run_seconds"] = float(run_seconds)
    if history_interval_seconds is not None:
        raw["domain"][0]["history_interval_s"] = float(history_interval_seconds)
    exp = build_experiment_from_config_tables(
        raw, source=authority, base_dir=Path(authority).parent)
    if physics_profile is not None:
        from woof.physics_compat import validate_single_domain_physics_profile
        validate_single_domain_physics_profile(physics_profile, config=exp.root.run,
            expert_acknowledgements=tuple(acknowledgements))
    # Published authorities live beside the prepared payload. Resolve only
    # companion-owned path keys, including not-yet-created cache directories.
    from woof.case_data import resolved_case_data_paths
    from woof.static.highres_production import parse_static_table
    base_dir = Path(authority).resolve().parent
    from woof.config import _anchor_config_file_paths
    for table in [raw.get("shared", {}), *raw.get("domain", [])]:
        _anchor_config_file_paths(table, Path(authority), keys=("wind_farm_table", "wind_farms"))
    if "case_data" in raw:
        raw["case_data"] = resolved_case_data_paths(
            raw["case_data"], base_dir=base_dir, source=authority)
    highres = parse_static_table(raw.get("static"), source=authority, base_dir=base_dir)
    if highres is not None and "highres" in raw["static"]:
        raw["static"]["highres"]["cache_root"] = str(highres.cache_root.resolve())
    return exp, raw


def native_configuration_defaults(*, experiment_config=None, namelist_input,
                                  domain_spec=None, wps_namelist=None,
                                  physics_profile=None, acknowledgements=()):
    """Load a producing configuration's clock before applying CLI overrides."""
    if experiment_config is not None:
        from woof.experiment import load_experiment
        return load_experiment(experiment_config)
    from woof.ingest.hrrr_target import load_hrrr_target_domain
    from woof.vertical_contract import explicit_vertical_from_wrf_namelist
    target = load_hrrr_target_domain(domain_spec)
    vertical = explicit_vertical_from_wrf_namelist(namelist_input,
        expected_nz=target.nz, context="native preparation configuration")
    exp, _ = resolve_root_experiment(target=target, vertical=vertical,
        namelist_input=namelist_input, start_time=None, run_seconds=None,
        wps_namelist=wps_namelist, physics_profile=physics_profile,
        acknowledgements=acknowledgements)
    return exp
