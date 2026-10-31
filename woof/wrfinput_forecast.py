"""Forecast WRF real.exe inputs through WOOF's shared domain-tree runtime."""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
import hashlib
import json
from pathlib import Path
import re
from types import MappingProxyType
from typing import Mapping

import numpy as np

from woof.forecast_initialization import DomainInitialization
from woof.progress_log import ProgressOptions, add_progress_arguments


#: WRF ``&time_control`` entries a ``--restart`` may change in namelist.input:
#: the history and restart cadence and each domain's history window, in
#: every unit form WRF reads.  With the output-only diagnostic switches
#: (``woof.checkpoint_identity.CONFIG_DIAGNOSTIC_FIELDS``, which a namelist
#: names the same way) they are what
#: :meth:`WrfInitialization.preparation_receipt_sha256` leaves out of a
#: checkpoint's identity, as ``woof.core.model.restart_identity_payload``
#: leaves them out of the experiment's.  The forecast length is not here:
#: the import receipt also derives other entries from the namelist window
#: (the eclipse check names it), and this door changes the length of a
#: resumed run with ``--run-seconds``, which leaves the namelist alone.
_RESTART_ADJUSTABLE_NAMELIST_KEY = re.compile(
    r"(history_interval|history_interval2|history_interval_change|history_begin"
    r"|history_end|restart_interval)(_[ydhms])?")


def restart_adjustable_namelist_key(key: str) -> bool:
    """Whether a namelist entry is one a ``--restart`` may change."""
    from woof.checkpoint_identity import CONFIG_DIAGNOSTIC_FIELDS
    key = str(key).strip().lower()
    return (key in CONFIG_DIAGNOSTIC_FIELDS
            or _RESTART_ADJUSTABLE_NAMELIST_KEY.fullmatch(key) is not None)


def _sha(path: Path) -> str:
    # Once per process for an unchanged file (woof.file_digest): the leg
    # hashes each input before and after its read, for the receipt and at
    # the end of the run, and a changed file is still hashed again.
    from woof.file_digest import sha256_file
    from woof.filesystem_paths import io_path
    return sha256_file(io_path(path))


@dataclass(frozen=True)
class WrfLanduseIdentity:
    attributes: Mapping[str, object]

    def landuse_global_attrs(self):
        return dict(self.attributes)


@dataclass(frozen=True)
class WrfDomainBundle:
    grid_id: int
    restored: object
    static_fields: Mapping[str, np.ndarray]
    authority_sha256: Mapping[str, str]
    landuse: object
    geog_selection: WrfLanduseIdentity
    #: &physics fractional_seaice the land use above was initialised with;
    #: the Noah mosaic tile door must split sea ice the same way.
    fractional_seaice: bool = False
    #: A nest's WRF start (:class:`woof.ingest.wrfinput_nest_initial
    #: .NestInputStart`): ``restored`` above already carries its blended
    #: HGT and adjusted T/QVAPOR; :meth:`WrfInitialization.finish_nest_start`
    #: applies its press_adj after the initial feedback.  ``None`` for the
    #: root and for a warm start.
    nest_start: object | None = None


@dataclass(frozen=True)
class WrfTreeInputs:
    prepared_root: Path
    experiment_config: Path
    experiment: object
    grids: tuple[object, ...]
    domains: tuple[WrfDomainBundle, ...]
    forcing_hours: tuple[float, ...]
    boundary_interval_seconds: int
    source_identity: Mapping[str, object]
    execution_plan: Mapping[str, object]
    authority_sha256: Mapping[str, str]
    artifact_paths: Mapping[str, Path]
    boundaries: object
    source: str = 'wrfinput'
    statics_corridor: object | None = None
    #: The prepared door's physics-profile assertion
    #: (:class:`woof.prepared_domain_tree_forecast.PreparedTreeInputs`).
    #: The tree runner reads it on every source when it writes the run
    #: report, so the WRF-input doors carry it with the same default: a
    #: stock real.exe pair completed its simulation and then failed with
    #: AttributeError before the report was written.
    physics_profile_assertion: Mapping[str, object] | None = None
    #: The tree runner's acoustic substep derivation, filled in by the
    #: runner itself (``prepared_domain_tree_forecast._with_terrain_acoustics``)
    #: from these domains' static terrain, as on the prepared door.
    acoustic_substeps: Mapping[str, object] | None = None
    #: The tree runner's long-step derivation, filled in beside the
    #: substep one (``woof.terrain_clock.clock_receipt``).
    terrain_clock: Mapping[str, object] | None = None
    #: ``mp_tend_radar = 1``: each forced domain's radar tendency, read
    #: from its state file (:mod:`woof.ingest.wrfinput_radar`).  Empty
    #: when no domain is forced.
    # A factory, not a shared MappingProxyType default: Python 3.11 and 3.12
    # refuse an unhashable dataclass default, so the module did not import
    # there (found by the 2.9.1 CI rehearsal; 3.13 accepts it).
    radar_tendencies: Mapping[int, object] = field(
        default_factory=lambda: MappingProxyType({}))


@dataclass(frozen=True)
class WrfInitialState:
    state: object
    coord: object
    base: object


def wrf_initial_result(restored, state):
    """Keep the actual WRF coordinate and base state for shared runtime output."""
    from woof.ingest.wrfinput import wrf_coordinate_and_base
    coord, base = wrf_coordinate_and_base(restored)
    return WrfInitialState(state, coord, base)


class WrfInitialization:
    def __init__(self, inputs: WrfTreeInputs):
        self.inputs = inputs
        self.lateral_boundaries = inputs.boundaries
        from woof.case_data import trace_gas_overrides_from_config
        self.trace_gas_overrides = trace_gas_overrides_from_config(
            inputs.experiment_config,
            expected_sha256=inputs.authority_sha256["experiment_config"])

    def restore_domain(self, domain, grid, bundle, *, start_time,
                       scratch_arena, dycore_state_workspace):
        from woof.ingest.wrfinput import restore_domain_state, initialize_wrfinput_physics
        from woof.ingest.lateral_bc import attach_lateral_boundaries

        state = restore_domain_state(bundle.restored, domain.run,
                                     scratch_arena=scratch_arena,
                                     dycore_state_workspace=dycore_state_workspace)
        if domain.parent_id == 0:
            attach_lateral_boundaries(state, self.inputs.boundaries)
        from woof.runtime import declared_constant_glw
        def initialize():
            from woof.core.cam_ozone import cam_ozone_setup, ozone_parent_for
            cam = cam_ozone_setup(exp=self.inputs.experiment, dc=domain, grid=grid)
            from woof.core.radiation_composition import make_radiation
            radiation = make_radiation(
                domain.run, start_time, bundle.restored.raw["XLAT"],
                bundle.restored.raw["XLONG"], p_top=self.inputs.experiment.vertical.p_top,
                column_chunk=self.inputs.experiment.column_chunk,
                trace_gas_overrides=self.trace_gas_overrides,
                ozone_parent=ozone_parent_for(cam))
            return initialize_wrfinput_physics(
                state, bundle.restored, domain.run, radiation=radiation,
                radiation_start_time=start_time,
                radiation_latitude=bundle.restored.raw['XLAT'],
                radiation_longitude=bundle.restored.raw['XLONG'],
                landuse=bundle.landuse,
                constant_glw_wm2=declared_constant_glw(self.inputs.experiment),
                fractional_seaice=bundle.fractional_seaice,
                hydro_parent_geometry=grid,
                **({"cam_ozone": cam} if cam is not None else {}))
        return DomainInitialization(wrf_initial_result(bundle.restored, state), initialize)

    def finish_nest_start(self, node, bundle):
        """The rest of med_nest_initial once the tree holds this nest.

        WRF feeds the nest back into its parent BEFORE the nest's own
        start_domain (share/mediation_integrate.F:774-794), so the
        restriction reads the file's MU and press_adj moves the nest's MU
        only after it.  The tree runner calls this after the nest's
        initial feedback (none at feedback = 0).
        """
        start = getattr(bundle, 'nest_start', None)
        if start is None:
            return
        from woof.ingest.wrfinput_nest_initial import (
            apply_press_adj, restart_parent_domain)
        apply_press_adj(node.state, start)
        parent_bundle = next(b for b in self.inputs.domains
                             if b.grid_id == node.cfg.parent_id)
        from woof.wrf_exact import ENABLED as wrf_exact
        if wrf_exact:
            from woof.ingest.wrfinput_nest_initial import finish_strict_nest_start
            finish_strict_nest_start(node, start, parent_bundle.restored)
        if int(self.inputs.experiment.feedback) == 1:
            restart_parent_domain(node.parent, parent_bundle.restored)

    def attach_step_forcings(self, nodes, steppers):
        """Attach ``mp_tend_radar = 1``'s radar tendency once the run's
        steppers are known; ``None`` when no domain is forced.  The caller
        detaches the returned handle when the integration ends."""
        if not self.inputs.radar_tendencies:
            return None
        from woof.wrfinput_radar_forcing import WrfinputRadarTendency
        handle = WrfinputRadarTendency(self.inputs.radar_tendencies)
        try:
            handle.attach(nodes, steppers)
        except BaseException:
            handle.detach()
            raise
        return handle

    def verify_inputs(self, inputs):
        current = {name: _sha(path) for name, path in inputs.artifact_paths.items()}
        if current != dict(inputs.authority_sha256):
            changed = sorted(name for name in current if current[name] != inputs.authority_sha256.get(name))
            raise RuntimeError(f'WRF input artifacts changed during execution: {changed}')

    def preparation_receipt_sha256(self):
        """The import receipt as a checkpoint binds it: every entry, except
        the namelist entries a ``--restart`` may change.

        The receipt binds the namelist.input bytes (``source_files``) and
        lists every translated key, so writing a restart-adjustable switch
        into the namelist moved it.  Named breakage (V2W WoFS, 2026-10-29):
        resuming an nwp_diagnostics = 0 checkpoint with nwp_diagnostics = 1
        in namelist.input refused "tree restart d01 was written for a
        different run: experiment_identity, preparation_receipt_sha256
        differ(s)" while the same message lists nwp_diagnostics as a change a
        restart may make.  The experiment side already leaves these keys out
        (``restart_identity_payload`` and the diagnostic-switch normalisation
        in ``woof.io.restart``); this is the same exemption for the receipt.
        The namelist binds by its parsed values without those entries; every
        other input file and receipt entry binds exactly as before.  The run
        length of a resumed run is ``--run-seconds``, not a namelist edit.
        """
        from woof.namelist_import import parse_namelist

        content = self.inputs.artifact_paths['preparation_receipt'].read_bytes()
        if hashlib.sha256(content).hexdigest() != self.inputs.authority_sha256['preparation_receipt']:
            raise RuntimeError('WRF import receipt changed before checkpoint identity was bound')
        payload = json.loads(content)
        if not isinstance(payload, dict) or payload.get('schema') != 'gpuwm-wrf-input-import-v1':
            raise ValueError('WRF-input checkpoint identity requires the verified WRF import receipt')
        namelist = Path(self.inputs.artifact_paths['namelist_input'])
        files = payload.get('source_files') or {}
        if _sha(namelist) != files.get('namelist_input'):
            raise RuntimeError(f'{namelist}: changed after its import receipt was written')
        kept = {group.lower(): {key.lower(): values for key, values in entries.items()
                                if not restart_adjustable_namelist_key(key)}
                for group, entries in parse_namelist(namelist).items()}
        files['namelist_input'] = {'restart_identity': hashlib.sha256(json.dumps(
            kept, sort_keys=True, separators=(',', ':'), default=str).encode('utf-8')).hexdigest()}
        translation = payload.get('namelist_translation')
        if isinstance(translation, dict):
            for name, entries in list(translation.items()):
                if isinstance(entries, list):
                    translation[name] = [
                        entry for entry in entries
                        if not (isinstance(entry, dict)
                                and restart_adjustable_namelist_key(entry.get('key', '')))]
        # The formatted rendering of the translation above, counts included.
        payload.pop('namelist_translation_text', None)
        authority = {'schema': 'gpuwm-wrf-input-restart-authority-v1', 'preparation': payload}
        return hashlib.sha256(json.dumps(authority, sort_keys=True, separators=(',', ':'),
                                         allow_nan=False).encode('utf-8')).hexdigest()

    def domain_content_sha256(self, bundle):
        recovery = getattr(bundle.restored, 'soil_recovery', {})
        if recovery:
            from woof.prepared_documents import json_bytes
            return hashlib.sha256(json_bytes({
                'wrfinput': bundle.authority_sha256['wrfinput'],
                'recovered_soil_fields': recovery['recovered_fields'],
            })).hexdigest()
        return bundle.authority_sha256['wrfinput']

    def domain_metadata(self, bundle):
        return {}


def wrfinput_window_seconds(run, run_seconds):
    """Bind a requested wrfinput run window to the shared forcing check.

    On a warm start ``run_seconds`` is the window AFTER the restart
    instant (what a reader asking for "one hour from the restart" means);
    the experiment's length is the simulation's, so the restart's elapsed
    seconds are added before the forcing check.
    """
    from woof.ingest.preflight import check_forcing_window
    clock = getattr(run, 'restart_clock', None)
    total = float(run_seconds) + (0.0 if clock is None else clock.elapsed_seconds)
    return check_forcing_window(total,
        coverage_seconds=run.coverage.coverage_seconds,
        source='woof run --wrfinput', last_valid_time=run.coverage.end)


class WrfRestartInitialization(WrfInitialization):
    """The WRF-input door's warm start: state, physics and clock from wrfrst.

    The shared restore takes WRF's restart branch for a domain read from a
    restart file (the file's base state and W, no cold-start rebuild); the
    physics overlay then fills every cross-step carrier from the file
    (:func:`woof.ingest.wrfrst.apply_warm_start_physics`), and the tree
    runner opens the clocks at the restart instant from
    :attr:`warm_start_clocks`.
    """

    def __init__(self, inputs: WrfTreeInputs):
        super().__init__(inputs)
        self.warm_start_clocks = MappingProxyType({
            int(bundle.grid_id): bundle.restored.warm_start.clock
            for bundle in inputs.domains})
        self.overlay_reports: dict[int, object] = {}

    def restore_domain(self, domain, grid, bundle, *, start_time,
                       scratch_arena, dycore_state_workspace):
        from woof.ingest.wrfrst import apply_warm_start_physics

        initialized = super().restore_domain(
            domain, grid, bundle, start_time=start_time,
            scratch_arena=scratch_arena,
            dycore_state_workspace=dycore_state_workspace)
        cold = initialized.initialize_physics
        state = initialized.state

        def initialize():
            driver = cold()
            report = apply_warm_start_physics(state, driver, bundle.restored,
                                              domain.run)
            self.overlay_reports[int(domain.grid_id)] = report
            path = self.inputs.prepared_root / f'warm-start-d{int(domain.grid_id):02d}.json'
            from woof.prepared_documents import json_bytes
            path.write_bytes(json_bytes({
                'schema': 'gpuwm-wrf-warm-start-overlay-v1',
                'restart_file': str(bundle.restored.path),
                'overlay': report.receipt(),
                'read_checks': dict(bundle.restored.warm_start.checks)}))
            for name, value in sorted(report.checks.items()):
                if name.startswith('eos_') or name.startswith('step'):
                    print(f'WRF warm start d{int(domain.grid_id):02d}: '
                          f'{name} = {value}', flush=True)
            return driver
        return DomainInitialization(initialized.initial_result, initialize)

    def domain_content_sha256(self, bundle):
        return bundle.authority_sha256['wrfrst']


def wrfinput_urban_columns(run) -> dict[int, int] | None:
    """Read only the urban land-cover planes before reserving a shared card.

    The complete WRF restore remains in the worker.  Unknown land cover
    keeps the configuration bound; malformed supplied planes are refused
    by the same numeric and geometry checks as that restore.
    """
    from woof import netcdf_bridge
    from woof.core.urban_state import (bem_workspace_counted,
                                        prepared_urban_columns)
    from woof.ingest.wrfinput import (_read_numeric,
                                       _validate_wrfinput_geometry)

    counts = {}
    metadata = getattr(run, 'metadata', {}) or {}
    for domain in getattr(run.experiment, 'domains', ()):
        cfg = domain.run
        if not bem_workspace_counted(cfg):
            continue
        item = metadata.get(int(domain.grid_id))
        attrs = getattr(item, 'global_attributes', {})
        if attrs.get('MMINLU') is None:
            continue
        expected = {'west_east': int(cfg.nx), 'south_north': int(cfg.ny)}
        with netcdf_bridge.open_dataset(item.path) as dataset:
            if 'LU_INDEX' not in dataset.variables:
                continue
            planes = {}
            for name in ('LU_INDEX', 'FRC_URB2D'):
                if name not in dataset.variables:
                    continue
                variable = dataset.variables[name]
                value = _read_numeric(variable)
                _validate_wrfinput_geometry(name, variable, expected, value)
                planes[name] = value
        count = prepared_urban_columns(
            cfg, planes['LU_INDEX'], landuse_dataset=str(attrs['MMINLU']),
            frc_urb2d=planes.get('FRC_URB2D'))
        if count is not None:
            counts[int(domain.grid_id)] = count
    return counts or None


def prepare_wrf_run(run, directory: Path, *, run_seconds: float | None = None,
                    soil_source=None, boundary_pair_file=None) -> WrfTreeInputs:
    """Validate the complete CPU handoff and bind the exact input bytes.

    ``boundary_pair_file`` (the analysis door's IAU start) is the WRF-form state the boundary's
    first record is the coupling of, when it is not the run's wrfinput: the analysis the IAU
    window forces the background toward.  It is read, held to the first record by the same
    pair check, and bound into the input hashes like every other input file."""
    import re
    import tomllib
    from woof.config import ConfigurationRefused, soil_layer_count
    from woof.core.landuse import initialize_landuse
    from woof.namelist_import import parse_namelist
    from woof.experiment import build_experiment
    from woof.ingest.wrfinput import read_wrfinput, read_wrfbdy
    from woof.prepared_domain_tree_forecast import resolve_execution_plan
    from woof.static.corridor import config_declares_follow_source
    from woof.static.projection import grids_from_projection_config

    config = run.toml_text
    if run_seconds is not None:
        # Not an inline copy of "the window is capped by the forcing" any
        # more: woof/metem_forecast.py reaches the same function, so the
        # two doors say one sentence about one configuration.
        run_seconds = wrfinput_window_seconds(run, run_seconds)
        config, count = re.subn(r'(?m)^run_seconds\s*=.*$', f'run_seconds = {float(run_seconds)}', config)
        if count != 1:
            raise ValueError('resolved experiment does not carry exactly one run_seconds')
    exp = build_experiment(tomllib.loads(config), source=str(run.namelist_input))
    if config_declares_follow_source(exp):
        raise ConfigurationRefused('moving nests require terrain and land-surface coverage for their future positions; this WRF directory contains only the initial footprints. Supply a prepared statics corridor through native preparation')
    grids = tuple(grids_from_projection_config(exp))
    warm = getattr(run, 'restart_paths', None) is not None
    paths = {'adapter_code': Path(__file__).resolve(),
             'state_adapter_code': Path(__file__).with_name('ingest') / 'wrfinput.py',
             'surface_adapter_code': Path(__file__).with_name('ingest') / 'wrfinput_noahmp.py',
             'soil_recovery_code': Path(__file__).with_name('ingest') / 'wrf_soil_recovery.py',
             'nest_initial_code': Path(__file__).with_name('ingest') / 'wrfinput_nest_initial.py',
             'namelist_input': run.namelist_input, 'wrfbdy': run.wrfbdy_path}
    if warm:
        paths['warm_start_code'] = Path(__file__).with_name('ingest') / 'wrfrst.py'
        paths.update({f'wrfrst_d{gid:02d}': path
                      for gid, path in run.restart_paths.items()})
    else:
        paths.update({f'wrfinput_d{gid:02d}': path
                      for gid, path in run.wrfinput_paths.items()})
    original_hashes = {name: _sha(path) for name, path in paths.items()}
    from woof.prepared_documents import preparation_directory, json_bytes, write_document

    fractional_values = parse_namelist(run.namelist_input).get('physics', {}).get('fractional_seaice', [0])
    fractional_seaice = bool(fractional_values[0])
    bundles = []
    restored_by_id, nest_starts = {}, {}
    for domain in exp.domains:
        cfg = domain.run
        dimensions = {'west_east':cfg.nx,'west_east_stag':cfg.nx+1,
                      'south_north':cfg.ny,'south_north_stag':cfg.ny+1,
                      'bottom_top':cfg.nz,'bottom_top_stag':cfg.nz+1,
                      'soil_layers_stag':soil_layer_count(cfg)}
        if warm:
            from woof.ingest.wrfrst import read_wrfrst
            restored = read_wrfrst(run.restart_paths[domain.grid_id],
                                   expected_dimensions=dimensions, cfg=cfg)
        else:
            restored = read_wrfinput(run.wrfinput_paths[domain.grid_id], expected_dimensions=dimensions, cfg=cfg,
                                    **({} if soil_source is None else {'soil_source': soil_source}))
        if domain.parent_id != 0 and not warm:
            # wrf.exe starts an input-file nest by med_nest_initial, not
            # from the file's words: blended terrain, T/QVAPOR at the
            # blended mass, press_adj (woof/ingest/wrfinput_nest_initial.py).
            # The parent is read as WRF holds it when the nest opens, so a
            # nest of a nest blends against its parent's blended HGT; with
            # feedback the parent's own file words then take the nest's
            # (terrain, land and surface fields), as med_nest_feedback
            # leaves them before the parent's second start_domain.
            from woof.ingest.wrfinput_nest_initial import (
                feed_back_file_fields, nest_input_start)
            if domain.parent_id not in restored_by_id:
                raise ConfigurationRefused(
                    f'd{domain.grid_id:02d}: its parent d{domain.parent_id:02d} is not '
                    'read before it; WRF opens a nest from its live parent')
            parent_cfg = next(d for d in exp.domains if d.grid_id == domain.parent_id).run
            nest_start = nest_input_start(
                restored, restored_by_id[domain.parent_id],
                i_parent_start=domain.i_parent_start, j_parent_start=domain.j_parent_start,
                ratio=domain.parent_grid_ratio, spec_bdy_width=exp.spec_bdy_width,
                blend_width=exp.blend_width, hypsometric_opt=int(cfg.hypsometric_opt))
            nest_starts[domain.grid_id] = nest_start
            restored = nest_start.restored
            if int(exp.feedback) == 1:
                restored_by_id[domain.parent_id] = feed_back_file_fields(
                    restored_by_id[domain.parent_id], restored,
                    i_parent_start=domain.i_parent_start,
                    j_parent_start=domain.j_parent_start,
                    ratio=domain.parent_grid_ratio,
                    spec_zone=int(parent_cfg.spec_zone),
                    smooth_option=int(exp.smooth_option))
        for role, source in getattr(restored, 'soil_recovery', {}).get('input_files', {}).items():
            key = f'soil_recovery_d{domain.grid_id:02d}_{role}'
            paths[key] = Path(source['path'])
            original_hashes[key] = source['sha256']
        restored_by_id[domain.grid_id] = restored
    for domain in exp.domains:
        cfg = domain.run
        restored = restored_by_id[domain.grid_id]
        nest_start = nest_starts.get(domain.grid_id)
        attrs = restored.global_attributes
        names = ('MMINLU', 'NUM_LAND_CAT', 'ISWATER', 'ISLAKE', 'ISICE', 'ISURBAN', 'ISOILWATER')
        missing = sorted(set(names) - set(attrs))
        if missing:
            raise ConfigurationRefused(f'{restored.path}: missing land-use identity {missing}')
        land_attrs = {name: attrs[name] for name in names}
        landuse = initialize_landuse(
            restored.raw['LU_INDEX'], soil_type=restored.raw['ISLTYP'],
            urban_legend=int(getattr(cfg, 'sf_urban_physics', 0)) > 0,
            landmask=restored.raw['LANDMASK'], snow=restored.raw['SNOW'],
            xice=restored.raw.get('XICE', restored.raw.get('SEAICE')),
            valid_time=(run.restart_clock.valid_time if warm
                        else domain.start_time or exp.start_time),
            cen_lat=float(attrs['CEN_LAT']),
            mminlu=str(attrs['MMINLU']), iswater=int(attrs['ISWATER']),
            islake=int(attrs['ISLAKE']), isice=int(attrs['ISICE']),
            isoilwater=int(attrs['ISOILWATER']), fractional_seaice=fractional_seaice,
            soil_temperature=restored.raw['TSLB'], sst=restored.raw.get('SST'),
            # real.exe already wrote the monthly ALBBCK (fraction, water
            # 0.08) and SNOALB into wrfinput; under usemonalb landuse_init
            # leaves them in place, so they are handed over as they are.
            usemonalb=bool(getattr(cfg, 'usemonalb', False)),
            albbck_monthly=restored.raw.get('ALBBCK'),
            snoalb=restored.raw.get('SNOALB'))
        static = {name:restored.raw[name] for name in ('LANDMASK','LU_INDEX','ISLTYP','MAPFAC_M','MAPFAC_U','MAPFAC_V','F','E')}
        static['HGT_M'] = restored.raw['HGT']
        state_key = ('wrfrst' if warm else 'wrfinput')
        bundles.append(WrfDomainBundle(domain.grid_id, restored, MappingProxyType(static),
                       MappingProxyType({state_key: original_hashes[f'{state_key}_d{domain.grid_id:02d}']}),
                       landuse, WrfLanduseIdentity(MappingProxyType(land_attrs)),
                       fractional_seaice, nest_start))
    pair_restored = None
    if boundary_pair_file is not None:
        root_cfg = exp.root.run
        paths['boundary_pair'] = Path(boundary_pair_file)
        original_hashes['boundary_pair'] = _sha(paths['boundary_pair'])
        pair_restored = read_wrfinput(
            paths['boundary_pair'], cfg=root_cfg,
            expected_dimensions={'west_east': root_cfg.nx, 'west_east_stag': root_cfg.nx + 1,
                                 'south_north': root_cfg.ny, 'south_north_stag': root_cfg.ny + 1,
                                 'bottom_top': root_cfg.nz, 'bottom_top_stag': root_cfg.nz + 1,
                                 'soil_layers_stag': soil_layer_count(root_cfg)},
            **({} if soil_source is None else {'soil_source': soil_source}))
    boundaries = read_wrfbdy(run.wrfbdy_path, run_seconds=exp.run_seconds,
                            restored=bundles[0].restored,
                            forcing_interval_seconds=run.coverage.forcing_interval_seconds,
                            spec_bdy_width=exp.root.run.spec_bdy_width,
                            cfg=exp.root.run, pair_restored=pair_restored)
    from woof.ingest.wrfinput_radar import prepare as prepare_radar_tendency
    radar_tendencies = prepare_radar_tendency(run, exp)
    for name, path in paths.items():
        if _sha(path) != original_hashes[name]:
            raise ValueError(f'{path}: changed while its input fields were read')
    directory, reused = preparation_directory(
        directory, receipt='wrf-import.json', source_files=original_hashes,
        documents={'experiment.toml': config.encode('utf-8')})
    directory.mkdir(parents=True, exist_ok=reused)
    config_path = directory/'experiment.toml'
    write_document(config_path, config.encode('utf-8'), reused=reused)
    receipt_path = directory/'wrf-import.json'
    receipt = {'schema':'gpuwm-wrf-input-import-v1', 'source_files':original_hashes,
               'initial_temperature':'T is dry perturbation potential temperature',
               'boundary_temperature':(
                   'moist coupled THM converted to dry theta at each forcing time'
                   if int(bundles[0].restored.global_attributes['USE_THETA_M']) == 1
                   else 'dry perturbation potential temperature'),
               'vertical_coordinate':'file ZNW, compared with explicit namelist eta when present',
               'initial_boundary_pair':'verified for every consumed field and side',
               'surface_input_dispositions': {
                   f'd{bundle.grid_id:02d}': dict(bundle.restored.surface_input_dispositions)
                   for bundle in bundles},
               'soil_unit_conversions': {
                   f'd{bundle.grid_id:02d}': dict(bundle.restored.soil_unit_conversions)
                   for bundle in bundles if bundle.restored.soil_unit_conversions},
               'soil_source_recovery': {
                   f'd{bundle.grid_id:02d}': dict(bundle.restored.soil_recovery)
                   for bundle in bundles if getattr(bundle.restored, 'soil_recovery', {})},
               **({'nest_initialization': {
                   f'd{bundle.grid_id:02d}': dict(bundle.nest_start.receipt)
                   for bundle in bundles if bundle.nest_start is not None}}
                  if any(bundle.nest_start is not None for bundle in bundles) else {}),
               **({'radar_tendency': {
                   f'd{gid:02d}': tendency.receipt()
                   for gid, tendency in sorted(radar_tendencies.items())}}
                  if radar_tendencies else {}),
               'namelist_translation':asdict(run.substitution_report),
               'namelist_translation_text':run.substitution_report.format()}
    if warm:
        clock = run.restart_clock
        receipt['initial_temperature'] = (
            'THM_2 converted to dry theta with QVAPOR (USE_THETA_M = 1)'
            if bundles[0].restored.warm_start.use_theta_m == 1
            else 'THM_2 is dry perturbation potential temperature')
        receipt['warm_start'] = {
            'restart_files': {f'd{gid:02d}': str(path)
                              for gid, path in run.restart_paths.items()},
            'restart_instant': clock.valid_time.strftime('%Y-%m-%d_%H:%M:%S'),
            'simulation_start': clock.simulation_start.strftime('%Y-%m-%d_%H:%M:%S'),
            'itimestep': clock.itimestep, 'dt_seconds': clock.dt_seconds,
            'elapsed_seconds': clock.elapsed_seconds,
            'dtbc_seconds': clock.dtbc_seconds,
            'declared_changes': list(run.restart_notes),
            'read_checks': {f'd{bundle.grid_id:02d}': dict(bundle.restored.warm_start.checks)
                            for bundle in bundles},
            'variable_dispositions': {
                f'd{bundle.grid_id:02d}': dict(bundle.restored.warm_start.dispositions)
                for bundle in bundles},
        }
    write_document(receipt_path, json_bytes(receipt), reused=reused)
    paths.update(experiment_config=config_path, preparation_receipt=receipt_path)
    hashes = {name:_sha(path) for name,path in paths.items()}
    return WrfTreeInputs(directory, config_path, exp, grids, tuple(bundles),
                         tuple((t-exp.start_time).total_seconds()/3600 for t in (*run.coverage.times, run.coverage.end)),
                         int(run.coverage.forcing_interval_seconds),
                         MappingProxyType({'handoff': 'WRF restart' if warm else 'WRF real.exe',
                                           'forcing_origin':'not declared by input files'}),
                         resolve_execution_plan(exp), MappingProxyType(hashes),
                         MappingProxyType(paths), boundaries,
                         radar_tendencies=MappingProxyType(radar_tendencies))


def announce_wrf_substitutions(run, receipt_path):
    """Show the actual scheme changes; retain the full translator record.

    A declared divergence (``item.reason``) prints its reason: the user is
    handing over a WRF run and must read, at the terminal, that WOOF will
    integrate a different variable and why.
    """
    for item in run.substitution_report.substitutions:
        print(f'WRF import: {item.wrf_name} ({item.key}={item.wrf_value}) '
              f'→ {item.gpuwm_name} ({item.gpuwm_key}={item.gpuwm_value}).'
              + (f'  Declared divergence: {item.reason}' if item.reason else ''),
              flush=True)
    if run.substitution_report.substitutions:
        print(f'WRF import details: {receipt_path}', flush=True)


def announce_warm_start(run, experiment=None):
    """Say at the terminal what a warm start continues, and what changed."""
    clock = getattr(run, 'restart_clock', None)
    if clock is None:
        return
    from datetime import timedelta
    experiment = run.experiment if experiment is None else experiment
    end = experiment.start_time + timedelta(seconds=float(experiment.run_seconds))
    print(f'WRF warm start: {clock.path} at '
          f'{clock.valid_time:%Y-%m-%d_%H:%M:%S}, step {clock.itimestep} of '
          f'{clock.dt_seconds:g} s from {clock.simulation_start:%Y-%m-%d_%H:%M:%S}; '
          f'the run ends {end:%Y-%m-%d_%H:%M:%S}',
          flush=True)
    for note in run.restart_notes:
        print(f'WRF warm start: declared change: {note}', flush=True)


def worker_exit_status(code):
    """Preserve a worker failure and name a terminating signal to the caller."""
    if code == 0:
        return 0
    import sys
    from woof.runplan import StageExitError
    print(str(StageExitError('WRF forecast worker', code)), file=sys.stderr)
    return min(255, 128 - code) if code < 0 else code


#: Where a WRF-input door's pictures go when no ``--render-dir`` was
#: given.  Named once, and claimed once: the early render of the first
#: committed frame and the finalize render at the end have to file into
#: ONE folder, or the receipt that licenses the finalize skip describes
#: a directory nothing looks in.
RENDER_ROOT_NAME = 'png'


def announce_render_readiness(door: str, *, announce: bool = True) -> str | None:
    """Say at plan review whether this install can draw, and how to fix it.

    Asked of :func:`woof.go_cli.render_extra_missing`, which is the one
    answer ``woof go``'s render stage already uses, so a WRF-input run
    and a chained run cannot disagree about whether this install draws.

    An absent renderer is ANNOUNCED, never refused: the forecast is
    worth running without pictures.  What it must not be is discovered
    at the end, which is what happens when the question is asked after
    the forecast: this is called before a card is selected or reserved,
    so the reader who needs `woof setup` reads it while nothing is
    running.

    ``announce=False`` asks the same question and keeps the answer
    quiet.  The supervised re-launch re-enters this door in a child
    process, after the parent has already printed the pair at plan
    review, and one run that cannot draw is one message and not two.
    """

    from woof.go_cli import render_extra_missing

    missing = render_extra_missing()
    if missing is not None and announce:
        print(f'{door}: this run will draw no pictures: {missing}', flush=True)
        print('  remedy: woof setup', flush=True)
        print('  # stages the rust render engine, which draws the full '
              'catalog', flush=True)
    return missing


def door_render_plan(outdir, *, render_products=None, render_dir=None,
                     init=None, can_draw=True) -> dict:
    """The plan dict this run's pictures are drawn from, claimed once.

    The very dict :func:`woof.go_cli.render_command` and
    :func:`woof.go_cli._render_stage` take, so a WRF-input run composes
    the same render command every other door composes and its products
    land where ``woof render`` looks for them.

    The run folder is claimed HERE, before the forecast opens, through
    :func:`woof.run_stamp.resolve` -- the single answer every front
    door asks.  ``render_command`` hardcodes ``--run-stamp off``
    (:func:`woof.run_stamp.stage_flags`), which is a stage saying the
    folder is claimed above it; something has to do the claiming, and a
    door that left it to the render stage would have got no folder and
    no ``latest-run.txt`` at all.  Claiming it before the run is what
    makes the early render and the finalize render share one folder.

    ``render_products = 'none'`` claims nothing: the stage skips itself
    with its own sentence, and an unclaimed folder is not created.
    ``can_draw=False`` claims nothing for the same reason and from the
    same answer the announcement above was made from: an install
    without the render engine draws nothing, and a stamped run folder
    per run that never receives a picture is a directory a reader opens
    for nothing.
    """

    from woof import run_stamp

    outdir = Path(outdir)
    root = (outdir / RENDER_ROOT_NAME if render_dir is None
            else Path(render_dir))
    plan = {'run': outdir, 'wrfout_dir': outdir / 'wrfout',
            'render': root, 'render_products': render_products}
    if not can_draw or str(render_products or '').strip().lower() == 'none':
        return plan
    plan['render'] = run_stamp.resolve(root, init=init, create=True,
                                       publish=False)
    return plan


def arm_door_first_products(plan: dict, *, outdir, started):
    """This door's renders of each committed frame as it lands, or ``None``.

    Built by the shared
    :func:`woof.prepared_single_domain_forecast._route_owned_first_products`
    rather than by a second decision here, so "did this run ask for
    pictures early?" is answered for these two doors exactly as it is
    answered for every other one: off by ABSENCE of a product spec,
    off for ``none``, on otherwise.
    """

    from types import SimpleNamespace

    from woof import prepared_single_domain_forecast as prepared_single

    return prepared_single._route_owned_first_products(
        SimpleNamespace(render_products=plan['render_products'],
                        render_dir=plan['render']),
        outdir=Path(outdir), observer=None, started=started)


def stop_door_renders(first_products, error: BaseException) -> None:
    """Stop this door's renders with a forecast that did not finish.

    A stop draws nothing more (the desktop and the terminal kill a run
    5 s after asking it to stop); a failure finishes drawing the frames
    it wrote rather than let the render die mid-picture with this
    process.  Nothing raises: the forecast's own failure is the news.
    """

    if first_products is None:
        return
    try:
        if isinstance(error, KeyboardInterrupt):
            getattr(first_products, 'halt', lambda: None)()
        else:
            first_products.wait()
    except BaseException:  # noqa: BLE001 - the failure is the news
        pass


def draw_door_products(plan: dict, *, first_products=None, door: str) -> bool:
    """Draw this run's pictures, then publish the pointer to them.

    The stage is :func:`woof.go_cli._render_stage` verbatim, trigger
    included: the early render is joined and its digest-verified frame
    is dropped from this stage's list, so a frame drawn at second 46 is
    not redrawn at the end.

    ``latest-run.txt`` is published HERE, and only once a PNG is on
    disk.  The stage cannot publish it -- it runs with ``--run-stamp
    off``, so :mod:`woof.render` allocates no folder of its own and its
    own pointer write is a no-op -- and a pointer written before
    anything was drawn names an empty folder, which is the one thing a
    script following the documented pointer must never read.
    """

    from types import SimpleNamespace

    from woof import go_cli, run_stamp

    drawn = go_cli._render_stage(
        plan, explain=False, door=door,
        observer=SimpleNamespace(first_products=first_products))
    folder = Path(plan['render'])
    if drawn and run_stamp.is_run_folder(folder) and next(folder.rglob('*.png'), None):
        run_stamp.record_latest(folder.parent, folder)
    return drawn


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wrfinput', type=Path, required=True, help='real.exe directory with wrfinput, wrfbdy and namelist.input')
    parser.add_argument('--soil-source', type=Path, default=None, metavar='DIR',
                        help='original met_em and producing Vtable directory for automatic soil recovery; defaults to the WRF input directory')
    parser.add_argument('--rrtmg-variant', choices=('rrtmg_legacy','rte-rrtmgp'), default=None, help='preserve WRF RRTMG by default; choose rte-rrtmgp explicitly to change radiation')
    parser.add_argument('--thompson-version', choices=('wrf_461','wrf_44'), default=None)
    parser.add_argument('--urban-ucm-generation', choices=('wrf_471','wrf_480'), default=None)
    parser.add_argument('--urban-treeoption', type=int, choices=(0,1), default=None)
    parser.add_argument('--urban-ttscheme', type=int, choices=(0,1), default=None)
    parser.add_argument('--urban-fvg', type=float, default=None)
    parser.add_argument('--urban-table-dir', default=None)
    parser.add_argument('--outdir', type=Path, required=True)
    parser.add_argument('--run-seconds', type=float, help='shorten the run inside the supplied boundary coverage')
    parser.add_argument('--io-mode', choices=('history','none'), default='history')
    parser.add_argument('--restart', type=Path)
    parser.add_argument('--restart-from', type=Path, default=None, metavar='WRFRST',
                        help='continue a WRF run from its wrfrst_d01 restart file (warm start)')
    parser.add_argument('--checkpoint-at', default=None, metavar='TIMES',
                        help='write only these scheduled checkpoints: comma-separated WRF valid times '
                             '(2022-06-13_22:00:00) or `end`; each must be a restart_interval instant of '
                             'the run.  Absent writes every scheduled checkpoint')
    parser.add_argument('--health-debug', action='store_true')
    from woof.ensemble.door import add_arguments
    add_arguments(parser)
    parser.add_argument('--ensemble-request', default=None, help=argparse.SUPPRESS)
    parser.add_argument('--products', dest='render_products', default=None, metavar='LIST',
                        help='which product plots this run draws, in woof render\'s own spelling: '
                             'a comma-separated list, `all`, or `none` for no pictures.  '
                             'Absent draws the default catalog at the end of the run')
    parser.add_argument('--render-dir', type=Path, default=None, metavar='DIR',
                        help='where this run\'s pictures go; defaults to OUTDIR/' + RENDER_ROOT_NAME)
    add_progress_arguments(parser)
    parser.add_argument('--allow-shared-gpu', action='store_true',
                        help='proceed when another CUDA compute process holds the selected GPU. The requested configuration is retained')
    parser.add_argument('--benchmark', action='store_true',
                        help='time the forecast as a benchmark: no products, no post-run '
                             'verification, then a read, init, first step, steps and write summary')
    from woof.da.analysis_door import add_analysis_arguments
    add_analysis_arguments(parser, worker=True)
    parser.add_argument('--_worker', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--_output-owner', help=argparse.SUPPRESS)
    return parser


DOOR = 'woof run --wrfinput'


def _door_initialization(initialization, plan_path, outdir):
    """``initialization``, or the analysis door's wrapper of it when a plan was resolved."""
    if plan_path is None:
        return initialization
    from woof.da.analysis_door import AnalysisInitialization, load_plan
    return AnalysisInitialization(initialization, load_plan(plan_path), outdir=outdir)


def run_wrf_forecast(directory, outdir, *, run_seconds=None, restart=None,
                     io_mode="history", health_debug=False, gpu_uuid=None,
                     exclusive_gpu=True, rrtmg_variant=None,
                     thompson_version=None, urban_ucm_generation=None,
                     urban_controls=None,
                     render_products=None, render_dir=None,
                     progress_options=None, relaunched=False, allow_shared_gpu=False,
                     output_owner=None, soil_source=None, ensemble_request=None,
                     restart_from=None, benchmark=False, analysis=None,
                     checkpoint_at=None):
    import os
    import subprocess
    import sys
    import time
    from woof.stage_reuse import claim_run_output
    from woof.wrfinput_door import resolve_wrfinput_run
    started = time.perf_counter()
    from woof import checkpoint_schedule
    # Parsed here, at plan review, so a malformed value is refused before
    # a card is reserved; resolved against the run's clock in the worker.
    checkpoint_instants = checkpoint_schedule.parse(checkpoint_at)
    marks = None
    if benchmark:
        # NV recipe dry run, D6: a timed run draws nothing and verifies
        # nothing after the forecast, and says where its wall clock went.
        from woof import benchmark_timing
        if render_products is not None and str(render_products).strip().lower() != 'none':
            raise ValueError(f'--benchmark draws no products; --products {render_products} '
                             'asks for some. Drop one of the two')
        render_products = 'none'
        marks = benchmark_timing.Marks()
        marks.mark('read_start')
    from woof.ensemble.door import request_for_inputs, production_run_scope
    request = request_for_inputs(override=ensemble_request)
    # THE ANALYSIS DOOR (woof.da.analysis_door).  The parent checks the
    # bundle against the run and writes the plan, rewriting the run's inputs
    # (the analysis or its background as the wrfinput, the boundary rebased
    # onto it); the worker reads the plan and applies or forces the
    # increment once the tree is restored.  Resolved before anything else
    # reads the directory, so every check below is the rewritten run's.
    analysis_plan = None
    fresh_analysis_work = None
    if analysis is not None and analysis.get('plan') is not None:
        from woof.da.analysis_door import load_plan
        analysis_plan = Path(analysis['plan'])
        load_plan(analysis_plan)
    elif analysis is not None:
        from woof.da.analysis_door import resolve as resolve_analysis
        if request is not None:
            raise ValueError('--analysis starts one forecast from one analysis; a member roster runs one '
                             'process per member, each with its own bundle (the cycle driver does)')
        resolved = resolve_analysis(analysis, directory=Path(directory), outdir=Path(outdir),
                                    run_seconds=run_seconds, restart=restart,
                                    restart_from=restart_from, soil_source=soil_source)
        if resolved['plan'] is not None:
            if restart is None:
                fresh_analysis_work = Path(resolved['plan']).parent
            directory = resolved['directory']
            soil_source = resolved['soil_source']
            analysis_plan = Path(resolved['plan'])
            defaults = resolved['forecast_defaults']
            if rrtmg_variant is None and 'rrtmg_variant' in defaults:
                rrtmg_variant = defaults['rrtmg_variant']
            if thompson_version is None and 'thompson_version' in defaults:
                thompson_version = defaults['thompson_version']
            if urban_ucm_generation is None and 'urban_ucm_generation' in defaults:
                urban_ucm_generation = defaults['urban_ucm_generation']
            for key in ('urban_treeoption', 'urban_ttscheme', 'urban_fvg'):
                if key in defaults and key not in (urban_controls or {}):
                    urban_controls = dict(urban_controls or {})
                    urban_controls[key] = defaults[key]
    from woof.ensemble.calibration_admission import refuse_native_random
    refuse_native_random(directory)
    run = resolve_wrfinput_run(directory, rrtmg_variant=rrtmg_variant,
                               restart_from=restart_from,
                               urban_ucm_generation=urban_ucm_generation, urban_controls=urban_controls,
                               **({} if thompson_version is None else {"thompson_version":thompson_version}))
    if run_seconds is not None:
        wrfinput_window_seconds(run, run_seconds)
    from woof.tables import _ensure_for_load as ensure
    ensure(run.experiment)
    # PLAN REVIEW, and it happens here rather than in the worker child
    # for one reason: everything below this in the supervised branch
    # takes a card.  A bad --outdir used to reach the reader as
    # [Errno 17] from inside the child, after select_gpu, GPUFileLock
    # and preflight_exclusive_gpu had reserved a GPU for a run that was
    # never going to start.
    try:
        output_claim = claim_run_output(outdir, flag='--outdir',
                                  protected_roots=(Path(directory),),
                                  resume=restart, owner_token=output_owner)
    except (ValueError, FileExistsError) as error:
        print(f'{DOOR}: --outdir refused: {error}', file=sys.stderr)
        if fresh_analysis_work is not None:
            import shutil
            shutil.rmtree(fresh_analysis_work, ignore_errors=True)
        return 2
    outdir = output_claim.path
    try:
        # Identity review includes the cold versus stepped surface-state
        # contract, before a supervised worker selects or reserves a card.
        # `relaunched` is the supervised child of the branch below, which
        # has already said this at plan review in the terminal both
        # processes print to.
        missing = announce_render_readiness(DOOR, announce=not relaunched)
        if exclusive_gpu:
            from woof.supervisor import priced_reservation_bytes
            from woof.ensemble.supervised_devices import input_device_lease
            command = [sys.executable, '-m', 'woof.wrfinput_forecast',
                       '--wrfinput', str(Path(directory).resolve()),
                       '--outdir', str(outdir), '--io-mode', io_mode, '--_worker',
                       '--_output-owner', output_claim.token]
            if rrtmg_variant is not None:
                command += ['--rrtmg-variant', rrtmg_variant]
            if thompson_version is not None:
                command += ['--thompson-version', thompson_version]
            if urban_ucm_generation is not None:
                command += ['--urban-ucm-generation', urban_ucm_generation]
            for key,value in (urban_controls or {}).items():
                command += ['--'+key.replace('_','-'), str(value)]
            if soil_source is not None:
                from woof.filesystem_paths import canonical_path
                command += ['--soil-source', str(canonical_path(soil_source))]
            if run_seconds is not None:
                command += ['--run-seconds', str(run_seconds)]
            if restart is not None:
                command += ['--restart', str(Path(restart).resolve())]
            if restart_from is not None:
                command += ['--restart-from', str(Path(restart_from).resolve())]
            if health_debug:
                command += ['--health-debug']
            if request is not None:
                import json
                command += ['--ensemble-request', json.dumps(request.receipt())]
            # The worker is this door's real body, so a flag this door was
            # given and did not pass on is a flag that did nothing.
            if render_products is not None:
                command += ['--products', str(render_products)]
            if render_dir is not None:
                command += ['--render-dir', str(Path(render_dir).resolve())]
            if benchmark:
                command += ['--benchmark']
            if checkpoint_instants is not None:
                command += ['--checkpoint-at', ','.join(checkpoint_instants)]
            if analysis_plan is not None:
                command += ['--analysis-plan', str(analysis_plan)]
            command += ProgressOptions.worker_flags(progress_options)
            urban_columns = wrfinput_urban_columns(run)
            pricing = ({} if urban_columns is None else
                       {'urban_columns': urban_columns})
            # An explicit hidden-device mask is a process constraint. Do
            # not select/reserve a physical card or reveal it on relaunch.
            from contextlib import nullcontext
            inherited_mask = os.environ.get('CUDA_VISIBLE_DEVICES')
            hidden = inherited_mask is not None and inherited_mask.strip() in ('', '-1')
            lease = (nullcontext(inherited_mask) if hidden else input_device_lease(
                request, gpu_uuid=gpu_uuid, run_id=f'wrf-input-{os.getpid()}',
                allow_shared_gpu=allow_shared_gpu,
                reservation_bytes=priced_reservation_bytes(run.experiment, **pricing)))
            with lease as cuda_mask:
                # Priced against THIS run's reservation through the same
                # function `woof run` prices from, so a co-tenant admitted
                # at one door is admitted at the other.
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=cuda_mask)
                if marks is not None:
                    # The door's own share of the start-up, for the worker's
                    # summary (benchmark_timing.DOOR_MARKS_ENV).
                    env[benchmark_timing.DOOR_MARKS_ENV] = benchmark_timing.door_marks(
                        marks, spawn=time.time())
                status = worker_exit_status(subprocess.run(
                    command, env=env, check=False).returncode)
            if benchmark and status == 0:
                benchmark_timing.report(
                    outdir, process_start=benchmark_timing.process_start_unix())
            return status
        if gpu_uuid is not None:
            raise ValueError('--gpu-uuid requires the default fresh worker; remove --no-supervise')
        from woof.go_cli import GoStageFailed
        from woof.prepared_domain_tree_forecast import run_prepared_tree
        from woof.filesystem_paths import io_path
        worker_output = io_path(outdir)
        # NV recipe dry run, D5: step 1 compiled the dycore's diffusion
        # units (32 s of a 33.5 s first step on a warm kernel cache);
        # they now compile on a background thread while the input is read.
        from woof import kernel_prewarm
        # Prebuilt kernel images under a mounted cache (woof.kernel_cache_seed);
        # nothing when WOOF_KERNEL_CACHE_SEED is unset.
        from woof import kernel_cache_seed, kernel_prebuild
        kernel_cache_seed.seed()
        # WOOF_KERNEL_RECORD: what this run compiles, for a prebuild on
        # other cards (woof.kernel_prebuild); nothing when unset.
        kernel_prebuild.install_recorder()
        # The configuration's whole kernel set compiles on every core while
        # the input is read (woof.kernel_plan), not one module at a time
        # inside step 1.
        kernel_prewarm.start_plan(getattr(run, 'experiment', None))
        prewarm = kernel_prewarm.start_for(getattr(run, 'experiment', None))
        plan_document = None
        if analysis_plan is not None:
            from woof.da.analysis_door import load_plan
            plan_document = load_plan(analysis_plan)
        if marks is not None:
            marks.mark('prepare_start')
        inputs = prepare_wrf_run(run, worker_output/'input', run_seconds=run_seconds,
                                 **({} if soil_source is None else {'soil_source': soil_source}),
                                 **({} if plan_document is None or not plan_document.get('boundary_pair_file')
                                    else {'boundary_pair_file': plan_document['boundary_pair_file']}))
        if marks is not None:
            marks.mark('read_end')
        announce_wrf_substitutions(run, inputs.prepared_root/'wrf-import.json')
        announce_warm_start(run, inputs.experiment)
        plan = door_render_plan(worker_output, render_products=render_products,
                                render_dir=None if render_dir is None else io_path(render_dir),
                                init=inputs.experiment.start_time,
                                can_draw=missing is None)
        with production_run_scope(request, output_directory=worker_output):
            first_products = arm_door_first_products(plan, outdir=worker_output, started=started)
            try:
                with checkpoint_schedule.selected(checkpoint_instants):
                    run_prepared_tree(inputs, output_directory=worker_output, io_mode=io_mode,
                                    restart=None if restart is None else io_path(restart),
                                    health_debug=health_debug,
                                    progress_options=progress_options,
                                    initialization=_door_initialization(
                                        (WrfRestartInitialization(inputs)
                                         if getattr(run, 'restart_paths', None) is not None
                                         else WrfInitialization(inputs)),
                                        analysis_plan, worker_output),
                                    **({} if first_products is None
                                       else {'first_products': first_products}))
            except BaseException as error:
                stop_door_renders(first_products, error)
                raise
            if marks is not None:
                if prewarm is not None:
                    marks.values['kernel_prewarm'] = prewarm.receipt()
                from woof import kernel_plan
                marks.values['kernel_plan'] = kernel_plan.receipt()
                marks.write(worker_output)
                if not relaunched:
                    benchmark_timing.report(
                        worker_output, process_start=benchmark_timing.process_start_unix())
                return 0
            try:
                draw_door_products(plan, first_products=first_products, door=DOOR)
            except GoStageFailed as failure:
                # The forecast is on disk and finished; the pictures are not.
                print(f'{DOOR}: the forecast finished and its output is in {outdir}; '
                      f'the render stage exited {failure.code}.', file=sys.stderr)
                return failure.code
        return 0
    finally:
        output_claim.close()


def analysis_request(args):
    from woof.da.analysis_door import request_from_args
    return request_from_args(args)


def main(argv=None, *, in_process=False):
    """The door.  ``in_process=True`` runs the forecast in this process (the
    worker's body, as ``woof run --no-supervise`` does) instead of a fresh
    supervised worker: the warm leg executor (:mod:`woof.leg_executor`)
    runs successive legs this way on the card it was started on."""
    args = build_parser().parse_args(argv)
    try:
        from woof.ensemble.calibration_admission import refuse_public_arguments
        refuse_public_arguments(args)
        from woof.provenance_gate import announce
        announce(DOOR)
        import json
        from woof.ensemble.door import request_for_inputs
        request = request_for_inputs(
            override=None if args.ensemble_request is None else json.loads(args.ensemble_request),
            members=args.members, keep_member_files=args.keep_member_files)
        return run_wrf_forecast(args.wrfinput, args.outdir, run_seconds=args.run_seconds,
                               restart=args.restart, io_mode=args.io_mode,
                               health_debug=args.health_debug,
                               exclusive_gpu=not (args._worker or in_process),
                               rrtmg_variant=args.rrtmg_variant,
                               thompson_version=args.thompson_version,
                               urban_ucm_generation=args.urban_ucm_generation,
                               urban_controls={key:getattr(args,key) for key in ("urban_treeoption","urban_ttscheme","urban_fvg","urban_table_dir") if getattr(args,key) is not None},
                               render_products=args.render_products,
                               render_dir=args.render_dir,
                               progress_options=ProgressOptions.from_args(args),
                               relaunched=args._worker,
                               allow_shared_gpu=args.allow_shared_gpu,
                               output_owner=args._output_owner, soil_source=args.soil_source,
                               restart_from=args.restart_from, benchmark=args.benchmark,
                               analysis=analysis_request(args),
                               checkpoint_at=args.checkpoint_at,
                               **({} if request is None else {"ensemble_request": request.receipt()}))
    except (ValueError, OSError) as error:
        import sys
        print(f'{DOOR}: {error}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
