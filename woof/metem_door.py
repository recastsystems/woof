"""Resolve a WPS metgrid directory through the existing namelist translator."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import numpy as np

from woof.ingest.metem import (met_em_series, met_em_source_top_pressure_pa,
                                parse_met_em_name)
from woof.netcdf_bridge import open_dataset


@dataclass(frozen=True)
class MetemRun:
    directory: Path
    namelist_input: Path
    paths: dict[int, tuple[Path, ...]]
    metadata: dict[int, object]
    experiment: object
    toml_text: str
    substitution_report: object
    interval_seconds: int
    controls: dict
    #: How far the root domain's met_em series forces the run, in seconds
    #: from ``experiment.start_time``.  This, and NOT the producing
    #: namelist's ``run_seconds``, is what limits the window a caller may
    #: ask for: a directory holding 48 h of forcing is a 48 h directory
    #: whatever number the namelist that made it happened to carry.
    #: Derived from file NAMES only (``parse_met_em_name``), so it is
    #: available at plan review with no file read.
    coverage_seconds: float = 0.0
    #: The smallest analyzed pressure in the root series, in pascals.  A
    #: requested model top above it is a ladder this source cannot
    #: support, and carrying the number here is what lets that be refused
    #: at plan review instead of inside the worker.
    source_top_pressure_pa: float | None = None
    #: What ``--lake-state`` chose, as one sentence for plan review; None
    #: when the flag was not given (the run keeps lake_state_source=auto).
    lake_state_note: str | None = None


def _metadata(path):
    """Read layout authority without decoding atmospheric payloads."""
    with open_dataset(path) as ds:
        attrs = dict(ds.global_attributes)
        for name in ('GRID_ID', 'PARENT_ID', 'I_PARENT_START', 'J_PARENT_START',
                     'PARENT_GRID_RATIO'):
            if name not in attrs and name.lower() in attrs:
                attrs[name] = attrs[name.lower()]
        for name in ('GRID_ID', 'PARENT_ID', 'I_PARENT_START', 'J_PARENT_START',
                     'PARENT_GRID_RATIO', 'MAP_PROJ'):
            value = attrs.get(name)
            if value is None or not np.isfinite(value) or float(value) != int(value):
                raise ValueError(f'{path.name}: missing or noninteger layout attribute {name}')
        _, time = parse_met_em_name(path)
        return SimpleNamespace(path=path, global_attributes=attrs,
            nx=len(ds.dimensions['west_east']), ny=len(ds.dimensions['south_north']),
            start_date=time.strftime('%Y-%m-%d_%H:%M:%S'),
            variables={name:tuple(var.shape) for name,var in ds.variables.items()})


def resolve_metem_run(directory, *, rrtmg_variant=None, lake_state=None, lake_state_path=None):
    from woof.experiment import build_experiment
    from woof.namelist_import import import_namelists, parse_namelist
    from woof.wrfinput_door import (refuse_latlon_files, synthesize_wps_namelist,
                                     require_preserved_wrf_selectors)
    import tomllib

    directory = Path(directory).resolve()
    namelist = directory/'namelist.input'
    if not directory.is_dir() or not namelist.is_file():
        raise ValueError('--met-em DIR requires met_em.d0*.nc and the producing namelist.input in DIR')
    ids = set()
    for path in directory.glob('met_em.d*'):
        domain, _ = parse_met_em_name(path)
        ids.add(int(domain[1:]))
    if not ids or ids != set(range(1, max(ids)+1)):
        raise ValueError('met_em domain files must include d01 and every consecutive child domain')
    paths = {gid:met_em_series(directory, f'd{gid:02d}') for gid in sorted(ids)}
    metadata = {gid:_metadata(files[0]) for gid,files in paths.items()}
    refuse_latlon_files(metadata)
    root_attrs = metadata[1].global_attributes
    for gid, item in metadata.items():
        for key in ('MMINLU', 'NUM_LAND_CAT'):
            if key not in item.global_attributes:
                raise ValueError(f'{item.path.name}: missing land-use identity {key}')
        if item.global_attributes['GRID_ID'] != gid:
            raise ValueError(f'{item.path.name}: grid_id disagrees with filename')
    with TemporaryDirectory(prefix='gpuwm-metgrid-') as temporary:
        wps_path = Path(temporary)/'namelist.wps'
        wps_path.write_text(synthesize_wps_namelist(metadata), encoding='utf-8')
        text, report = import_namelists(wps_path, namelist, rrtmg_variant=rrtmg_variant,
            landuse_identity={key:root_attrs[key] for key in ('MMINLU', 'NUM_LAND_CAT')},
            metgrid_initialization=True)
    require_preserved_wrf_selectors(report)
    # --lake-state, through the one function the wrfinput door calls too.
    from woof.ingest.lake_state import select_lake_state
    text, lake_state_note = select_lake_state(text, lake_state, lake_state_path)
    exp = build_experiment(tomllib.loads(text), source=str(namelist))
    if tuple(domain.grid_id for domain in exp.domains) != tuple(paths):
        raise ValueError('namelist max_dom differs from the met_em domain inventory')
    parsed = parse_namelist(namelist)
    interval = parsed['time_control'].get('interval_seconds', [10800])[0]
    if isinstance(interval, bool) or not np.isfinite(interval) or interval <= 0 or int(interval) != interval:
        raise ValueError('interval_seconds must be a positive integer')
    for domain in exp.domains:
        files = paths[domain.grid_id]
        times = [parse_met_em_name(path)[1] for path in files]
        if not times or times[0] != exp.start_time or (domain.parent_id == 0 and len(times) < 2):
            raise ValueError(f'd{domain.grid_id:02d}: met_em must start at the namelist start and include a later boundary time')
        if any((b-a).total_seconds() != interval for a,b in zip(times,times[1:])):
            raise ValueError(f'd{domain.grid_id:02d}: met_em times must follow interval_seconds={interval}')
        if domain.parent_id == 0 and times[-1] < exp.start_time + timedelta(seconds=exp.run_seconds):
            raise ValueError(f'd{domain.grid_id:02d}: met_em stops at {times[-1]:%Y-%m-%d_%H:%M:%S}, the last valid time in the series, but the namelist asks for {exp.run_seconds:.0f} s ending {exp.start_time+timedelta(seconds=exp.run_seconds):%Y-%m-%d_%H:%M:%S}; supply forcing through the requested end, or shorten the namelist duration')
        item = metadata[domain.grid_id]
        if (domain.run.nx, domain.run.ny) != (item.nx, item.ny):
            raise ValueError(f'd{domain.grid_id:02d}: namelist dimensions differ from met_em')
        # At plan review, so `woof check --met-em` and `woof run --met-em`
        # give one answer: the run's land model reads LAI12M only with
        # rdlai2d (woof.metem_forecast.metgrid_soil takes VEGPARM.TBL leaf
        # area otherwise, as WRF does).
        if metgrid_run_reads_lai12m(domain.run) and 'LAI12M' not in item.variables:
            raise ValueError(f'{item.path.name}: land-surface initialization needs [\'LAI12M\'] '
                             '(rdlai2d = .true. makes the land model read the monthly LAI12M field; '
                             'with rdlai2d = .false. it takes leaf area from VEGPARM.TBL); '
                             'retain these metgrid/geogrid fields')
    return MetemRun(directory, namelist, paths, metadata, exp, text, report, int(interval), parsed,
                    metem_coverage_seconds(paths, exp.start_time),
                    met_em_source_top_pressure_pa(paths[1][0]), lake_state_note)


def metgrid_run_reads_lai12m(cfg):
    """Does this run's land model read the monthly LAI12M field?

    Only Noah (2) and RUC (3) with ``rdlai2d`` do (woof/config.py refuses
    the flag for any other land model); every other run takes leaf area from
    its tables, exactly as WRF does.  It lives here, in the door the
    preparation-only package carries, so plan review needs no forecast module
    (THE BREAKAGE: the RW-WPS wheel staging refused metem_door.py importing
    woof.metem_forecast, an excluded executor module).
    """
    return int(getattr(cfg, 'sf_surface_physics', 0)) in (2, 3) and bool(getattr(cfg, 'rdlai2d', False))


def metem_coverage_seconds(paths, start_time):
    """How far the root met_em series forces, from its file NAMES alone.

    ``parse_met_em_name`` reads the name and nothing else
    (woof/ingest/metem.py:130-145), so this is a plan-review quantity: no
    file is opened and no boundary array is decoded to learn it.
    """
    times = [parse_met_em_name(path)[1] for path in paths[1]]
    return (times[-1] - start_time).total_seconds()


def metgrid_initialization_controls(case, run, *, cfg=None):
    """Resolve WRF's actual preparation branches from controls and field flags."""
    from woof.ingest.real import DECLARED_ANALYZED_HYDROMETEORS
    from woof.ingest.analyzed_numbers import METGRID_NUMBER_FIELDS
    if cfg is not None:
        check_analyzed_scalar_capability(case.attributes, cfg)
    values = run.controls.get('domains', {})
    requested = values.get('sfcp_to_sfcp', [False])
    use_sh = values.get('use_sh_qv', [False])
    for name, value in (('sfcp_to_sfcp', requested), ('use_sh_qv', use_sh)):
        if len(value) != 1 or not isinstance(value[0], bool):
            raise ValueError(f'{name} requires one logical value')
    # WRF real.F:1213 forces sfcprs2 for an ascending (hybrid) input
    # pressure stack. Use the file's actual order, before native sorting.
    ascending = bool(np.all(np.diff(case.snapshot.levels_hpa) > 0))
    sfcp = requested[0] or ascending
    operation = 'sfcprs2' if sfcp else 'sfcprs3'
    flags = ('PSFC', 'SOILHGT') if sfcp else ('PSFC', 'SOILHGT', 'SLP')
    for flag in flags:
        if case.attributes.get('FLAG_'+flag) != 1:
            raise ValueError(f'{case.path.name}: {operation} requires FLAG_{flag}=1')
    if not sfcp and 'PMSL' not in case.snapshot.fields:
        raise ValueError(f'{case.path.name}: sfcprs3 requires the declared PMSL field')
    if 'num_metgrid_levels' in values and values['num_metgrid_levels'] != [case.geometry['num_metgrid_levels']]:
        raise ValueError('num_metgrid_levels differs from the file vertical dimension')
    from woof.ingest.metem import RH_DERIVED_LANE
    lane = getattr(case, 'lane', None)
    extra = {}
    if lane in ('rh', RH_DERIVED_LANE):
        # A file with RH and neither FLAG_QV nor FLAG_SH: real.exe sets
        # use_sh_qv = .FALSE. itself (module_initialize_real.F:1119), so
        # the namelist's value cannot select the direct-qv lane here.
        use_sh = [False]
    if lane == RH_DERIVED_LANE:
        # Nor does real.exe make its FLAG_SH surface test (:1150, SH at the
        # first point below 1e-6 replaces the surface level) on an RH file.
        extra['flag_sh_surface_fallback'] = False
    return {'sfcp_to_sfcp':sfcp, 'use_sh_qv':use_sh[0],
            'analyzed_species':tuple(name for name in DECLARED_ANALYZED_HYDROMETEORS if name in case.snapshot.fields),
            'analyzed_surface_fields':tuple(name for name in DECLARED_ANALYZED_HYDROMETEORS if name in case.snapshot.fields),
            'analyzed_number_fields':tuple(name for name in METGRID_NUMBER_FIELDS if name in case.snapshot.fields),
            **extra}


def check_analyzed_scalar_capability(attributes, cfg):
    """Refuse only flagged state that the selected active package would lose.

    QNWFA and QNIFA are rows of ``analyzed_numbers.METGRID_NUMBER_FIELDS``
    now, so they need no arm here: a package that transports nwfa/nifa
    keeps them through the same generic route as QNI/QNC/QNR, and a
    package that does not discards them exactly like an inactive P_QN*.
    QNBCA is the one gap that is real, and it was the one this function
    could never reach: no package in this build declares an ``nbca``
    species, so intersecting the alias with ``nest_field_kinds`` was
    always empty.  It is tested directly instead.
    """
    from woof.ingest.analyzed_numbers import METGRID_NUMBER_FIELDS
    for name in (*METGRID_NUMBER_FIELDS, 'QNBCA'):
        if attributes.get('FLAG_'+name, 0) not in (0, 1):
            raise ValueError(f'FLAG_{name} must be 0 or 1')
    if attributes.get('FLAG_QNBCA', 0) == 1 and int(cfg.mp_physics) == 28:
        raise ValueError('FLAG_QNBCA=1 supplies an analyzed black-carbon number, and this build\'s mp=28 package carries no qnbca species (Registry/registry.new3d_wif:82; absent from core/moist.py:119 and core/state.py), so the field would be silently dropped. Regenerate met_em without QNBCA, or run mp=28 with the water/ice-friendly pair alone')


def check_analyzed_scalar_capabilities(run, exp=None):
    """Ask the capability question of every domain of a resolved run.

    THE ONE FUNCTION BOTH DOORS CALL, and the reason it exists: the
    per-domain check below was reachable only from
    :func:`metgrid_initialization_controls` and
    :func:`metgrid_memory_admission`, both of which run inside
    ``prepare_metem_run`` -- after ``select_gpu``, after the GPU file lock
    and after the output directory was created.  A refusal that fires
    there fires after the run has started.  Everything it needs is in the
    resolved ``MetemRun``, so ``run_metem_forecast`` asks it at plan
    review and the worker asks the same function of the same bytes.

    ``exp`` defaults to the run's own resolved experiment; the preparation
    passes the experiment it rebuilt from the substituted TOML, so the
    domains checked are the domains that will be built.
    """

    experiment = run.experiment if exp is None else exp
    for domain in experiment.domains:
        check_analyzed_scalar_capability(
            run.metadata[domain.grid_id].global_attributes, domain.run)


def metgrid_analysis_shapes(metadata):
    """The actual fields shared initialize_real uploads, from Rust inventory."""
    attrs, variables = metadata.global_attributes, metadata.variables
    specific = attrs.get('FLAG_SH') == 1
    shapes = {}
    pairs = [('TT','T2'),('GHT',None),('UU','U10'),('VV','V10')]
    pairs += [('SPECHUMD','Q2'),('PRES',None)] if specific else [('RH','RH2')]
    if not specific and 'PRES' in variables:
        # An RH file on levels that are not isobaric uploads the derived
        # specific humidity in RH's place AND its 3-D PRES
        # (woof.ingest.metem.RH_DERIVED_LANE).  The inventory cannot tell
        # isobaric PRES from model-level PRES without decoding it, so PRES
        # is priced for every RH file: one level stack over on a
        # pressure-level file, never under on a model-level one.
        pairs.append(('PRES', None))
    from woof.ingest.analyzed_numbers import METGRID_NUMBER_FIELDS
    pairs += [(name,name+'_SFC') for name in ('QC','QR','QI','QS','QG','QH') if attrs.get('FLAG_'+name) == 1]
    pairs += [(name,name+'_SFC') for name in METGRID_NUMBER_FIELDS if attrs.get('FLAG_'+name) == 1]
    for name, surface in pairs:
        shape = variables.get(name)
        if shape is None or len(shape) != 4 or shape[0] != 1 or shape[1] < 3:
            raise ValueError(f'{metadata.path.name}: {name} needs Time/level/y/x inventory for native initialization')
        shapes[name] = (shape[1]-1,*shape[2:])
        if surface is not None: shapes[surface] = tuple(shape[2:])
    for name in ('PSFC','SOILHGT', *(('PMSL',) if attrs.get('FLAG_SLP') == 1 else ())):
        shape=variables.get(name)
        if shape != (1,metadata.ny,metadata.nx):
            raise ValueError(f'{metadata.path.name}: {name} requires the mass-grid surface inventory')
        shapes[name] = shape[1:]
    return shapes


def metgrid_memory_admission(run, exp, *, preprocess_backend=None):
    """Validate analyzed fields and report the advisory forecast memory estimate.

    The preparation's own fit is decided, not advised, by the price
    :func:`woof.metem_forecast.prepare_metem_run` weighs before it
    allocates; this receipt still records its phase estimate.
    """
    from woof.core.preflight import (device_memory_probe_subprocess, device_memory_probe_reason,
        profile_from_device_probe, estimate_phases)
    inventory = {gid:metgrid_analysis_shapes(item) for gid,item in run.metadata.items()}
    check_analyzed_scalar_capabilities(run, exp)
    probe = device_memory_probe_subprocess()
    free = None if probe is None else int(probe['free_bytes'])
    profile = profile_from_device_probe(probe)
    from woof.core.streaming import planner_machine
    from woof.preprocess_policy import resolve_preprocess_backend
    road = resolve_preprocess_backend(source='met_em', experiment=exp,
                                     requested=preprocess_backend)
    machine = planner_machine(vram_bytes=free, name='met_em estimate probe',
                              device_profile=profile)
    phases = estimate_phases(exp,source='met_em',forcing_interval_seconds=run.interval_seconds,
        ingest_forcing_interval_seconds=run.interval_seconds,forcing_intervals=len(run.paths[1])-1,
        analysis_shapes_by_domain=inventory,sequential_domains=True,
        profile=profile, machine=machine, preprocess_backend=road)
    # The FORECAST's advisory only.  The preparation is no longer advised
    # about: prepare_metem_run prices it and decides before it allocates
    # (auto to the CPU, explicit cuda refused), so an over-budget
    # preparation stopped being something this could only warn about.
    device_over = free is not None and phases.forecast_envelope_bytes > free
    host = phases.streamed
    # The one streamed host admission `woof go`, `woof check` and `woof
    # domain` read, so this advisory cannot pass a store they refuse.
    host_over = host is not None and phases.streamed_host_refusal() is not None
    warnings = []
    if device_over:
        warnings.append('The estimated forecast peak exceeds currently free VRAM. The requested settings are retained; allocation will report any actual memory failure. Free other memory or choose different settings if needed.')
    if host_over:
        warnings.append('The estimated streamed forecast store exceeds available host RAM. The requested settings are retained; allocation will report any actual memory failure. Free other memory or choose different settings if needed.')
    return {'policy':'advisory', 'warnings':warnings, 'preprocess_backend':road,
        'device_estimate_exceeds_available':device_over, 'host_estimate_exceeds_available':host_over,
        'host_store_bytes':None if host is None else host.host_bytes,
        'host_budget_bytes':None if host is None else host.host_budget_bytes,
        'analysis_shapes_by_domain':inventory, 'forecast_peak_bytes':phases.forecast_envelope_bytes,
        'preparation_peak_bytes':phases.ingest_envelope_bytes,'available_device_bytes':free,
        'device_probe_note':None if probe is not None else device_memory_probe_reason(),
        'retained_boundary_intervals':len(run.paths[1])-1,'sequential_domain_preparation':True}
