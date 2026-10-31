"""Seal one ERA5 trajectory for a heterogeneous prepared ensemble roster.

The registry owns this hook. Acquisition and decoding use the ordinary ERA5
owners; the direct native preparer publishes the same portable bundle as prep.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tomllib


def configure_member(raw, *, target, original):
    from woof.domain_wizard import _PACKAGED_VTABLE
    from woof.fetch import era5_combined_name
    from woof.geog_assets import default_geog_root
    fetch = raw['fetch']
    fetch.setdefault('era5_provider', 'cds' if fetch.get('member') is not None else 'arco')
    data = raw.setdefault('case_data', {})
    # An inherited case table's relative authorities belong to the original
    # configuration, never the scratch folder of another member.
    for key in ('vtable', 'source_orography', 'geog_root'):
        value = data.get(key)
        if isinstance(value, str) and '${' not in value:
            data[key] = str((original.parent / value).resolve())
    data.setdefault('vtable', str(_PACKAGED_VTABLE.resolve()))
    data.setdefault('geog_root', str(default_geog_root()))
    data.update(forcing=[str((original.parent / fetch.get('out', 'data') /
                             era5_combined_name(fetch['era5_provider'])).resolve())],
                forcing_interval_s=int(fetch.get('cadence', 6)) * 3600,
                wps_namelist=str(target.with_suffix('.namelist.wps')))
    data.setdefault('sfcp_to_sfcp', True)
    data.setdefault('output_domain', 1)
    data.setdefault('output_title', 'woof ensemble')


def review(config_path, exp, *, raw, scratch, posting):
    from woof.fetch import validate_fetch_hints
    from woof.go_cli import pinned_transport
    validate_fetch_hints(raw['fetch'], source=str(config_path))
    pinned_transport(raw['fetch'], posting.get('transport'))
    if not raw.get('case_data'):
        raise ValueError('The source preparation owner needs its member case_data table')


def prepare(plan, *, config_path, exp, observer, run_dir, prepare_only):
    from woof import runplan, era5_direct, netcdf_bridge
    from woof.bridges import find_bridge
    from woof.fetch import era5_combined_name
    from woof.toml_document import emit_experiment_toml
    raw = tomllib.loads(Path(config_path).read_text())
    hints = runplan._pinned_fetch_hints(plan, raw['fetch'])
    hints['retrieve'] = True
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path(plan.run_options['data_dir'])
    observer.enter_stage('fetch', phase='fetch')
    fetched = runplan._run_fetch(runplan._fetch_arguments_from_hints(hints, out=data_dir),
                                run_dir, events=getattr(observer, 'events', None))
    observer.finish_stage(fetch=fetched)
    forcing = data_dir / era5_combined_name(hints.get('era5_provider'))
    bridge = (netcdf_bridge.find_netcdf_bin() if forcing.suffix.lower() == '.nc'
              else find_bridge('grib1_bridge'))
    if bridge is None:
        raise ValueError('The selected ERA5 container needs its matching native decoder; install the matching bridge bundle')
    effective = deepcopy(raw)
    effective['case_data'].update(forcing=[str(forcing.resolve())],
        geog_root=plan.run_options['geog_root'],
        wps_namelist=str(Path(config_path).with_suffix('.namelist.wps')))
    target = run_dir / 'experiment.toml'
    target.write_text(emit_experiment_toml(effective))
    namelist = Path(config_path).with_suffix('.namelist.wps')
    roles = dict(grib=forcing, vtable=Path(effective['case_data']['vtable']),
                 bridge=Path(bridge), wps_namelist=namelist, experiment_config=target)
    def digest(path):
        value = hashlib.sha256()
        with path.open('rb') as stream:
            for chunk in iter(lambda: stream.read(8 * 1024**2), b''):
                value.update(chunk)
        return value.hexdigest()
    manifest = run_dir / 'source-inputs.json'
    manifest.write_text(json.dumps(dict(schema=era5_direct.INPUT_MANIFEST_SCHEMA,
        files={key: dict(name=value.name, sha256=digest(value))
               for key, value in roles.items()}), indent=2) + '\n')
    output = run_dir / 'prepared'
    arguments = ['--source', raw['fetch']['source'], '--output-root', str(output),
        '--source-sha256s', str(manifest), '--source-sha256s-sha256', digest(manifest)]
    if plan.run_options.get('experiment_config_suite'):
        arguments.append('--experiment-config-suite')
    flags = dict(grib='--grib', vtable='--vtable', bridge='--bridge',
                 wps_namelist='--wps-namelist', experiment_config='--experiment-config')
    for key, value in roles.items():
        arguments += [flags[key], str(value)]
    observer.enter_stage('prepare', phase='prepare')
    runplan._prepare_stage(output, arguments=arguments, stated={},
                          run=lambda: runplan._run_prep(arguments))
    observer.finish_stage()
    return runplan._prepared_chain_result(output, target, namelist)
