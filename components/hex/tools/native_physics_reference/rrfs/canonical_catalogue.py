#!/usr/bin/env python3
"""Join sealed canonical case evidence to explicitly selected forecast companions.

Recorded native byte/finiteness verification stays labelled as recorded. This
catalogue authenticates metadata and selected companion bytes, without rereading
all weather outputs, decoding fields or claiming new execution/identity.
"""
import argparse
from datetime import datetime, timedelta
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from publish_cache import sha
from ruled_reference import validate_init_case, validate_init_configuration
from verify_diagnostic_contract import publication


def require(value, reason):
    if not value:
        raise ValueError(reason)


def selected_file(row, cached):
    path = Path(row['path'])
    key = json.dumps(row, sort_keys=True)
    if key not in cached:
        if 'publication' in row:
            pinned = row['publication']
            parent = Path(pinned['path'])
            require(sha(parent/'PUBLICATION.json') == pinned['publication_sha256'],
                    'companion publication differs')
            pub = json.loads((parent/'PUBLICATION.json').read_text())
            member = str(path.relative_to(parent))
            require('..' not in Path(member).parts, 'companion escapes publication')
            require(pub['files'].get(member) == row['sha256'],
                    'companion is not a published member')
        require(sha(path) == row['sha256'], 'selected companion bytes differ')
        cached.add(key)
    return row


def assemble(request):
    records, products, cached = {}, [], set()
    authority = None
    total_words = total_files = 0
    for pinned in request['input_products']:
        parent = Path(pinned['path'])
        publication(parent/'PUBLICATION.json', pinned['publication_sha256'])
        index = json.loads((parent/'INDEX.json').read_text())
        witness = json.loads((parent/'VERIFIED.json').read_text())
        require(index['complete'] and index['nonfinite_words'] == 0
                and witness['verified'] and not witness['forecast_identity'],
                'input product lacks recorded complete native evidence')
        seen = {v['path']:v for v in witness['cases']}
        require(len(seen) == len(witness['cases']) == len(index['products']) == index['stages'],
                'native case evidence membership differs')
        words = files = 0
        for row in index['products']:
            directory = Path(row['path'])
            require(row['path'] not in records, 'duplicate canonical case')
            require(sha(directory/'PUBLICATION.json') == row['publication_sha256'],
                    'native case publication differs')
            pub = json.loads((directory/'PUBLICATION.json').read_text())
            require(sha(directory/'MANIFEST.json') == pub['files'].get('MANIFEST.json'),
                    'native case manifest differs')
            native = json.loads((directory/'MANIFEST.json').read_text())
            current = validate_init_case(native)
            require(current and current == index['source_correction']
                    and native.get('reference_label') == current['label'],
                    'canonical initializer authority differs')
            for name,digest in native['configuration'].items():
                require(pub['files'].get(name) == digest and sha(directory/name) == digest,
                        'canonical configuration differs')
            validate_init_configuration((directory/'namelist.init_atmosphere').read_text())
            authority = current if authority is None else authority
            require(current == authority, 'mixed canonical initializer authorities')
            checked = seen.get(row['path'], {})
            require(checked.get('publication_sha256') == row['publication_sha256']
                    and checked.get('outputs') == native['outputs'] == row['outputs']
                    and checked.get('authority') == current,
                    'recorded native output verification differs')
            require(native['completed'] and native['exit_status'] == 0
                    and row['nonfinite_words'] == 0 and row['floating_words'] > 0,
                    'canonical stage incomplete or nonfinite')
            require(all(pub['files'].get(n) == d and (directory/n).is_file()
                        for n,d in native['outputs'].items()), 'native output member absent')
            records[row['path']] = dict(row=row, native=native)
            words += row['floating_words']; files += len(row['outputs'])
        require(words == index['floating_words'] and files == index['netcdf_files'],
                'recorded native aggregate differs')
        total_words += words; total_files += files
        products.append(dict(**pinned, stages=index['stages'], netcdf_files=files,
                             recorded_finite_floating_words=words))
    groups, used, ids = [], set(), set()
    for requested in request['case_groups']:
        require(requested['id'] not in ids, 'duplicate canonical group id')
        ids.add(requested['id'])
        paths = [requested['case7'], *requested['case9']]
        require(len(set(paths)) == len(paths) and not used.intersection(paths),
                'canonical stage reused across groups')
        require(all(p in records for p in paths), 'case group selects unpublished canonical stage')
        used.update(paths)
        initial = records[requested['case7']]['native']
        variables = initial['row']['variables']
        require(variables['init_case'] == 7, 'group initializer is not case7')
        require(variables['decomp_file_prefix'] == requested['decomp_prefix'],
                'group decomposition differs')
        companions = {n:selected_file(v,cached) for n,v in requested['companions'].items()}
        require({'grid','static.nc','invariant.nc','ugwp_oro_data.nc'} <= companions.keys(),
                'forecast companion inventory incomplete')
        require(initial['inputs']['static.nc'] == companions['static.nc']['sha256'],
                'canonical static dependency differs')
        start = datetime.strptime(variables['start_time'], '%Y-%m-%d_%H:%M:%S')
        times = []
        for path in requested['case9']:
            native = records[path]['native']; values = native['row']['variables']
            require(values['init_case'] == 9
                    and all(values[k] == variables[k] for k in ('nvertlevels','nsoillevels')),
                    'boundary levels or native case differ')
            require(all(native['inputs'].get(n) == companions[n]['sha256']
                        for n in ('static.nc','invariant.nc')),
                    'boundary companion dependency differs')
            times.append(datetime.strptime(values['start_time'], '%Y-%m-%d_%H:%M:%S'))
        expected = [start+timedelta(seconds=s) for s in requested['boundary_lead_seconds']]
        require(times == expected and len(set(times)) == len(times),
                'boundary endpoint clock or membership differs')
        if times:
            require(len(times) > 1 and times[0] == start
                    and all(b-a == timedelta(seconds=requested['boundary_interval_seconds'])
                            for a,b in zip(times,times[1:])), 'boundary interval is not continuous')
        groups.append(dict(id=requested['id'], mesh=requested['mesh'], start=variables['start_time'],
            soil_levels=variables['nsoillevels'], model_levels=variables['nvertlevels'],
            interface_levels=variables['nvertlevels']+1, companions=companions,
            case7=records[requested['case7']]['row'],
            case9=[records[p]['row'] for p in requested['case9']],
            boundary_lead_seconds=requested['boundary_lead_seconds']))
    require(records and used == records.keys(), 'canonical catalogue does not cover every stage')
    tables = {n:selected_file(v,cached) for n,v in request['forecast_tables'].items()}
    require(tables, 'explicit forecast table inventory required')
    deck_indexes = [selected_file(row,cached) for row in request['deck_indexes']]
    return dict(schema='mpas.rrfs-canonical-case-catalogue/v1', source_correction=authority,
        input_products=products, case_groups=groups, forecast_tables=tables, deck_indexes=deck_indexes,
        groups=len(groups), stages=len(records), netcdf_files=total_files,
        recorded_finite_floating_words=total_words, recorded_nonfinite_words=0,
        complete=True, native_weather_outputs_rehashed=False, selected_companions_rehashed=True,
        weather_execution=False, module_replay=False, forecast_identity=False,
        scope='Complete canonical input membership and exact selected companions; native finiteness/byte verification is recorded evidence from sealed source products, not new field execution')


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--request',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    require(not a.out.exists(), 'refusing existing catalogue')
    result=assemble(json.loads(a.request.read_text()))
    a.out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(out=str(a.out),sha256=sha(a.out),groups=result['groups'],stages=result['stages'],netcdf_files=result['netcdf_files'])))
