#!/usr/bin/env python3
"""Generate the complete fork initialization matrix from explicit runtime rows.

All mesh/date/geography selections are request data. Every weather product is
computed by native ungrib/init; this tool only writes workflow configuration.
"""
import argparse
import copy
from datetime import datetime, timedelta
import json
from pathlib import Path


def generate(request, work, reference):
    levels = Path(request['vertical_levels'])
    interfaces = levels.read_text().splitlines()
    if len(interfaces) < 2:
        raise ValueError('vertical interface file is empty')
    rows, decode = [], []
    for date in request['dates']:
        if len(date['hourly_gribs']) != date['forecast_hours'] + 1:
            raise ValueError('hourly LBC inputs must include both forecast endpoints')
        if any(len(parts) != 2 for parts in date['hourly_gribs']):
            raise ValueError('each valid hour needs paired pgrb2 and pgrb2b inputs')
        start = datetime.strptime(date['start'], '%Y-%m-%d_%H:%M:%S')
        for hour, gribs in enumerate(date['hourly_gribs']):
            valid = start + timedelta(hours=hour)
            decode.append(dict(name=valid.strftime('gfs-%Y%m%d-%H'), gribs=[gribs],
                               start=valid.strftime('%Y-%m-%d_%H:%M:%S'),
                               stop=valid.strftime('%Y-%m-%d_%H:%M:%S'), interval_seconds=3600))
        for mesh in request['meshes']:
            for soils in request['soil_levels']:
                name = mesh['name'] + '-soil' + str(soils)
                dated = date['start'].replace(':', '.')
                prefix = work / name / dated
                published = reference / 'cases' / mesh['name'] / dated / ('soil' + str(soils))
                variables = dict(init_case=7, start_time=date['start'], end_time=date['start'],
                                 nvertlevels=len(interfaces)-1, nsoillevels=soils,
                                 nfglevels=request['first_guess_levels'],
                                 nfgsoillevels=request['first_guess_soil_levels'], prefix=request['met_prefix'],
                                 interval_seconds=3600, ztop=interfaces[-1].strip(), zeta_levels='ZETA_LEVELS.txt',
                                 decomp_file_prefix=mesh['name']+'.graph.info.part.',
                                 lbc_hydrometeors_rrfs='true', lbc_hydrometeors_gfs='false')
                base = dict(decoded=str(work / 'decoded' / start.strftime('gfs-%Y%m%d-%H')),
                            variables=variables, ranks=mesh['ranks'], runtime_files={'ZETA_LEVELS.txt':str(levels)},
                            text_replacements=mesh.get('text_replacements', []))
                if mesh.get('precomputed'):
                    static = Path(mesh['precomputed']['static'])
                    invariant = Path(mesh['precomputed']['invariant'])
                    if soils != mesh['precomputed']['soil_levels']:
                        continue
                else:
                    for stage, inputs, outputs in (
                        ('static', {'grid.nc':mesh['grid'], **mesh.get('graph_files', {})}, ['static.nc','ugwp_oro_data.nc']),
                        ('invariant', {'static.nc':str(prefix/'static/static.nc')}, ['invariant.nc'])):
                        row = copy.deepcopy(base)
                        # The workflow static/invariant templates enable terrain
                        # blending, which requires the native first-guess file.
                        row.update(name=str(Path(name)/dated/stage), inputs=inputs,
                                   namelist_template=request['templates'][stage]['namelist'],
                                   streams_template=request['templates'][stage]['streams'],
                                   namelist_replacements={'config_geog_data_path':request['geography']},
                                   expected_outputs=outputs, publish=str(published/stage),
                                   description=f'Primary strict fork {mesh["name"]} {date["start"]} {stage}, soil levels {soils}; native workflow receipt')
                        rows.append(row)
                    static = prefix/'static/static.nc'
                    invariant = prefix/'invariant/invariant.nc'
                row = copy.deepcopy(base)
                row['runtime_files'].update(request['case7_runtime_files'])
                row.update(name=str(Path(name)/dated/'case7'), inputs={'static.nc':str(static)},
                           namelist_template=request['templates']['case7']['namelist'],
                           streams_template=request['templates']['case7']['streams'],
                           stream_replacements={'input_stream':'static.nc','output_stream':'init.nc','lbc_interval':1},
                           expected_outputs=['init.nc'], publish=str(published/'case7'),
                           invariant=str(invariant),
                           description=f'Primary strict fork {mesh["name"]} {date["start"]} case7, soil levels {soils}; workflow TEMPO aerosol inputs; native receipt')
                rows.append(row)
                if mesh['limited_area']:
                    for hour in range(len(date['hourly_gribs'])):
                        valid = start + timedelta(hours=hour)
                        row = copy.deepcopy(base)
                        # exrrfs_lbc.sh also supplies the native TEMPO monthly
                        # aerosol file. Retain the same runtime inputs as ICs.
                        row['runtime_files'].update(request['case7_runtime_files'])
                        row['variables'].update(init_case=9,start_time=valid.strftime('%Y-%m-%d_%H:%M:%S'),
                                                end_time=valid.strftime('%Y-%m-%d_%H:%M:%S'),
                                                lbc_hydrometeors_rrfs='false',lbc_hydrometeors_gfs='true')
                        # exrrfs_lbc.sh uses the precomputed vertical invariant
                        # as the immutable input, with static.nc alongside it.
                        row.update(name=str(Path(name)/dated/f'case9-{hour:03d}'),
                                   inputs={'static.nc':str(static), 'invariant.nc':str(invariant)},
                                   decoded=str(work/'decoded'/valid.strftime('gfs-%Y%m%d-%H')),
                                   namelist_template=request['templates']['case7']['namelist'],
                                   streams_template=request['templates']['case7']['streams'],
                                   stream_replacements={'input_stream':'invariant.nc','output_stream':'unused-init.nc','lbc_interval':1},
                                   expected_outputs=[valid.strftime('lbc.%Y-%m-%d_%H.%M.%S.nc')],
                                   publish=str(published/f'case9-{hour:03d}'),
                                   description=f'Primary strict fork {mesh["name"]} hourly case9 {valid:%Y-%m-%d_%H:%M:%S}, soil levels {soils}; workflow GFS hydrometeors; native receipt')
                        rows.append(row)
    return dict(schema='mpas.rrfs-native-case-matrix/v1', request=request, decode=decode, cases=rows,
                input_verification_directory=request.get('input_verification_directory'))


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('request','work','reference','out'):
        p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args()
    result=generate(json.loads(a.request.read_text()),a.work.resolve(),a.reference.resolve())
    a.out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'decode':len(result['decode']),'cases':len(result['cases'])}))
