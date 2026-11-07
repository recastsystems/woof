"""Native non-Chem EM trajectory configuration and immutable point input.

Numerics are in trajectory.cu; this module only reads configuration.
Source and licence pins: tools/trajectory_wrf461_oracle/source_manifest.json.
"""
from __future__ import annotations
import re
from datetime import datetime

COORDINATES = ('traj_i', 'traj_j', 'traj_k', 'traj_long', 'traj_lat')
CONTROLS = ('traj_opt', 'dm_has_traj', 'num_traj', 'trajectory_input')
PACKAGES = {'hyd': 'hyd_spc', 'trc': 'trc_spc', 'dyn': 'dyn_var', 'msc': 'msc_var'}
DYNAMIC = ('p', 'T', 'z', 'u', 'v', 'w', 'rainprod', 'evapprod')


def enabled(cfg):
    """The native Registry allocates coordinates on every enabled domain."""
    return cfg.traj_opt == 1


def active(cfg):
    return enabled(cfg) and cfg.dm_has_traj


def drop_inactive(values):
    if not values.get('traj_opt', 0):
        for key in CONTROLS:
            values.pop(key, None)


def validate(cfg):
    if type(cfg.traj_opt) is not int or cfg.traj_opt not in (0, 1):
        raise ValueError('traj_opt must be native EM 0 or 1')
    if type(cfg.dm_has_traj) is not bool:
        raise TypeError('dm_has_traj must be a bool')
    if type(cfg.num_traj) is not int or not 1 <= cfg.num_traj <= 1000:
        raise ValueError('num_traj must fit module_trajectory traj_max=1000')
    if active(cfg):
        points = parse_points(cfg.trajectory_input)
        if len(points) > cfg.num_traj:
            raise ValueError('trajectory point count exceeds the native num_traj allocation')


def _indexed_components(text):
    # Rewrite only assignment identifiers, never strings or comments. The
    # shared Fortran reader retains repeats, nulls and final array sections.
    token = re.compile(r"'(?:[^']|'')*'|\"(?:[^\"]|\"\")*\"|![^\n]*|traj_type\s*\(\s*(\d+)\s*\)\s*%", re.I)
    return token.sub(lambda m: f'traj_type{m[1]}%' if m[1] else m[0], text)


def parse_points(text):
    if not text.strip():
        return ()  # WRF permits a missing wrfinput_traj_dNN.
    from woof.fortran_namelist import parse_namelist_text
    groups = parse_namelist_text(_indexed_components(text), allow_unset=True)
    extra = set(groups) - {'traj_default', 'traj_spec'}
    if extra:
        raise ValueError(f'unknown trajectory input groups: {sorted(extra)}')
    defaults = groups.get('traj_default', {})
    spec = groups.get('traj_spec', {})
    default_keys = {'traj_def%' + k for k in ('start_time','stop_time','chm_name','dchm_name','hyd_name','trc_name','dyn_name','msc_name')}
    point_keys = {'lon','lat','lev','is_stationary','z_is_agl','start_time','stop_time','chm_spc','dchm_spc',*PACKAGES.values()}
    if set(defaults)-default_keys:
        raise ValueError(f'unknown trajectory defaults: {sorted(set(defaults)-default_keys)}')
    for key in spec:
        match=re.fullmatch(r'traj_type(\d+)%(.+)',key)
        if not match or not 1<=int(match[1])<=1000 or match[2] not in point_keys:
            raise ValueError(f'unknown trajectory specification: {key}')
    points = []
    def first(row, key, fallback=None):
        return next((x for x in row.get(key, ()) if x is not None), fallback)
    for n in range(1, 1001):
        prefix = f'traj_type{n}%'
        row = {key[len(prefix):]: value for key, value in spec.items() if key.startswith(prefix)}
        # Native stops at the first missing lon, lat or lev.
        if any(first(row, key, -9999.) == -9999. for key in ('lon', 'lat', 'lev')):
            break
        point = {key: float(first(row, key)) for key in ('lon', 'lat', 'lev')}
        for key in ('is_stationary', 'z_is_agl'):
            point[key] = first(row, key, key == 'z_is_agl')
            if type(point[key]) is not bool:
                raise ValueError(f'traj_type({n})%{key} must be logical')
        for key in ('start_time', 'stop_time'):
            point[key] = first(row, key, first(defaults, 'traj_def%' + key, ''))
            if point[key]:
                datetime.strptime(point[key], '%Y-%m-%d_%H:%M:%S')
        for package, field in PACKAGES.items():
            values = list(row.get(field, ()))
            if len(values)>100:raise ValueError('native trajectory variable arrays have 100 entries')
            if not values or not values[0]:
                inherited=[]
                for value in defaults.get('traj_def%' + package + '_name', ()):
                    if value is None or not str(value).strip():break
                    inherited.append(value)
                if len(inherited)>100:raise ValueError('native trajectory variable arrays have 100 entries')
                values[:len(inherited)]=inherited
            # get_var_cnt packs all nonblank entries, including entries
            # after a gap. Fortran character(len=32) truncates each name.
            names = [str(value)[:32].strip() for value in values if value is not None and str(value).strip()]
            point[package] = tuple(names)
        points.append(point)
    return tuple(points)


def validate_checkpoint(header, stored, cfg):
    """Validate the entire optional point-output namespace before restore."""
    import hashlib
    import numpy as np
    from woof.io.restart import RestartMismatchError
    keys={key for key in stored if key.startswith('trajectory/')}
    row=header.get('trajectory')
    if row is None:
        if keys:raise RestartMismatchError('trajectory prefix has no checkpoint metadata')
        status=stored.get('state/traj_status')
        if status is not None and status[0]>0:
            raise RestartMismatchError('active trajectory checkpoint is missing its point-output prefix')
        return
    if not active(cfg) or keys!={'trajectory/output_prefix'}:
        raise RestartMismatchError('trajectory checkpoint prefix inventory does not match the active option')
    prefix=stored['trajectory/output_prefix']
    if not isinstance(row,dict) or set(row)!={'sha256','records','points'}:
        raise RestartMismatchError('malformed trajectory checkpoint metadata')
    if prefix.dtype!=np.uint8 or prefix.ndim!=1 or prefix.size<32 or prefix[:4].tobytes()!=b'CDF\x02':
        raise RestartMismatchError('trajectory checkpoint prefix is not a CDF-2 byte vector')
    if hashlib.sha256(prefix).hexdigest()!=row['sha256']:
        raise RestartMismatchError('trajectory checkpoint output prefix hash differs')
    status=stored.get('state/traj_status')
    if (type(row['records']) is not int or row['records']<0
            or not isinstance(row['points'],list) or not 1<=len(row['points'])<=cfg.num_traj
            or status is None or tuple(status.shape)!=(2,)
            or int(status[0])!=len(row['points']) or int(status[1])!=row['records']):
        raise RestartMismatchError('trajectory checkpoint point roster, record clock and state disagree')
    import json
    allowed={json.dumps(point,sort_keys=True) for point in parse_points(cfg.trajectory_input)}
    if any(json.dumps(point,sort_keys=True) not in allowed for point in row['points']):
        raise RestartMismatchError('trajectory checkpoint roster differs from immutable native point input')



def scratch_shapes(cfg):
    """Conservative flat allocations, including unknown names native discards.

    Metadata uses typed views of float32 words. Two output banks overlap
    asynchronous I/O; neither may alias a sibling or another scheme.
    """
    if not active(cfg):return {}
    points=parse_points(cfg.trajectory_input);n=len(points)
    variables=len({(pkg,name) for p in points for pkg in PACKAGES for name in p[pkg]})
    sizes=dict(projection=14,input=3*n,agl=n,initial_coordinates=5*cfg.num_traj,requested=variables*n,
               intervals=6*n,moving=n,enabled=n,pointers=6*variables,
               descriptors=4*variables,
               output0=1000*(5+variables)*n,output1=1000*(5+variables)*n)
    return {'trajectory_'+key:(size,) for key,size in sizes.items()}
