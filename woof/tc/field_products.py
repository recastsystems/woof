"""Orchestration of native TC field diagnostics; no atmospheric array maths here."""
from __future__ import annotations
import ctypes as C
import json
from pathlib import Path


def diagnose(state, grid, centre, *, swath=None, radii_centre=None):
    fields = state.physics.fields
    lat, lon = grid.latlon_mass()
    raw = dict(lat=lat, lon=lon, ht=state.ht,
               **{k: fields[k] for k in ('psfc', 't2', 'q2', 'u10', 'v10', 'xland')})
    return diagnose_arrays(raw, centre, swath=swath, radii_centre=radii_centre)


def diagnose_arrays(raw, centre, *, swath=None, radii_centre=None, unstructured=False):
    """Bind native or regular-grid surface arrays to the shared Rust reducer."""
    import numpy as np
    from woof.tc.frames import _Field
    from woof.tc.mapped_vortex import library_path
    host = {}
    for key, value in raw.items():
        if hasattr(value, '__cuda_array_interface__'):
            import cupy as cp
            value = cp.asnumpy(value)
        host[key] = np.ascontiguousarray(value, dtype='<f8')
    ny, nx = (1, host['lat'].size) if unstructured else host['lat'].shape
    bindings = (_Field * len(host))()
    names = []
    for i, (key, value) in enumerate(host.items()):
        names.append(key.encode())
        bindings[i] = _Field(names[-1], value.ctypes.data, value.size)
    request = json.dumps(dict(centre_deg=list(centre), radii_centre_deg=radii_centre,
                              nx=nx, ny=ny, swath=swath, unstructured=unstructured), allow_nan=False).encode()
    library = C.CDLL(str(library_path()))
    library.tc_frames_abi_version.restype = C.c_uint32
    if library.tc_frames_abi_version() != 4:
        raise ValueError("field-derived TC products require the matching frame I/O ABI 4 library")
    library.tc_fields_diagnose.argtypes = [C.c_char_p, C.POINTER(_Field), C.c_size_t]
    library.tc_fields_diagnose.restype = C.c_void_p
    library.tc_fields_free.argtypes = [C.c_void_p]
    library.tc_frames_last_error.restype = C.c_char_p
    result = library.tc_fields_diagnose(request, bindings, len(bindings))
    if not result:
        raise ValueError(library.tc_frames_last_error().decode())
    try:
        return json.loads(C.string_at(result))
    finally:
        library.tc_fields_free(result)


def atcf_rows(fix, *, storm_id, initial_time, valid_time, aid='ARWN'):
    """ATCF's NEQ radii rows; missing/censored quadrants stay empty."""
    lat, lon = fix['latitude_deg'], fix['longitude_deg']
    lead = (valid_time - initial_time).total_seconds() / 3600
    # ATCF's lead is integer hours: sub-hourly fixes retain full precision in JSONL.
    if abs(lead - round(lead)) > 1e-8:
        return []
    coord = lambda x, a, b: f'{int(round(abs(x)*10))}{a if x>=0 else b}'
    rows = []
    for threshold in ('34', '50', '64'):
        radii = [str(int(round(r))) if not censored else '' for r, censored in
                 zip(fix['radii_nm'][threshold], fix['radii_reach_grid_edge'][threshold])]
        rows.append(', '.join([storm_id[:2], storm_id[2:4], initial_time.strftime('%Y%m%d%H'),
                    '00', aid, str(int(round(lead))), coord(lat,'N','S'), coord(lon,'E','W'),
                    str(int(round(fix['vmax10_kt']))), str(int(round(fix['mslp_hpa']))),
                    '', threshold, 'NEQ', *radii]))
    return rows


def publish(path, fix, *, storm_id, initial_time, valid_time, aid='ARWN'):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_suffix('.fixes.jsonl').open('a', encoding='utf-8') as f:
        f.write(json.dumps(dict(fix, valid_time=valid_time.isoformat(),
                                lead_hours=(valid_time-initial_time).total_seconds()/3600),
                           allow_nan=False) + '\n')
    if storm_id:
        rows = atcf_rows(fix, storm_id=storm_id, initial_time=initial_time, valid_time=valid_time, aid=aid)
        if rows:
            with path.with_suffix('.atcf').open('a', encoding='ascii') as f:
                f.write('\n'.join(rows) + '\n')
