"""WRF 4.7.1 level-output controls; CPU-only admission and import facts."""
from __future__ import annotations

import math

DEFAULTS = {
    'p_lev_diags': 0, 'z_lev_diags': 0,
    'p_lev_diags_dfi': 0, 'z_lev_diags_dfi': 0,
    'press_levels': (), 'z_levels': (),
    'use_tot_or_hyd_p': 2, 'extrap_below_grnd': 1,
    'p_lev_missing': -999., 'z_lev_missing': -999.,
    'p_lev_interval_s': 0., 'z_lev_interval_s': 0.,
}
for _stream in (22, 23):
    DEFAULTS.update({f'auxhist{_stream}_interval_s': 0.,
                     f'auxhist{_stream}_begin_s': 0.,
                     f'auxhist{_stream}_end_s': 0.,
                     f'auxhist{_stream}_outname': f'auxhist{_stream}_d<domain>_<date>',
                     f'frames_per_auxhist{_stream}': 1})


def active(cfg):
    return bool(cfg.p_lev_diags or cfg.z_lev_diags)


def drop_inert(values):
    """Default-off documents/checkpoints retain their earlier bytes."""
    if not values.get('p_lev_diags', 0) and not values.get('z_lev_diags', 0):
        for name in DEFAULTS:
            values.pop(name, None)


def validate(cfg):
    for prefix, stream, targets in (('p', 23, cfg.press_levels), ('z', 22, cfg.z_levels)):
        switch = getattr(cfg, prefix+'_lev_diags')
        if type(switch) is not int or switch not in (0, 1):
            raise ValueError(prefix+'_lev_diags must be 0 or 1')
        dfi = getattr(cfg, prefix+'_lev_diags_dfi')
        if type(dfi) is not int or dfi not in (0, 1):
            raise ValueError(prefix+'_lev_diags_dfi must be 0 or 1')
        if not switch:
            continue
        if not 1 <= len(targets) <= 100 or any(
                isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x)
                for x in targets):
            raise ValueError(prefix+' level diagnostics require 1..100 finite target levels')
        if prefix == 'p' and any(x <= 0 for x in targets):
            raise ValueError('press_levels must be positive for native logarithms')
        stem = f'auxhist{stream}'
        interval = getattr(cfg, stem+'_interval_s')
        if not math.isfinite(interval) or interval <= 0 or interval != int(interval):
            raise ValueError(stem+' requires a positive whole-second interval, as native NINT/MOD')
        calculation = getattr(cfg, prefix+'_lev_interval_s') or interval
        if not math.isfinite(calculation) or calculation < .5:
            raise ValueError(prefix+'_lev_interval_s must have positive native NINT')
        begin, end = getattr(cfg, stem+'_begin_s'), getattr(cfg, stem+'_end_s')
        if any(not math.isfinite(x) or x < 0 or int(x) != x for x in (begin, end)):
            raise ValueError(stem+' begin/end must be nonnegative whole seconds')
        if end and end < begin:
            raise ValueError(stem+' end must follow begin')
        frames = getattr(cfg, f'frames_per_auxhist{stream}')
        if type(frames) is not int or frames < 1:
            raise ValueError(f'frames_per_auxhist{stream} must be positive')
        name = getattr(cfg, stem+'_outname')
        if not isinstance(name, str) or not name or '\x00' in name:
            raise ValueError(stem+'_outname must be a nonempty filename template')
    if active(cfg) and (cfg.use_tot_or_hyd_p not in (1, 2) or cfg.extrap_below_grnd not in (1, 2)):
        raise ValueError('use_tot_or_hyd_p and extrap_below_grnd must each be 1 or 2')
    if not all(math.isfinite(x) for x in (cfg.p_lev_missing, cfg.z_lev_missing)):
        raise ValueError('level missing sentinels must be finite')


def import_controls(diags, tc, max_dom, error):
    """Consume the native groups and return one config mapping per domain."""
    rows = [dict() for _ in range(max_dom)]
    for name, default in DEFAULTS.items():
        if name.startswith(('auxhist', 'frames_per_')) or name in ('press_levels', 'z_levels', 'p_lev_interval_s', 'z_lev_interval_s'):
            continue
        value = diags.scalar(name, default)
        for row in rows:
            row[name] = value
    for prefix, stream, name in (('p', 23, 'press_levels'), ('z', 22, 'z_levels')):
        count = diags.scalar('num_'+name, 0)
        values = diags.take(name, [])
        if type(count) is not int or count < 0 or count > 100 or len(values) < count:
            raise error('diags', 'num_'+name, [count], 'requires 0..100 supplied native target levels')
        for row in rows:
            row[name] = tuple(values[:count])
        stem = f'auxhist{stream}'
        columns = {}
        for part in ('interval', 'begin', 'end'):
            bare = tc.registry_col(stem+'_'+part, max_dom, 0)
            units = {u: tc.registry_col(stem+'_'+part+'_'+u, max_dom, 0)
                     for u in ('d', 'h', 'm', 's', 'y')}
            # gen_streams.c uses a minutes fallback for all three. Its
            # TimeIntervalSet calls omit the fetched year components.
            for raw in (bare, *units.values()):
                if any(type(x) is not int or x < 0 for x in raw):
                    raise error('time_control', stem+'_'+part, raw, 'requires nonnegative integer time components')
            columns[stem+'_'+part+'_s'] = [
                units['d'][n]*86400+units['h'][n]*3600+(units['m'][n] or bare[n])*60+units['s'][n]
                for n in range(max_dom)]
            if part == 'interval':
                # check_a_mundo adds BOTH bare minutes and _m. Keep its
                # calculation cadence distinct from the stream alarm.
                columns[prefix+'_lev_interval_s'] = [
                    units['d'][n]*86400+units['h'][n]*3600+(units['m'][n]+bare[n])*60+units['s'][n]
                    for n in range(max_dom)]
        io = tc.scalar(f'io_form_auxhist{stream}', 0)
        columns[stem+'_outname'] = [tc.scalar(stem+'_outname', DEFAULTS[stem+'_outname'])] * max_dom
        columns[f'frames_per_auxhist{stream}'] = tc.registry_col(f'frames_per_auxhist{stream}', max_dom, 1)
        if rows[0][prefix+'_lev_diags'] and io != 2:
            raise error('time_control', f'io_form_auxhist{stream}', [io], 'native level output requires NetCDF IO form 2')
        for key, values in columns.items():
            for n, row in enumerate(rows):
                row[key] = values[n]
    diags.finish()
    return rows


def registry_rows(registry):
    for name, default in DEFAULTS.items():
        registry.setdefault('parameters', {})[name] = {
            'default': list(default) if isinstance(default, tuple) else default,
            'type': ('array' if isinstance(default, tuple) else 'string' if isinstance(default, str)
                     else 'integer' if type(default) is int else 'real'),
            'per_domain': name.startswith(('auxhist', 'frames_per_')) or name in ('p_lev_interval_s', 'z_lev_interval_s'),
            'consuming_read': 'woof/core/level_output.py'}
    for switch, stream, suffix in (('p_lev_diags', 23, 'PL'), ('z_lev_diags', 22, 'ZL')):
        registry.setdefault('diagnostic_packages', {})[switch] = {
            '0': {'label': 'Off', 'fields': []},
            '1': {'label': 'WRF 4.7.1 native level output', 'stream': stream,
                  'source_pin': 'f52c197ed39d12e087d02c50f412d90d418f6186',
                  'consumer': 'woof/core/level_output.py',
                  'fields': [('P_PL' if suffix == 'PL' else 'Z_ZL'), *[f'{x}_{suffix}' for x in ('U','V','T','RH','GHT','S','TD','Q',
                            *(() if suffix == 'PL' else ('P',)))]]}}
