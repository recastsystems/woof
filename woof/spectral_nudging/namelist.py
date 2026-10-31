"""WRF FDDA controls, with Registry defaults and per-domain array semantics."""
from .config import NudgingConfig


def translate(section, grid_ids, drop):
    from woof.wrf_namelist_registry import wrf_namelist_keys
    registry = wrf_namelist_keys()
    count = len(grid_ids)

    def column(key):
        spec = registry[('fdda', key)]
        convert = int if spec['type'] == 'integer' else float
        default = convert(spec['default'])
        values = (section.registry_col(key, count, default)
                  if spec['nentries'] == 'max_domains' else
                  [section.scalar(key, default)] * count)
        for value in values:
            if isinstance(value, bool) or convert(value) != value:
                raise ValueError(f'&fdda/{key} must contain {spec["type"]} values')
        return [convert(value) for value in values]

    modes = column('grid_fdda')
    if any(value not in (0, 1, 2) for value in modes):
        raise ValueError('&fdda/grid_fdda supports 0 (off), 1 (grid), or 2 (spectral)')
    keys = [name for name in NudgingConfig.__dataclass_fields__ if ('fdda',name) in registry]
    values = {key: column(key) for key in keys}
    values['cutoff_x'] = column('xwavenum')
    values['cutoff_y'] = column('ywavenum')
    for key in ('grid_sfdda', 'obs_nudge_opt'):
        entries = section.take(key, [0])
        if any(value != 0 for value in entries):
            raise ValueError(f'FDDA_OBSERVATION_INPUT: {key} needs surface or point-observation forcing, which the gridded analysis source does not supply')
        if key in section.consumed:
            drop('fdda',key,entries,'inactive observation-nudging selector')
    for key in ('gfdda_inname','io_form_gfdda'):
        entries = section.take(key)
        if entries is not None:
            drop('fdda',key,entries,'the engine reads its own prepared analysis store')
    # Date-component alternatives change the forcing window. Do not silently
    # accept them as aliases of gfdda_end_h / gfdda_interval_m.
    for key in tuple(section.entries):
        if key.startswith(('gfdda_begin_', 'gfdda_end_', 'gfdda_interval')):
            entries = section.take(key)
            if any(value != 0 for value in entries):
                raise ValueError(f'FDDA_ALTERNATE_CLOCK: {key} would change the analysis window; use gfdda_interval_m and gfdda_end_h')
            drop('fdda',key,entries,'inactive alternate analysis clock')
    section.finish()
    domains = {}
    for index, grid_id in enumerate(grid_ids):
        if modes[index] == 0:
            continue
        row = {key: entries[index] for key,entries in values.items()}
        row.update(enabled=True, mode='grid' if modes[index] == 1 else 'spectral',
                   convention='wrf', boundary_cells=1)
        NudgingConfig(**row)
        domains[str(grid_id)] = row
    return {'domain': domains}
