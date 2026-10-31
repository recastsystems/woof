"""Opt-in MPAS experiment rows. Native suites and default routing stay unchanged."""
from dataclasses import replace
from ..physics_suites import load_table
from .radshared_v842 import icloud_bl_enabled


def resolve_profile(name, mode):
    if name is None:
        return None
    rows = load_table().get('experimental_profiles', {})
    if name not in rows:
        raise ValueError(f'unknown physics profile {name!r}; choices: {", ".join(rows)}')
    row = rows[name]
    if mode != row['mode']:
        raise ValueError(f'physics profile {name!r} requires mode {row["mode"]}; exact MPAS is unchanged')
    return row


def finish_profile(name, row, config, init=None):
    if config is None:
        raise ValueError('physics profile requires an MPAS route')
    from .resident_v842 import TARGET
    expected = dict(TARGET, **row['resident_selectors'])
    wrong = {k:config.selectors.get(k) for k,v in expected.items() if config.selectors.get(k) != v}
    if wrong:
        raise ValueError(f'physics profile {name!r} has unsupported selector overrides: {wrong}')
    if any(str(config.backends[k]) != 'hex-cuda' for k in config.selectors if config.active(k)):
        raise ValueError('physics profiles require the CUDA resident route')
    if init is not None:
        from netCDF4 import Dataset
        with Dataset(init) as ds:
            missing = set(row['required_init_fields']) - set(ds.variables)
            if missing:
                raise ValueError(f'physics profile {name!r} needs prepared init fields {sorted(missing)}; '
                                 'prepare aerosol climatology with initialize_species; no synthetic fallback')
    options = dict(config.options, **row['extensions'], hex_physics_profile=name)
    icloud_bl_enabled(options)
    return replace(config, options=options)
