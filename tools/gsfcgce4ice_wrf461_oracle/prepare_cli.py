"""Prepare a short CLI fixture using Rust NetCDF read and write.

The source inputs remain untouched. Missing package masses and their
coupled lateral values start at the zero WRF real.exe uses for unsupplied
hydrometeors. Every unchanged variable is read by rw_netcdf and written
through the Rust ClassicTape seam. This is a port smoke, not a matched
WRF forecast or a skill experiment.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request

import numpy as np

from woof.netcdf_bridge import open_dataset
from woof.io.classic_tape import ClassicTape
from woof.fortran_namelist import parse_namelist_text
from woof.ingest.wrfinput import read_wrfinput_metadata
from woof.wrfinput_door import synthesize_wps_namelist

PUBLIC_URL = 'https://raw.githubusercontent.com/wrf-model/WRF/d66e442fccc04111067e29274c9f9eaccc3cef28/test/em_real/namelist.input'
MASSES = ('QICE', 'QSNOW', 'QGRAUP', 'QHAIL')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def copy_inputs(source, target, *, boundary):
    additions = []
    with open_dataset(source) as src, ClassicTape(target, container='cdf2') as dst:
        for name, dim in src.dimensions.items():
            dst.createDimension(name, None if dim.isunlimited() else len(dim))
        dst.setncatts(dict(src.global_attributes))
        dst.setncattr('MP_PHYSICS', np.int32(7))
        declared = {}
        for name, var in src.variables.items():
            out = dst.createVariable(name, 'S1' if var.is_character else var.dtype, var.dimensions)
            out.setncatts(dict(var.attributes))
            declared[name] = (out, var)
        new_fields = {}
        for mass in MASSES:
            if boundary:
                for suffix in ('BXS', 'BXE', 'BYS', 'BYE', 'BTXS', 'BTXE', 'BTYS', 'BTYE'):
                    name = mass + '_' + suffix
                    if name in declared:
                        continue
                    template = src.variables['QVAPOR_' + suffix]
                    out = dst.createVariable(name, 'f4', template.dimensions)
                    attrs = dict(template.attributes)
                    attrs['description'] = mass + ' coupled lateral boundary value or tendency'
                    out.setncatts(attrs)
                    new_fields[name] = (out, template)
                    additions.append(name)
            elif mass not in declared:
                template = src.variables['QVAPOR']
                out = dst.createVariable(mass, 'f4', template.dimensions)
                out.setncatts({'description': mass + ' mixing ratio', 'units': 'kg kg-1',
                              'stagger': '', 'FieldType': np.int32(104), 'MemoryOrder': 'XYZ'})
                new_fields[mass] = (out, template)
                additions.append(mass)
        for out, var in declared.values():
            values = np.asarray(var[...])
            if var.dimensions and src.dimensions[var.dimensions[0]].isunlimited():
                for record in range(values.shape[0]):
                    out[record] = values[record]
            else:
                out[0] = values
        for out, template in new_fields.values():
            values = np.zeros(template.shape[1:], np.float32)
            for record in range(template.shape[0]):
                out[record] = values
    return additions


def literal(value):
    if isinstance(value, bool):
        return '.true.' if value else '.false.'
    if isinstance(value, str):
        return "'" + value.replace("'", "''") + "'"
    return str(value)


def configure_smoke(sections):
    sections.setdefault('physics', {})['mp_physics'] = [7]
    sections['physics'].pop('physics_suite', None)
    control = sections.setdefault('time_control', {})
    # WRF adds the interval components, so retire the reference cadence.
    for name in ('history_interval', 'history_interval_s', 'history_interval_m',
                 'history_interval_h', 'history_interval_d', 'history_interval_y'):
        control.pop(name, None)
    control['history_interval'] = [0]
    control['history_interval_s'] = [60]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('source', type=Path)
    ap.add_argument('reference_namelist', type=Path)
    ap.add_argument('output', type=Path)
    ap.add_argument('--receipt', type=Path, required=True)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    public = urllib.request.urlopen(PUBLIC_URL, timeout=60).read()
    (args.output / 'public-em-real.namelist.input').write_bytes(public)
    sections = parse_namelist_text(public.decode())
    reference = parse_namelist_text(args.reference_namelist.read_text())
    overrides = {}
    for section, entries in reference.items():
        target = sections.setdefault(section, {})
        for name, values in entries.items():
            if target.get(name) != values:
                overrides[f'{section}.{name}'] = {'public': target.get(name), 'smoke': values}
            target[name] = values
    configure_smoke(sections)
    original_public = parse_namelist_text(public.decode())
    for section, name in (('physics', 'mp_physics'), ('time_control', 'history_interval'),
                          ('time_control', 'history_interval_s')):
        overrides[f'{section}.{name}'] = {
            'public': original_public.get(section, {}).get(name),
            'smoke': sections[section][name]}

    text = '\n\n'.join('&' + section + '\n' + '\n'.join(
        ' ' + name + ' = ' + ', '.join(literal(v) for v in values) + ','
        for name, values in entries.items()) + '\n/' for section, entries in sections.items()) + '\n'
    (args.output / 'namelist.input').write_text(text)
    added = {}
    for name in ('wrfinput_d01', 'wrfbdy_d01'):
        added[name] = copy_inputs(args.source / name, args.output / name, boundary=name.startswith('wrfbdy'))
    metadata = read_wrfinput_metadata(args.output / 'wrfinput_d01')
    (args.output / 'namelist.wps').write_text(synthesize_wps_namelist({1: metadata}))
    receipt = {'public_namelist_url': PUBLIC_URL, 'public_sha256': hashlib.sha256(public).hexdigest(),
               'adaptations': overrides, 'selected_microphysics': 7, 'added_zero_fields': added,
               'source_sha256': {n: sha(args.source / n) for n in added},
               'produced_sha256': {n: sha(args.output / n) for n in added},
               'reference_namelist_sha256': sha(args.reference_namelist),
               'scope': 'adapted public em_real namelist and shared WRF initialization fixture; no matched forecast claim'}
    args.receipt.write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps({'selected_microphysics': 7, 'added_zero_fields': added}))


if __name__ == '__main__':
    main()
