#!/usr/bin/env python3
"""Prepare a short private native run from the real-input release fixture."""
import argparse
import re
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('root', type=Path)
p.add_argument('--data', type=Path, required=True)
p.add_argument('--native', type=Path, required=True)
a = p.parse_args()
a.root.mkdir(parents=True, exist_ok=True)
for directory in (a.data, a.native / 'run'):
    for source in directory.iterdir():
        target = a.root / source.name
        if source.name.startswith(('rsl.', 'wrfout', 'wrfrst', 'wrfdiag', 'met_em')) or source.suffix == '.exe':
            continue
        if source.is_file() and not target.exists() and source.name != 'namelist.input':
            target.symlink_to(source.resolve())
text = (a.data / 'namelist.input').read_text()
for name, value in dict(run_hours=0, run_minutes=2, end_day=25,
                        end_hour=18, end_minute=2, bl_pbl_physics=5).items():
    text = re.sub(r'(?m)^(\s*' + name + r'\s*=)[^\n]*', r'\g<1> ' + str(value) + ',', text)
text = text.replace('&physics', '&physics\n tke_budget=1,\n bl_mynn_output=1,')
(a.root / 'namelist.input').write_text(text)
exe = a.root / 'wrf.exe'
if exe.is_symlink():
    exe.unlink()
exe.symlink_to(a.native / 'main/wrf.exe')
