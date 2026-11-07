"""Run the real namelist CLI for the requested NSSL selections.

Uses the repository's valid WPS/input fixture, changing only microphysics and
its parameters. This proves import reachability, not forecast qualification.
Any refused selection makes the check fail. Artifacts stay on the CPU box.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('out_dir')
    args = parser.parse_args()
    # The box environment places this tree's tests on PYTHONPATH.
    from test_namelist_import import WPS_TEXT, INPUT_TEXT
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    wps = out / 'namelist.wps'
    wps.write_text(WPS_TEXT)
    selections = [(m.label, 18, m.selectors) for m in cases.MODES]
    selections += [(f'deprecated-{mp}', mp, {}) for mp in (17, 19, 21, 22)]
    selections += [('three-moment-selector-2', 18, {'nssl_3moment': 2})]
    report = {}
    for label, mp, selectors in selections:
        text, replacements = re.subn(
            r'(?m)^(\s*mp_physics\s*=)[^\n]+$',
            lambda match: f'{match[1]} {mp}, {mp},', INPUT_TEXT)
        if replacements != 1:
            raise ValueError('fixture must contain exactly one mp_physics assignment')
        text = text.replace('&physics\n', '&physics\n' + ''.join(
            f' {key} = {value},\n' for key, value in selectors.items()))
        inp, toml, log = (out / f'{label}.{suffix}' for suffix in ('input', 'toml', 'log'))
        inp.write_text(text)
        command = [sys.executable, '-m', 'woof.cli', 'import-namelist',
                   str(wps), str(inp), '--output', str(toml), '--name', 'nssl-variant-check']
        result = subprocess.run(command, capture_output=True, text=True)
        log.write_text(result.stdout + result.stderr)
        report[label] = {'passed': result.returncode == 0,
                         'returncode': result.returncode, 'command': command,
                         'log': str(log), 'input': str(inp),
                         'error': result.stderr[-2000:] if result.returncode else None}
        print(label, 'PASS' if result.returncode == 0 else 'REFUSED', flush=True)
        (out / 'receipt.json').write_text(json.dumps(report, indent=2) + '\n')
    return int(any(not item['passed'] for item in report.values()))


if __name__ == '__main__':
    raise SystemExit(main())
