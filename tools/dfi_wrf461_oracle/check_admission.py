"""Verify stock DFI controls reach every domain in both nest feedback modes.

Use the existing stock two-domain namelist fixture without importing its
unrelated forecast test modules or loading radiation tables. No model mocks.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from woof.namelist_import import import_namelists
from woof.experiment import load_experiment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    fixture = ROOT / 'tests/test_namelist_import.py'
    constants = {}
    for item in ast.parse(fixture.read_text()).body:
        if isinstance(item, ast.Assign) and len(item.targets) == 1:
            target = item.targets[0]
            if isinstance(target, ast.Name) and target.id in ('INPUT_TEXT', 'WPS_TEXT'):
                constants[target.id] = ast.literal_eval(item.value)
    wps = out / 'namelist.wps'
    inp = out / 'namelist.input'
    wps.write_text(constants['WPS_TEXT'])
    receipt = {'scope': 'stock namelist admission only, no forecast',
               'fixture_sha256': hashlib.sha256(fixture.read_bytes()).hexdigest(),
               'cases': []}
    for feedback in (0, 1):
        base = constants['INPUT_TEXT'].replace(' feedback = 0,', f' feedback = {feedback},')
        for opt in (1, 2, 3):
            inp.write_text(base + f"""&dfi_control
 dfi_opt = {opt}, dfi_nfilter = 7,
 dfi_bckstop_year = 1999, dfi_bckstop_month = 5, dfi_bckstop_day = 3,
 dfi_bckstop_hour = 11, dfi_bckstop_minute = 58,
 dfi_fwdstop_year = 1999, dfi_fwdstop_month = 5, dfi_fwdstop_day = 3,
 dfi_fwdstop_hour = 12, dfi_fwdstop_minute = 2,
/""")
            text, report = import_namelists(wps, inp)
            resolved = out/f'feedback-{feedback}-opt-{opt}.toml'
            resolved.write_text(text)
            experiment = load_experiment(resolved)
            assert all(d.run.dfi_opt == opt and d.run.dfi_nfilter == 7
                       for d in experiment.domains)
            assert experiment.feedback == feedback
            receipt['cases'].append({'feedback':feedback,'dfi_opt':opt,
                'admitted':True,'domains':len(experiment.domains)})
    (out / 'admission-receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
