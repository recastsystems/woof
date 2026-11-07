#!/usr/bin/env python3
"""Write native input-format snapshots at cold start without changing state."""
from pathlib import Path
import argparse, hashlib, json

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('wrf', type=Path)
    parser.add_argument('--receipt', type=Path, required=True)
    args = parser.parse_args()
    path = args.wrf/'share/mediation_integrate.F'
    source = path.read_text()
    needle = '\n CALL WRFU_ClockGet( grid%domain_clock, CurrTime=currTime, StartTime=startTime )\n'
    capture = (needle + '\n! GPUWM input capture: IO only, native output_input schema.\n'
               ' IF ( currTime .EQ. startTime .AND. config_flags%write_input ) THEN\n'
               '   CALL med_filter_out ( grid , config_flags )\n'
               ' ENDIF\n')
    if 'GPUWM input capture' in source:
        raise SystemExit('input capture already installed')
    assert source.count(needle) == 1
    patched = source.replace(needle, capture)
    path.write_text(patched)
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps({'path':str(path.resolve()),'source_sha256':hashlib.sha256(source.encode()).hexdigest(),
        'instrumented_sha256':hashlib.sha256(patched.encode()).hexdigest(),
        'licence':'WRF public domain statement; no file override in mediation_integrate.F',
        'change':'one native med_filter_out call at each cold-start instant; IO only; no arithmetic changed',
        'line':patched[:patched.index('! GPUWM input capture')].count('\n')+1},indent=2)+'\n')
if __name__ == '__main__':
    main()
