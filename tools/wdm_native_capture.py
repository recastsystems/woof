"""Capture one real WDM5/WDM7 call from the pinned public WRF source.

Usage: python tools/wdm_native_capture.py WRF_CHECKOUT RECEIPT.json

Run only in a private build copy. This inserts stream writes before wdm*2D;
it changes no physics expression. Build WRF with scalar strict arithmetic,
run a real forecast with mp_physics=14 or 26 and do_radar_ref=1, then replay
wdm5-native.in or wdm7-native.in using the corresponding oracle tools.
The southwest MPI tile's middle row is captured on timestep two.
"""
from pathlib import Path
import hashlib
import json
import sys

PINS = {
    5: '0ee4071f1bc85f985130f3b4cf1f5a9acf87f23078e5e06b169d3681ff13c01c',
    7: '384aa324dc3664af1a8d0e6244f671a76ef888a0018feb50f511803b7732edbb',
}


def instrument(root: Path, receipt: Path):
    records = []
    # Validate both files before changing either one.
    sources = {}
    for scheme, expected in PINS.items():
        path = root / f'phys/module_mp_wdm{scheme}.F'
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != expected:
            raise ValueError(f'{path}: expected pristine WRF v4.6.1 source {expected}')
        sources[scheme] = raw.decode()
    for scheme, data in sources.items():
        path = root / f'phys/module_mp_wdm{scheme}.F'
        declaration = '  INTEGER ::               i,j,k'
        if data.count(declaration) != 1:
            raise ValueError(f'{path}: capture declaration anchor changed')
        data = data.replace(declaration, declaration + '\n  INTEGER :: wdm_capture_unit', 1)
        fields = ['th', 'q', 'qc', 'qr', 'qi', 'qs']
        fields += ['qg', 'qh'] if scheme == 7 else []
        fields += ['nn', 'nc', 'nr', 'den', 'pii', 'p', 'delz']
        head = 'ite-its+1, kte-kts+1, 1, ' + ('0, ' if scheme == 7 else '') + 'itimestep'
        capture = [
            '! Lane evidence only: capture one real native call, before any process.',
            '         IF (itimestep == 2 .AND. its <= ids+10 .AND. jts <= jds+10 .AND. j == jts+(jte-jts)/2) THEN',
            f"           OPEN(newunit=wdm_capture_unit,file='wdm{scheme}-native.in', &",
            "                access='stream',form='unformatted',status='replace',convert='little_endian')",
            '           WRITE(wdm_capture_unit) ' + head,
            '           WRITE(wdm_capture_unit) delt, ccn0',
        ]
        capture += [f'           WRITE(wdm_capture_unit) {name}(its:ite,kts:kte,j)' for name in fields]
        surfaces = ['xland', 'rain', 'snow', 'graupel', 'hail'] if scheme == 7 else ['rain', 'snow']
        capture += [f'           WRITE(wdm_capture_unit) {name}(its:ite,j)' for name in surfaces]
        capture += ['           CLOSE(wdm_capture_unit)', '         ENDIF']
        anchor = f'         CALL wdm{scheme}2D('
        if anchor not in data:
            anchor = f'         CALL wdm{scheme}2d('
        if data.count(anchor) != 1:
            raise ValueError(f'{path}: native module call anchor changed')
        data = data.replace(anchor, '\n'.join(capture) + '\n' + anchor, 1)
        path.write_text(data)
        records.append(dict(file=str(path), pristine_sha256=PINS[scheme],
                            instrumented_sha256=hashlib.sha256(data.encode()).hexdigest()))
    receipt.write_text(json.dumps(records, indent=2) + '\n')


if __name__ == '__main__':
    instrument(Path(sys.argv[1]), Path(sys.argv[2]))
