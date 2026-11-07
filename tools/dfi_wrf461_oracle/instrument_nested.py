#!/usr/bin/env python3
"""Observe every native nested DFI sample, including repeated endpoints.

Only filenames and observations are added to the pinned public WRF source.
A sequence number prevents nested integrate calls overwriting the same clock.
"""
import argparse
import hashlib
import json
from pathlib import Path
from instrument_native import PIN, capture_source


def instrument(original):
    if hashlib.sha256(original).hexdigest() != PIN:
        raise ValueError('private original dfi.F does not match the pin')
    text = original.decode()
    anchors = [
        ('      hn = grid%hcoeff(grid%itimestep+1)', 0, False),
        ('      grid%hcoeff_tot = grid%hcoeff_tot + hn', 1, True),
        ('      grid%MU_2(:,:)    = grid%dfi_mu(:,:)      / grid%hcoeff_tot', 2, False),
        ('     if ( grid%sf_surface_physics .EQ. RUCLSMSCHEME ) then\n!      grid%qvg(:,:)', 3, False),
    ]
    for anchor, mode, after in anchors:
        assert text.count(anchor) == 1
        call = f'      CALL lane_dfi_capture(grid,{mode})'
        text = text.replace(anchor, anchor+'\n'+call if after else call+'\n'+anchor)
    restored = text
    for mode in range(4):
        restored = restored.replace(f'      CALL lane_dfi_capture(grid,{mode})\n', '')
    assert restored.encode() == original
    writer = capture_source().replace(' INTEGER :: unit,i0,i1,j0,j1,k0,k1,n,ios',
        ' INTEGER :: unit,i0,i1,j0,j1,k0,k1,n,ios,serial=0')
    writer = writer.replace(' IF (grid%id /= 1) RETURN\n', '')
    begin = writer.index(' WRITE(filename,')
    end = writer.index(' OPEN(newunit=', begin)
    writer = writer[:begin]+" serial=serial+1\n WRITE(filename,'(A,I3.3,A,I7.7,A,I1,A)') &\n  'lane-dfi-d',grid%id,'-c',serial,'-m',mode,'.bin'\n"+writer[end:]
    return text+writer


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--build',type=Path,required=True)
    p.add_argument('--receipt',type=Path,required=True)
    a=p.parse_args()
    path=a.build/'share/dfi.F'
    original=path.read_bytes()
    text=instrument(original)
    path.write_text(text)
    a.receipt.write_text(json.dumps(dict(original_sha256=PIN,
        instrumented_sha256=hashlib.sha256(text.encode()).hexdigest(),
        native_arithmetic_changed=False, tile='i/j 31..38, k 2..5',
        scope='all domains; sequence-numbered observations, no endpoint deduplication'),indent=2)+'\n')


if __name__=='__main__':main()
