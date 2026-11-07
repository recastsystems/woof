#!/usr/bin/env python3
"""Observe complete DFI save/reset/rebalance on full real native columns.

Use only a private copy of the pinned native source. The inserted calls
write arrays and never change an original statement or array value.
"""
import argparse
import hashlib
import json
from pathlib import Path

from instrument_native import PIN, FIELDS, RADII, GSFC

WATER = ['snow', 'snowh', 'snowc', 'canwat']
SOIL = ['tsk', 'tslb', 'smois']
RUC = ['qvg', 'tsnav', 'soilt1', 'smfr3d', 'keepfr3dflag']
INTERFACE = {'w_2', 'ww', 'phb', 'ph0', 'ph_2', 'dfi_w', 'dfi_ww',
             'dfi_phb', 'dfi_ph0', 'dfi_ph'}
SOIL3D = {'tslb', 'smois', 'smfr3d', 'keepfr3dflag'}
COORDS = ['c1f', 'c2f', 'c1h', 'c2h', 'c3f', 'c4f', 'c3h', 'c4h',
          'rdnw', 'rdn', 'dnw']


def writer():
    lines = ['''
SUBROUTINE lane_dfi_reset_capture(grid,mode)
 USE module_domain, ONLY: domain
 USE module_state_description
 IMPLICIT NONE
 TYPE(domain) :: grid
 INTEGER, INTENT(IN) :: mode
 INTEGER :: unit,i0,i1,j0,j1,nz,ns,ios,n,serial=0
 CHARACTER(160) :: filename
 CHARACTER(32) :: field
 IF (grid%id /= 1) RETURN
 IF (grid%sp31 > 31 .OR. grid%ep31 < 38) RETURN
 IF (grid%sp33 > 31 .OR. grid%ep33 < 38) RETURN
 i0=31; i1=38; j0=31; j1=38; nz=grid%ep32-1; ns=grid%num_soil_layers
 serial=serial+1
 WRITE(filename,'(A,I5.5,A,I2.2,A)') 'lane-reset-',serial,'-m',mode,'.bin'
 OPEN(newunit=unit,file=TRIM(filename),access='stream',form='unformatted', &
      status='replace',action='write',iostat=ios)
 IF (ios /= 0) CALL wrf_error_fatal('DFI reset capture open failed')
 WRITE(unit) 1145456946,grid%dfi_opt,grid%dfi_stage,grid%itimestep,mode, &
   grid%dfi_savehydmeteors,grid%mp_physics_dfi,num_moist,num_scalar, &
   grid%sf_surface_physics,grid%use_theta_m,grid%hypsometric_opt, &
   nz,ns,P_QV,PARAM_FIRST_SCALAR,8,8
 WRITE(unit) grid%hcoeff_tot,grid%p_top
''']

    def emit(name, extent, dims):
        lines.extend([f" field='{name}'", f" n=SIZE(grid%{name}({extent}))",
                      f" WRITE(unit) field,{dims},n", f" WRITE(unit) grid%{name}({extent})"])

    for live, saved, dims in FIELDS:
        for name in (live, saved):
            k = 'nz+1' if name in INTERFACE else 'nz'
            extent = 'i0:i1,j0:j1' if dims == 2 else f'i0:i1,1:{k},j0:j1'
            shape = '8,8,1,1' if dims == 2 else f'8,{k},8,1'
            if dims == 4:
                extent += ',:'
                shape = f'8,nz,8,SIZE(grid%{name},4)'
            emit(name, extent, shape)
    for name in ('alb', 'p_hyd', 'dfi_rh'):
        emit(name, 'i0:i1,1:nz,j0:j1', '8,nz,8,1')
    for name in ('mub', 'psfc'):
        emit(name, 'i0:i1,j0:j1', '8,8,1,1')
    for name in WATER + SOIL + RUC:
        if name == RUC[0]:
            lines.append(' IF (grid%sf_surface_physics == RUCLSMSCHEME) THEN')
        for f in (name, 'dfi_' + name):
            if name in SOIL3D:
                emit(f, 'i0:i1,1:ns,j0:j1', '8,ns,8,1')
            else:
                emit(f, 'i0:i1,j0:j1', '8,8,1,1')
    lines.append(' ENDIF')
    lines.append(' IF (grid%mp_physics_dfi /= NUWRF4ICESCHEME_DFI) THEN')
    for group in (RADII, GSFC):
        if group is GSFC:
            lines.append(' ELSE')
        for name in group:
            for f in (name, 'dfi_' + name):
                emit(f, 'i0:i1,1:nz,j0:j1', '8,nz,8,1')
    lines.append(' ENDIF')
    for name in COORDS:
        emit(name, '1:nz+1', 'nz+1,1,1,1')
    lines.extend([' CLOSE(unit)', 'END SUBROUTINE lane_dfi_reset_capture\n'])
    return '\n'.join(lines)


def insert_calls(text):
    original = text
    anchors = [
        ('\n   END SUBROUTINE dfi_save_arrays\n', 4, False),
        ('      grid%MU_2(:,:)    = grid%dfi_mu(:,:)      / grid%hcoeff_tot', 5, False),
        ('\n   END SUBROUTINE dfi_array_reset\n', 6, False),
        ('            grid%ph_2(i,1,j) = 0.', None, False),
    ]
    # The rebalance capture must precede its FIRST mutation, outside loops.
    anchors[-1] = ('            ktf=MIN(kte,kde-1)', 7, True)
    anchors.append(('\n   END SUBROUTINE rebalance_dfi\n', 8, False))
    anchors += [
        ('      grid%dfi_mu(:,:)       = 0.', 9, False),
        ('\n   END SUBROUTINE dfi_clear_accumulation\n', 10, False),
    ]
    for anchor, mode, after in anchors:
        assert text.count(anchor) == 1, anchor
        call = f'      CALL lane_dfi_reset_capture(grid,{mode})'
        replacement = (anchor + '\n' + call if after else
                       '\n' + call + anchor if anchor.startswith('\n') else
                       call + '\n' + anchor)
        text = text.replace(anchor, replacement)
    restored = text
    for mode in range(4, 11):
        restored = restored.replace(f'      CALL lane_dfi_reset_capture(grid,{mode})\n', '')
    assert restored == original
    return text


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--build', type=Path, required=True)
    p.add_argument('--receipt', type=Path, required=True)
    args = p.parse_args()
    path = args.build / 'share/dfi.F'
    original = path.read_bytes()
    if hashlib.sha256(original).hexdigest() != PIN:
        raise SystemExit('private original dfi.F does not match the pin')
    text = insert_calls(original.decode()) + writer()
    path.write_text(text)
    args.receipt.write_text(json.dumps({
        'path': str(path), 'original_sha256': PIN,
        'instrumented_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
        'native_arithmetic_changed': False, 'tile': '8 x 8 full columns',
        'modes': {'4': 'after save', '5': 'before reset', '6': 'after complete reset',
                  '7': 'before rebalance', '8': 'after rebalance',
                  '9': 'before clear', '10': 'after clear'},
    }, indent=2) + '\n')


if __name__ == '__main__':
    main()
