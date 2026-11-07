#!/usr/bin/env python3
"""Add bounded real-run KEPS entry/exit capture to the pinned native source.

No arithmetic is changed. The native undefined variables remain undefined.
Set WOOF_KEPS_CAPTURE to an absolute path prefix when running native WRF.
"""
from __future__ import annotations
import hashlib
from pathlib import Path
import sys

SOURCE_SHA256 = 'a79058405affcd293740b15137002e788d168a5eba51586d14cee41342ffc204'
MASS = ('u_phy', 'v_phy', 'th_phy', 'qv_curr', 'qc_curr', 'rho', 'dz8w',
        'pi_phy', 'rthraten')
SURFACE = ('ustar', 'hfx', 'qfx', 'br', 'znt', 'psim', 'psih')
STATE = ('tke_pbl', 'diss_pbl', 'tpe_pbl', 'pr_pbl', 'tke_adv', 'diss_adv', 'tpe_adv')
OUTPUT = ('rublten', 'rvblten', 'rthblten', 'rqvblten', 'rqcblten', 'wu', 'wv',
          'wt', 'exch_h', 'exch_m')

def mass_write(name, top='kte'):
    return f'write(keps_unit) ((({name}(ix,iz,iy),ix=its,keps_ix),iy=jts,keps_iy),iz=kts,{top})\n'

def surface_write(name):
    return f'write(keps_unit) (({name}(ix,iy),ix=its,keps_ix),iy=jts,keps_iy)\n'

def instrument(source: bytes) -> bytes:
    assert hashlib.sha256(source).hexdigest() == SOURCE_SHA256, 'native KEPS pin differs'
    t = source.decode()
    declaration = '''      integer keps_unit, keps_status, keps_ix, keps_iy
      logical keps_capture
      character(len=1024) keps_prefix, keps_filename
'''
    t = t.replace('      integer xtime\n', '      integer xtime\n' + declaration, 1)
    before = '''
      keps_capture=.false.
      call get_environment_variable('WOOF_KEPS_CAPTURE',keps_prefix,status=keps_status)
      if (keps_status == 0 .and. its == ids .and. jts == jds) then
        keps_capture=(xtime <= 4 .or. &
                     (mod(xtime,60) == 0 .and. xtime <= 180))
      endif
      if (keps_capture) then
        keps_ix=min(ite,its+8)
        keps_iy=min(jte,jts+7)
        write(keps_filename,'(A,A,I6.6,A)') trim(keps_prefix),'-',xtime,'.bin'
        open(newunit=keps_unit,file=trim(keps_filename),access='stream', &
             form='unformatted',status='replace',convert='little_endian')
        write(keps_unit) kte-kts+1,keps_iy-jts+1,keps_ix-its+1,xtime
        write(keps_unit) dt,cp,g
'''
    before += ''.join(mass_write(n) for n in MASS)
    before += mass_write('p8w', 'kte+1')
    before += ''.join(surface_write(n) for n in SURFACE)
    before += ''.join(mass_write(n) for n in STATE) + surface_write('pblh')
    before += '      endif\n'
    t = t.replace('         rublten=0.', before + '         rublten=0.', 1)
    after = '      if (keps_capture) then\n'
    after += ''.join(mass_write(n) for n in OUTPUT + STATE) + surface_write('pblh')
    after += '        close(keps_unit)\n      endif\n'
    t = t.replace('      return\n      end subroutine keps',
                  after + '      return\n      end subroutine keps', 1)
    assert 'write(keps_unit)' in t
    return t.encode()

if __name__ == '__main__':
    path = Path(sys.argv[1])
    path.write_bytes(instrument(path.read_bytes()))
