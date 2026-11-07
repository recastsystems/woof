"""Instrument an owned WRF source copy at the ETARA argument boundary.

Writes the first LW and SW calls on the tile starting at (1,1). No profiles
are generated here. Table and argument words come from the live WRF run.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('source', type=Path, help='owned, unmodified WRF source copy to instrument')
parser.add_argument('layout', type=Path)
parser.add_argument('--after-minutes', type=int, default=0,
                    help='capture the first two calls at/after this real forecast time')
args = parser.parse_args()
if args.after_minutes < 0:
    parser.error('--after-minutes must be nonnegative')
source, layout_path = args.source, args.layout
pin = 'b6b2e9e006eff0b61f637a30e31a70f21b3c2bd8275182acb3c49aaf3fac0b4a'
if hashlib.sha256(source.read_bytes()).hexdigest() != pin:
    parser.error('module_ra_gfdleta.F does not match the public WRF v4.6.1/v4.7.1 pin; instrument an unmodified owned copy')
layout = json.loads(layout_path.read_text())
text = source.read_text().replace('ONLY : FPVS,GPVS', 'ONLY : FPVS,GPVS,TBPVS,C1XPVS,C2XPVS')
start = text.index('      SUBROUTINE ETARA(')
end = text.index('  END SUBROUTINE ETARA', start)
body = text[start:end]
declarations = '''
      INTEGER, SAVE :: gd_capture_count=0
      INTEGER :: gd_u,gd_c,gd_i,gd_j,gd_k,gd_b,gd_h(8)
      CHARACTER(80) :: gd_filename
      REAL :: gd_f(1177,8),gd_t(123192),gd_clock(4)
      INTEGER :: gd_int(27,8)
      LOGICAL :: gd_capture
'''
execution = body.index('      IF(GFDL_LW.AND.GFDL_SW )')
body = body[:execution] + declarations + body[execution:]

def pack():
    lines = ['      gd_f=0.; gd_int=0', '      DO gd_c=1,8',
             '      gd_i=its+gd_c-1; gd_j=jts']
    scalars = {'dt','g','gmt','xtime','julian','nstepra','nphs','itimestep','julyr','julday'}
    for field in layout['fields']:
        name = field['name'].lower()
        dst = f"{'gd_f' if field['kind']=='real' else 'gd_int'}({field['offset']+1},gd_c)"
        if name in {'gfdl_lw','gfdl_sw'}:
            expr = f'merge(1,0,{name})'
        elif name in {'ids','ide','jds','jde','ims','ime','jms','jme','its','ite','jts','jte'}:
            expr = '2' if name in {'ide','jde'} else '1'
        elif name in {'kms','kts','kds'}:
            expr = '1'
        elif name in {'kme','kde'}:
            expr = 'kme-kms+1'
        elif name=='kte':
            expr = 'kte-kts+1'
        elif field['count']>1:
            lines += ['      DO gd_k=kms,kme',
                      f"      gd_f({field['offset']+1}+gd_k-kms,gd_c)={name}(gd_i,gd_k,gd_j)",
                      '      ENDDO']
            continue
        elif name in scalars:
            expr = name
        else:
            expr = f'{name}(gd_i,gd_j)'
        lines.append(f'      {dst}={expr}')
    lines.append('      ENDDO')
    return '\n'.join(lines)+'\n'

tables = ['      gd_t=0.']
offset = 1
for name, (kind, shape) in layout['tables'].items():
    dims = list(map(int, shape.split(','))) if shape else [1]
    count = math.prod(dims)
    if dims==[82,82]:
        tables += [f'      DO gd_b=1,size({name},2)',
                   f'      gd_t({offset}+(gd_b-1)*82:{offset}-1+(gd_b-1)*82+size({name},1))={name}(:,gd_b)',
                   '      ENDDO']
    else:
        value = f'reshape({name},[size({name})])' if shape else f'[{name}]'
        if kind=='INTEGER': value=f'transfer({value},[0.],size({name}))'
        size = f'size({name})' if shape else '1'
        tables.append(f'      gd_t({offset}:{offset}-1+{size})={value}')
    offset += count
assert offset-1 == 123192, offset
entry = '''
      gd_capture=its==1.and.jts==1.and.ite-its>=7.and.gd_capture_count<2
      gd_capture=gd_capture.and.xtime>=CAPTURE_AFTER_MINUTES
      IF(gd_capture)THEN
      gd_capture_count=gd_capture_count+1
      write(gd_filename,'(A,I0,A)') 'gfdleta-real-',gd_capture_count,'.bin'
      open(newunit=gd_u,file=trim(gd_filename),access='stream',form='unformatted',status='replace')
      gd_h=[990461,8,kte-kts+1,1177,27,123192,kme-kms+1,gd_capture_count]
      write(gd_u) gd_h
'''.replace('CAPTURE_AFTER_MINUTES', str(args.after_minutes) + '.')+ '\n'.join(tables)+'\n'+pack()+'''
      gd_clock=[R1,RSIN1,RCOS1,RCOS2]
      write(gd_u) gd_t,gd_clock,gd_f,gd_int
      ENDIF
'''
body = body.replace('      IF(GFDL_LW.AND.GFDL_SW )', entry+'      IF(GFDL_LW.AND.GFDL_SW )', 1)
exit_block = '''
      IF(gd_capture)THEN
'''+pack()+'''
      gd_clock=[R1,RSIN1,RCOS1,RCOS2]
      write(gd_u) gd_clock,gd_f,gd_int
      close(gd_u)
      ENDIF
'''
text = text[:start] + body+exit_block + text[end:]
source.write_text(text)
