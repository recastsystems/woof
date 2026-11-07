"""Capture both native fractional LSM transformations without changing math."""
from pathlib import Path
import json
import re
import sys
root, dest = map(Path, sys.argv[1:])
p = root/'phys/module_surface_driver.F'
source = p.read_text()
names = ('xice albedo emiss tsk tsk_sea tsk_save sst flhc flqc cpm cqs2 '
         'chs2 chs qsfc qgh qz0 hfx qfx lh').split()
names += [n+'_sea' for n in
          'flhc flqc cpm cqs2 chs2 chs qsfc qgh qz0 hfx qfx lh'.split()]

def dump(land, mode, arm):
    lines = ['         IF (ITIMESTEP.LE.3) THEN',
             '          BLOCK', '           INTEGER :: capture_unit',
             '           CHARACTER(len=256) :: capture_file',
             '           WRITE(capture_file,\'(A,I0,A,I0,A,I0,A,I0,A,I0,A)\') &',
             f'             "captures/weighting-",{land},"-",{mode},"-",ITIMESTEP,"-",i_start(ij),"-",j_start(ij),"-{arm}.bin"',
             '           OPEN(newunit=capture_unit,file=trim(capture_file),access="stream",form="unformatted",status="replace",convert="little_endian")',
             f'           WRITE(capture_unit) i_end(ij)-i_start(ij)+1,j_end(ij)-j_start(ij)+1,ITIMESTEP,{land},{mode},MERGE(1,0,isisfc)',
             '           WRITE(capture_unit) XICE_THRESHOLD']
    for name in names:
        lines.append('           WRITE(capture_unit) '+name.upper()+
                     '(i_start(ij):i_end(ij),j_start(ij):j_end(ij))')
    lines += ['           CLOSE(capture_unit)', '          END BLOCK', '         ENDIF']
    return '\n'.join(lines)+'\n'

for scheme, land in [('LSMSCHEME',2), ('NOAHMPSCHEME',4)]:
    start = source.index('     CASE ('+scheme+')')
    end = source.index('     CASE (',start+10)
    body = source[start:end]
    matches = list(re.finditer(r'(?im)^\s*IF\s*\(\s*FRACTIONAL_SEAICE\s*==\s*1\s*\)\s*THEN',body))
    assert len(matches)==2,(scheme,len(matches))
    for mode, match in reversed(list(enumerate(matches))):
        level = 0
        finish = None
        for line in re.finditer(r'(?im)^.*$',body[match.start():]):
            text=line.group().strip()
            if re.match(r'(?i)^IF\b.*\bTHEN\b',text): level += 1
            elif re.match(r'(?i)^END\s*IF\b',text):
                level -= 1
                if level == 0:
                    finish=match.start()+line.end()
                    break
        assert finish is not None,scheme
        body = body[:finish]+'\n'+dump(land,mode,'out')+body[finish:]
        body = body[:match.start()]+dump(land,mode,'in')+body[match.start():]
    source=source[:start]+body+source[end:]
p.write_text(source)
dest.mkdir(parents=True,exist_ok=True)
(dest/'weighting-layout.json').write_text(json.dumps({'f2':names},indent=2)+'\n')
