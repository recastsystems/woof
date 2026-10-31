"""Compile verbatim pinned nearest_j routines with their native coordinates.

Framework-only harness; no geography or weather-field processing in Python.
Compiler execution remains inside build-slot (four build jobs).
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--source', type=Path, required=True)
p.add_argument('--scratch', type=Path, required=True)
p.add_argument('--receipt', type=Path, required=True)
a = p.parse_args()
a.scratch.mkdir(parents=True, exist_ok=True)
rows = []
for rk in (4, 8):
    for small, ny in ((True, 21600), (False, 4320)):
        name = 'sm' if small else 'lg'
        source = a.source / f'mpas_gsl_oro_data_{name}_scale.F'
        text = source.read_text()
        routines = '\n'.join(re.search(rf'^function {fn}\(lat_in\).*?^end function {fn}', text, re.M | re.S).group() for fn in ('nearest_j_north', 'nearest_j_south'))
        harness = f'''module witness
implicit none
integer, parameter :: RKIND={rk}, topo_y={ny}
real(RKIND), parameter :: pi=2.0_RKIND*asin(1.0_RKIND),p5=0.5_RKIND
real(RKIND), allocatable :: lat1d_fine(:)
contains
{routines}
end module
program replay
use witness
implicit none
integer :: j,k
real(RKIND) :: queries(8)
allocate(lat1d_fine(topo_y))
do j=1,topo_y
 lat1d_fine(j)=(-90._RKIND+(180._RKIND/topo_y)*(j-p5))*pi/180._RKIND
end do
queries=(/lat1d_fine(1),-p5*pi,0._RKIND,lat1d_fine(topo_y), &
 (lat1d_fine(topo_y)+p5*pi)*p5,p5*pi,-pi,pi/)
do k=1,size(queries)
 print *,k,nearest_j_north(queries(k)),nearest_j_south(queries(k))
end do
end program
'''
        path = a.scratch / f'{name}-{rk}.F90'
        path.write_text(harness)
        exe = a.scratch / f'{name}-{rk}'
        flags = ['-O3','-ffp-contract=off','-fno-tree-vectorize','-fno-unsafe-math-optimizations','-ffree-form','-ffree-line-length-none','-fconvert=big-endian']
        cmd = ['build-slot','make','-j4','-f',str(a.scratch/'Makefile'),str(exe)]
        (a.scratch/'Makefile').write_text(f'{exe}: {path}\n\tgfortran '+ ' '.join(flags)+f' {path} -o {exe}\n')
        subprocess.run(cmd,cwd=a.scratch,check=True)
        result = subprocess.check_output([str(exe)],text=True)
        values = [[int(x) for x in line.split()] for line in result.splitlines()]
        expected_last = ny if small else ny-1
        assert all(row[1:] == [ny,expected_last] for row in values[3:6]),values
        rows.append({'scale':name,'rkind_bytes':rk,'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'verbatim_source_lines':'673-726' if small else '959-1012','queries':['first row','south pole','equator','last row','between last row and north pole','north pole','below south pole','above north pole'],'native_indices':values,'compile_flags':flags})
a.receipt.write_text(json.dumps({'schema':'gsl-native-latitude-witness/v1','runs':rows,'terminal_indices_match':True,'excluded_words':[]},indent=2)+'\n')
