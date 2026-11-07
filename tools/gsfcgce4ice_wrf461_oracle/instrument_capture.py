"""Instrument a COPY of the pinned WRF module at its real entry/exit boundary.

Use one native tile per MPI rank and nproc_x=1 so the selected j index belongs
only to one rank. Capture files use layout.py and are directly replayable by
run_gpu.py. No prognostic values or physics expressions are changed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

if __package__:
    from .layout import LEVEL_FIELDS, PREC_FIELDS, SURFACE_FIELDS
else:
    from layout import LEVEL_FIELDS, PREC_FIELDS, SURFACE_FIELDS

SOURCE_SHA256 = '79b782f6832ce45c0a13f463841d1820cabc4de0049ac60b5c2a23d24389cb8b'


def instrument(source: bytes, *, j_index: int, timesteps: list[int]) -> bytes:
    if hashlib.sha256(source).hexdigest() != SOURCE_SHA256:
        raise ValueError('capture requires the pinned public WRF 4.6.1 module')
    if j_index < 1 or not timesteps or any(t < 1 for t in timesteps):
        raise ValueError('j index and all capture timesteps must be positive')
    if sorted(set(timesteps)) != timesteps:
        raise ValueError('capture timesteps must be unique and increasing')
    s = source.decode()
    declarations = '''
  integer :: cap_u, cap_i, cap_j, cap_k, cap_nx, cap_ny, cap_radar, cap_diag
  integer, save :: cap_last = -1
  logical :: cap_take
  character(len=128) :: cap_name
'''
    declaration_site = '      INTEGER:: NCALL=0'
    entry_site = '   itaobraun = 0'
    exit_site = '  END SUBROUTINE gsfcgce_4ice_nuwrf'
    for site in (declaration_site, entry_site, exit_site):
        if s.count(site) != 1:
            raise ValueError(f'capture anchor is not unique: {site!r}')
    s = s.replace(declaration_site, declaration_site + '\n' + declarations, 1)
    fields = [{'qc': 'ql'}.get(n, n) for n in LEVEL_FIELDS]
    steps = ','.join(map(str, timesteps))
    pre = f'''
  cap_take = jts <= {j_index} .and. jte >= {j_index} .and. &
    any(itimestep == [{steps}]) .and. cap_last /= itimestep
  if (cap_take) then
    cap_last = itimestep
    cap_nx = ite-its+1
    cap_ny = jte-jts+1
    cap_radar=0
    if(present(do_radar_ref)) cap_radar=do_radar_ref
    cap_diag=0
    if(present(diagflag)) then
      if(diagflag) cap_diag=1
    endif
    write(cap_name,'("case-real-",i6.6,".bin")') itimestep
    open(newunit=cap_u,file=trim(cap_name),access='stream',form='unformatted', &
      status='replace',convert='little_endian')
    write(cap_u) cap_nx*cap_ny, kte-kts+1, 1, cap_radar
    write(cap_u) dt_in, dx
    write(cap_u) itimestep, cap_diag
    write(cap_u) ht(its:ite,jts:jte), xland(its:ite,jts:jte)
'''
    for name in fields:
        pre += f'    write(cap_u) (({name}(its:ite,cap_k,cap_j),cap_j=jts,jte),cap_k=kts,kte)\n'
    pre += '    write(cap_u) ((w(its:ite,cap_k,cap_j),cap_j=jts,jte),cap_k=kts,kte+1)\n'
    for name in SURFACE_FIELDS:
        pre += f'    write(cap_u) {name}(its:ite,jts:jte)\n'
    pre += '    close(cap_u)\n  endif\n'
    s = s.replace(entry_site, pre + '\n' + entry_site, 1)
    post = '''
  if(cap_take) then
    write(cap_name,'("case-real-",i6.6,".expected.out")') itimestep
    open(newunit=cap_u,file=trim(cap_name),access='stream',form='unformatted', &
      status='replace',convert='little_endian')
'''
    for name in fields + list(PREC_FIELDS):
        post += f'    write(cap_u) (({name}(its:ite,cap_k,cap_j),cap_j=jts,jte),cap_k=kts,kte)\n'
    for name in SURFACE_FIELDS:
        post += f'    write(cap_u) {name}(its:ite,jts:jte)\n'
    post += '    close(cap_u)\n  endif\n'
    return s.replace(exit_site, post + '\n' + exit_site, 1).encode()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--j-index', type=int, required=True)
    parser.add_argument('--timesteps', type=int, nargs='+', required=True)
    args = parser.parse_args()
    if args.source.resolve() == args.output.resolve():
        parser.error('output must be a copy; never instrument the original source')
    output = instrument(args.source.read_bytes(), j_index=args.j_index,
                        timesteps=args.timesteps)
    args.output.write_bytes(output)
    print(json.dumps({'source_sha256': SOURCE_SHA256,
                      'instrumented_sha256': hashlib.sha256(output).hexdigest(),
                      'j_index': args.j_index, 'timesteps': args.timesteps}))


if __name__ == '__main__':
    main()
