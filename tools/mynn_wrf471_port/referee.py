#!/usr/bin/env python3
"""Generate an unmodified-module native replay executable for kept captures."""
import argparse,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('schema',type=Path);p.add_argument('output',type=Path);a=p.parse_args()
meta=json.loads(a.schema.read_text());fields=meta.get('arguments',meta)
nz=str(meta.get('nz',49))
decls=[];reads=[];writes=[];args=[]
for name,f in fields.items():
 if f['type']=='character':continue
 typ={'real':'real','integer':'integer','logical':'logical'}[f['type']]
 dims=f['dims']
 if dims and any(n in dims for n in ('nchem','ndvel')):
  decls.append(typ+', allocatable :: '+name+'('+','.join(':' for _ in dims.split(','))+')')
  reads.append('allocate('+name+'('+dims.replace('kts:kte','1:'+nz)+'))')
 else:
  if dims:dims=dims.replace('kts:kte','1:'+nz)
  decls.append(typ+' :: '+name+('('+dims+')' if dims else ''))
 reads.append('read(77) '+name);writes.append('write(78) '+name);args.append(name)
# errmsg is the only character argument and is restored to the native ABI.
args.insert(-1,'errmsg')
out=['program referee','use module_bl_mynnedmf','implicit none',*decls,
 'character(len=1024) :: infile,outfile,arg,value','character(len=512) :: errmsg','integer :: ia',
 'call get_command_argument(1,infile)','call get_command_argument(2,outfile)',
 "open(77,file=trim(infile),access='stream',form='unformatted',convert='big_endian')",*reads,'close(77)',
 'do ia=3,command_argument_count(),2','call get_command_argument(ia,arg)','call get_command_argument(ia+1,value)','select case(trim(arg))']
for name,f in fields.items():
 if not f['dims'] and f['type']!='character':out+=['case("'+name+'")','read(value,*) '+name]
out+=['case default','stop 9','end select','enddo',
 'call mynnedmf('+', &\n'.join(','.join(args[k:k+5]) for k in range(0,len(args),5))+')',
 "open(78,file=trim(outfile),access='stream',form='unformatted',status='replace',convert='big_endian')",*writes,'close(78)','end program']
a.output.write_text('\n'.join(out)+'\n')
