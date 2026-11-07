"""Profile unmodified native REAL32 Deng/NSCV on kept real captures.

The wrapper restores captured in/out state outside each measured call.
Diagnostic-only WRF host services are stubbed. No physics is replaced.
"""
import argparse
import json
from pathlib import Path
import subprocess
import time
import translate

FLAGS=['-O2','-fno-fast-math','-ffp-contract=off','-fno-tree-vectorize',
       '-ffree-form','-ffree-line-length-none','-cpp','-pg']


def deng_wrapper():
    _, rows=translate.routines();r=rows[0]
    decl=[];allocate=[];reads=[];copies=[];saves=[]
    for name in r['args'][18:]:
        v=r['vars'][name];typ={'float':'real','int':'integer','bool':'logical'}[v.typ]
        if not v.dims:
            decl.append(f'{typ} :: {name}');reads.append(f'read(unit) {name}');continue
        rank=len(v.dims)
        shape='nz+1' if name=='dsigma' else 'nx,ny' if rank==2 else 'nx,100,ny' if name=='ainckfsa' else 'nx,nz+1,ny'
        suffix='(:'+ ',:'*(rank-1)+')'
        decl.append(f'{typ},allocatable :: {name}{suffix}')
        allocate.append(f'allocate({name}({shape})); {name}=0')
        section=name if rank!=3 or name in ('w','z_at_w','ainckfsa') else f'{name}(:,1:nz,:)'
        reads.append(f'read(unit) {section}')
        if v.intent in ('inout','out'):
            decl.append(f'{typ},allocatable :: saved_{name}{suffix}')
            saves.append(f'saved_{name}={name}');copies.append(f'{name}=saved_{name}')
    bounds=dict(ids='1',ide='nx+1',jds='1',jde='ny+1',kds='1',kde='nz+1',
                ims='1',ime='nx',jms='1',jme='ny',kms='1',kme='nz+1',
                its='1',ite='nx',jts='1',jte='ny',kts='1',kte='nz')
    args=[n+'='+bounds.get(n,n) for n in r['args']]
    return '\n'.join(['program profile','use module_shcu_deng','implicit none',
        'integer :: header(7),unit,nx,ny,nz,sample,repeats',
        'real :: tables(110423),begin_time,end_time,total',
        'character(1024) :: input,count',*decl,
        'call get_command_argument(1,input);call get_command_argument(2,count)',
        'read(count,*)repeats',
        "open(newunit=unit,file=trim(input),access='stream',form='unformatted',convert='little_endian')",
        'read(unit)header,tables','nx=header(3)-header(2)+1;ny=header(5)-header(4)+1;nz=header(7)-header(6)+1',
        *allocate,*reads,'close(unit)',*saves,
        'call kf_lutab(svp1,svp2,svp3,svpt0)','total=0.',
        'do sample=1,repeats',*copies,'call cpu_time(begin_time)',
        'call deng_shcu_driver('+', &\n'.join(args)+')',
        'call cpu_time(end_time)','total=total+(end_time-begin_time)','enddo',
        "write(*,'(A,F14.6,A,I0)') 'profile_cpu_seconds=',total,',calls=',repeats",'end program'])+'\n'


STUB='''module module_wrf_error
contains
subroutine get_wrf_debug_level(level)
integer,intent(out)::level
level=0
end subroutine
subroutine wrf_message(message)
character(*),intent(in)::message
end subroutine
end module
logical function wrf_dm_on_monitor()
wrf_dm_on_monitor=.false.
end function
'''



def nscv_wrapper():
    source=(Path(__file__).parents[1]/'shallow_wrf471_nscv_witness.F90').read_text()
    source=source.replace('integer :: header(8),mp,pbl,nx,ny,nz,u,v,ios',
        'integer :: header(8),mp,pbl,nx,ny,nz,u,v,ios,sample,repeats\n real :: begin_time,end_time,total')
    source=source.replace('read(seed,*) witness_seed','read(seed,*) repeats')
    source=source.replace(' call shcu_nscv(',
        ' total=0.\n do sample=1,repeats\n o=0.;d=0.\n call cpu_time(begin_time)\n call shcu_nscv(')
    marker=' open(newunit=u,file=trim(output)'
    start=source.index(marker)
    source=source[:start]+" call cpu_time(end_time)\n total=total+end_time-begin_time\n enddo\n write(*,'(A,F14.6,A,I0)') 'profile_cpu_seconds=',total,',calls=',repeats\nend program\n"
    return source


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('captures',type=Path);p.add_argument('output',type=Path)
    p.add_argument('--repeats',type=int,default=100)
    p.add_argument('--nscv-source',type=Path);p.add_argument('--nscv-captures',type=Path)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    output=a.output.resolve()
    (output/'stubs.f90').write_text(STUB)
    (output/'profile.f90').write_text(deng_wrapper())
    command=['gfortran',*FLAGS,'stubs.f90',str(translate.SOURCE.resolve()),'profile.f90','-o','profile']
    subprocess.run(command,cwd=output,check=True)
    records=[]
    # Four independent tiles; each process is a short native-only profile.
    for capture in sorted(a.captures.glob('deng-00031-*.bin')):
        started=time.monotonic()
        run=subprocess.run([str(output/'profile'),str(capture.resolve()),str(a.repeats)],
                           cwd=output,check=True,capture_output=True,text=True)
        label=capture.stem
        (output/(label+'.log')).write_text(run.stdout+run.stderr)
        prof=subprocess.run(['gprof','-b','profile','gmon.out'],cwd=output,check=True,capture_output=True,text=True)
        (output/(label+'.gprof')).write_text(prof.stdout)
        records.append(dict(capture=str(capture),stdout=run.stdout,wall_seconds=time.monotonic()-started))
    if a.nscv_source is not None:
        (output/'nscv-profile.f90').write_text(nscv_wrapper())
        subprocess.run(['gfortran',*FLAGS,str(a.nscv_source.resolve()),'nscv-profile.f90','-o','nscv-profile'],cwd=output,check=True)
        for capture in sorted(a.nscv_captures.glob('*00031*.bin')):
            run=subprocess.run([str(output/'nscv-profile'),str(capture.resolve()),'unused',str(a.repeats)],cwd=output,check=True,capture_output=True,text=True)
            label='nscv-'+capture.stem
            (output/(label+'.log')).write_text(run.stdout+run.stderr)
            prof=subprocess.run(['gprof','-b','nscv-profile','gmon.out'],cwd=output,check=True,capture_output=True,text=True)
            (output/(label+'.gprof')).write_text(prof.stdout)
            records.append(dict(capture=str(capture),stdout=run.stdout))
    (output/'manifest.json').write_text(json.dumps(dict(command=command,records=records),indent=2)+'\n')


if __name__=='__main__':main()
