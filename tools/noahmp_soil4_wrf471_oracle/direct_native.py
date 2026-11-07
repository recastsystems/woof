"""Build unchanged strict WRF 4.7.1 SFLX; replay explicit direct vectors twice.

Infrastructure-only wrapper: restores all captured formal inputs and the
complete parameter type, then supplies declared depth-varying conductivity.
Both calls execute the full unmodified native routine. The second call uses
the first call's INOUT state. No expected GPU outputs enter this producer.
"""
from pathlib import Path
import argparse,hashlib,json,re,shutil,subprocess
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--source',type=Path,required=True)
p.add_argument('--metadata',type=Path,required=True)
p.add_argument('--captures',type=Path,required=True)
p.add_argument('--out',type=Path,required=True)
a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
meta=json.loads(a.metadata.read_text());src=a.source/'phys/noahmp/src/module_sf_noahmplsm.F'
assert hashlib.sha256(src.read_bytes()).hexdigest()==meta['original_sha256']=='74b475600ad1c999fe43299bcaf4d3e5eaf59b9f861c47fdbff4d69739e3a45b'
args=meta['args'];pars=meta['parameters']
source=src.read_text();sig=source[source.index('  SUBROUTINE NOAHMP_SFLX ('):];sig=sig[:sig.index(')')+1]
sig=re.sub(r'#ifdef WRF_HYDRO\s*\n.*?#endif', '', sig, flags=re.S)
sig=''.join(line.split('!')[0] for line in sig.splitlines()).replace('&','')
formals=[v.strip().lower() for v in sig[sig.index('(')+1:-1].split(',')]
assert set(formals)=={'parameters'}|{r['name'] for r in args}
def dim(row):
 return ',dimension('+':'.join(row['bounds'])+')' if row['bounds'] else ''
lines=['program direct_replay','use module_sf_noahmplsm','implicit none','type(noahmp_parameters) :: parameters',
       'integer :: u,v,k,idx,bits,ios,step,c1,c2,c3,c4','character(len=1024) :: file,outdir,line',
       'character(len=256) :: tag,stage,field,raw,output']
for row in args:
 d=dim(row)
 d=re.sub('nsoil','4',d,flags=re.I);d=re.sub('nsnow','3',d,flags=re.I)
 lines.append(row['kind']+d+' :: '+row['name'])
lines+=['call get_command_argument(1,file)','call get_command_argument(2,outdir)',
        'open(newunit=u,file=trim(file),status="old")','read(u,*)',
        'do',' read(u,"(A)",iostat=ios)line',' if(ios/=0)exit',
        ' read(line,*,iostat=ios)tag,stage,field,idx,raw',' if(ios/=0.or.trim(stage)/="input")cycle',
        ' if(trim(field)=="llanduse")then',
        '  c1=index(line,","); c2=c1+index(line(c1+1:),","); c3=c2+index(line(c2+1:),","); c4=c3+index(line(c3+1:),",")',
        '  llanduse=trim(line(c4+1:)); cycle',' endif',' read(raw,"(Z8)")bits',' select case(trim(field))']
for row,ispar in [(r,False)for r in args if r['intent'] in ('IN','INOUT') and r['name']!='llanduse']+[(r,True)for r in pars]:
 expr=('parameters%' if ispar else '')+row['name']+('(idx)' if row['bounds'] else '')
 val='bits/=0' if row['kind']=='logical' else 'bits' if row['kind']=='integer' else 'transfer(bits,0.0)'
 lines += [' case("'+('par_' if ispar else '')+row['name']+'")',expr+'='+val]
lines+=[' end select','enddo','close(u)',
        'call noahmp_options(4,1,1,3,1,1,1,3,2,1,2,1,1,4,1,0,0,0,0,0)',
        'soil_update_steps=1','calculate_soil=.true.',
        '! Declared noncategory direct input; all other parameter words stay captured.',
        'parameters%dksat=parameters%dksat*[1.05,1.0,.95,.9]',
        'parameters%dwsat=parameters%dwsat*[1.05,1.0,.95,.9]',
        'do step=1,2',' write(output,"(A,I0,A,I0,A)")trim(outdir)//"/noahmp-real-",iloc,"-",step,".csv"',
        ' open(newunit=v,file=trim(output),status="replace")','write(v,"(A)")"case,stage,field,index,hex"']
def emit(row,stage,ispar=False):
 name=row['name'];expr=('parameters%'if ispar else '')+name;label=('par_'if ispar else '')+name
 if row['kind'].startswith('character'):
  return ['write(v,"(A)")"real,input,llanduse,0,"//trim(llanduse)']
 val='merge(1,0,'+expr+')' if row['kind']=='logical' else expr
 if row['bounds']:
  lo,hi=row['bounds'];val=val.replace(expr,expr+'(k)')
  return ['do k='+lo+','+hi,'write(v,"(A,I0,A,Z8.8)")"real,'+stage+','+label+',",k,",",transfer('+val+',0)','enddo']
 return ['write(v,"(A,Z8.8)")"real,'+stage+','+label+',0,",transfer('+val+',0)']
for r in args:
 if r['intent'] in ('IN','INOUT'):lines+=emit(r,'input')
for r in pars:lines+=emit(r,'input',True)
for name in ('dveg','opt_btr','opt_rsf','opt_alb','opt_stc','soil_update_steps'):
 lines+=emit(dict(name=name,kind='integer',bounds=None),'input')
lines+=['call NOAHMP_SFLX('+', &\n'.join(formals)+')']
for r in args:
 if r['intent'] in ('OUT','INOUT'):lines+=emit(r,'output')
lines+=['close(v)','enddo','end program direct_replay']
(a.out/'direct_replay.f90').write_text('\n'.join(lines)+'\n')
flags=['-cpp','-ffree-form','-ffree-line-length-none','-O2','-fno-fast-math','-ffp-contract=off','-fno-tree-vectorize']
files=[Path(__file__).parent.parent/'noahmp_wrf461_oracle/stub_wrf.F90',a.source/'phys/module_sf_gecros.F',src,a.out/'direct_replay.f90']
files=[p.resolve() for p in files]
commands=[]
for f in files:
 cmd=['gfortran',*flags,'-c',str(f)];commands.append(cmd);subprocess.run(cmd,cwd=a.out,check=True)
cmd=['gfortran',*flags,'-o','direct_replay',*[f.stem+'.o'for f in files]];commands.append(cmd);subprocess.run(cmd,cwd=a.out,check=True)
for i,path in enumerate(sorted(a.captures.glob('noahmp-real-*.csv'))):
 out=a.out/f'column-{i}';out.mkdir(exist_ok=True)
 subprocess.run([str((a.out/'direct_replay').resolve()),str(path.resolve()),str(out.resolve())],cwd=a.out,check=True)
receipt=dict(compiler=subprocess.check_output(['gfortran','--version'],text=True).splitlines()[0],flags=flags,commands=commands,unmodified_native_lsm=True,columns=i+1,calls_per_column=2,source_sha256={str(f):hashlib.sha256(f.read_bytes()).hexdigest()for f in files},declared_direct_change='DKSAT/DWSAT depth factors 1.05,1.0,.95,.9; second call retains native INOUT state')
(a.out/'build.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt,indent=2))
