"""Build the unchanged public native cloud routine with a C ABI wrapper."""
from pathlib import Path
import argparse,re,subprocess,json,hashlib
from verify_sources import verify
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',required=True,type=Path);a=p.parse_args()
r=Path(__file__).resolve().parent;verify(r);a.out.mkdir(parents=True,exist_ok=True)
s=(r/'upstream/phys/module_radiation_driver.F').read_text()
m=re.search(r'(?is)   SUBROUTINE cal_cldfra1\(.*?END SUBROUTINE cal_cldfra1',s);assert m
m2=re.search(r'(?is)   SUBROUTINE cal_cldfra2\(.*?END SUBROUTINE cal_cldfra2',s);assert m2
source='module module_state_description\n integer,parameter::KFCUPSCHEME=1,KFETASCHEME=11,FER_MP_HIRES=5,FER_MP_HIRES_ADVECT=15\nend module\nmodule oracle\ncontains\n' +m.group(0)+'\n'+m2.group(0)+'\n'+'subroutine run(n,qv,qc,qi,qs,t,p,c,fi,fs,mp,fc,simple) bind(c)\n use iso_c_binding\n integer(c_int),value::n,fi,fs,mp,fc,simple\n real(c_float)::qv(n),qc(n),qi(n),qs(n),t(n),p(n),c(n)\n integer::flags(n),m\n logical::a,b\n a=fi/=0;b=fs/=0;m=mp\n if(simple/=0)then\n call cal_cldfra2(c,qc,qi,fc/=0,a, &\n ids=1,ide=n,jds=1,jde=1,kds=1,kde=1,ims=1,ime=n,jms=1,jme=1,kms=1,kme=1, &\n its=1,ite=n,jts=1,jte=1,kts=1,kte=1)\n else\n call cal_cldfra1(c,qv,qc,qi,qs,.true.,fc/=0,a,b,t,p, &\n mp_physics=m,cldfra1_flag=flags, &\n ids=1,ide=n,jds=1,jde=1,kds=1,kde=1,ims=1,ime=n,jms=1,jme=1,kms=1,kme=1, &\n its=1,ite=n,jts=1,jte=1,kts=1,kte=1)\n endif\nend subroutine\nend module\n'
f=a.out/'cloud_oracle.f90';f.write_text(source)
cmd=['gfortran','-shared','-fPIC','-O0','-ffree-line-length-none','-ffp-contract=off','-fno-fast-math',str(f),'-o',str(a.out/'cloud_oracle.so')]
subprocess.run(cmd,cwd=a.out,check=True)
(a.out/'receipt.json').write_text(json.dumps(dict(command=cmd,source_sha256=hashlib.sha256(f.read_bytes()).hexdigest(),native_arithmetic='unchanged cal_cldfra1 and cal_cldfra2; constants-only module and C wrapper'),indent=2)+'\n')
