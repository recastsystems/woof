"""Offline translation of the complete pinned WRF-v4 tile procedures."""
from __future__ import annotations
import ast,hashlib,re
from .build import LSM_SHA256
from .generate_dynamic_vic import split_args
SCALARS={"PAR_"+n:f'p[{k+4}]' for k,n in enumerate(('KLAT_FAC','TDSMC_FAC','TD_DC','TD_DCOEF','TD_RADI','TD_SPAC','TD_DDRAIN','TD_ADEPTH','TD_D','TD_DEPTH','DRAIN_LAYER_OPT'))}
ARRAYS=('SH2O','SMC','SICE','ZSOIL','SLDPTH','DZSNSO','WCND','OVRFC','AVFC','ZLAYER','TDFRAC','TD_SATZ','KLATK','RMSH2O')
SIG={
'TILE_DRAIN':'const float*p,float*sh2o,float*smc,const float*sice,const float*zsoil,float&qtldrn,float dt',
'TILE_HOOGHOUDT':'const float*p,const float*wlt,const float*wcnd,float*sh2o,float*smc,const float*sice,const float*zsoil,const float*dzsnso,float dt,float dx,float&qtldrn,float&zwt',
'TD_EQUIVALENT_DEPTH':'float td_d,float td_l,float td_rd,float&td_de',
'TD_FINDZWAT':'const float*p,const float*smc,const float*sh2o,const float*sice,const float*zsoil,const float*sldpth,float&watbled'}
def expression(s):
 s=s.upper().replace('PARAMETERS%','PAR_')
 for a,b in (('.GT.','>'),('.GE.','>='),('.LT.','<'),('.LE.','<='),('.EQ.','=='),('.NE.','!='),('.AND.',' and '),('.OR.',' or ')):s=s.replace(a,b)
 def e(n):
  if isinstance(n,ast.Name):return SCALARS.get(n.id, 'NSOIL' if n.id=='NSOIL' else n.id.lower())
  if isinstance(n,ast.Constant):return repr(float(n.value))+'f'
  if isinstance(n,ast.UnaryOp) and isinstance(n.op,ast.USub):return 'f_neg('+e(n.operand)+')'
  if isinstance(n,ast.BinOp):return {ast.Add:'FADD',ast.Sub:'FSUB',ast.Mult:'FMUL',ast.Div:'FDIV'}[type(n.op)]+'('+e(n.left)+','+e(n.right)+')'
  if isinstance(n,ast.Compare):return '('+e(n.left)+{ast.Gt:'>',ast.GtE:'>=',ast.Lt:'<',ast.LtE:'<=',ast.Eq:'==',ast.NotEq:'!='}[type(n.ops[0])]+e(n.comparators[0])+')'
  if isinstance(n,ast.BoolOp):return '('+('&&' if isinstance(n.op,ast.And) else '||').join(e(v) for v in n.values)+')'
  if isinstance(n,ast.Call):
   name=n.func.id
   if name in ('PAR_SMCREF','PAR_SMCWLT'):return ('p[(int)('+e(n.args[0])+')-1]' if name=='PAR_SMCREF' else 'wlt[(int)('+e(n.args[0])+')-1]')
   if name in ARRAYS:return name.lower()+'[(int)('+e(n.args[0])+')-1]'
   return {'LOG':'glibc_logf','EXP':'glibc_expf','INT':'int'}[name]+'('+','.join(e(v) for v in n.args)+')'
  raise ValueError(ast.dump(n))
 return e(ast.parse(s.strip(),mode='eval').body)
def statement(s,arrays):
 if s=='EXIT':return 'break;'
 if s.startswith('CALL '):
  name,args=s[5:].split('(',1);a=[x for x in split_args(args.rsplit(')',1)[0]) if x not in ('NSOIL','NSNOW')]
  if name.strip()=='TD_FINDZWAT':a.insert(1,'WLT')
  return name.lower()+'('+','.join('p' if v=='PARAMETERS' else expression(v) for v in a)+');'
 a,b=s.split('=',1);a=a.strip();b=b.strip()
 if a in arrays:return 'for(int q=0;q<NSOIL;++q) '+a.lower()+'[q]='+expression(b)+';'
 return expression(a)+'='+expression(b)+';'
def routine(source,name):
 m=re.search(r'^\s*SUBROUTINE\s+'+name+r'\s*\(',source,re.M|re.I);z=re.search(r'END SUBROUTINE\s+'+name,source[m.end():],re.I);text=source[m.start():m.end()+z.start()]
 # Native WRF, without the separately licensed WRF-Hydro consumer.
 keep=[];taking=True;stack=[]
 for line in text.splitlines():
  raw=line.strip()
  if raw.startswith('#ifdef WRF_HYDRO'):stack.append(taking);taking=False
  elif raw.startswith('#else'):taking=stack[-1] and not taking
  elif raw.startswith('#endif'):taking=stack.pop()
  elif taking:keep.append(line)
 text='\n'.join(keep)
 text=text[text.upper().index('IMPLICIT NONE'):]
 local=[];body=[];pending='';arrays=set()
 for line in text.splitlines():
  raw=line.split('!',1)[0].strip().upper()
  if not raw:continue
  if '::' in raw:
   if 'INTENT' not in raw and 'TYPE' not in raw:
    vals=split_args(raw.split('::',1)[1]);kind='int' if raw.startswith('INTEGER') else 'float'
    for v in vals:
     v=v.strip()
     if 'DIMENSION' in raw:arrays.add(v);local.append(kind+' '+v.lower()+'[NSOIL];')
     elif '=' in v:
      a,b=v.split('=',1);local.append(kind+' '+a.lower()+'='+('3.142857074737548828125f' if a.strip()=='PII' else expression(b))+';')
     else:local.append(kind+' '+v.lower()+';')
   continue
  if raw.startswith(('SUBROUTINE','IMPLICIT')):continue
  if raw.endswith('&'):pending+=raw[:-1];continue
  raw=pending+raw.lstrip('&');pending=''
  if raw in ('END IF','ENDIF','END DO','ENDDO'):body.append('}')
  elif raw=='ELSE':body.append('} else {')
  elif re.match(r'(ELSE\s+)?IF\s*\(',raw):
   a=raw.index('(');i=a+1;depth=1
   while depth:depth+=(raw[i]=='(')-(raw[i]==')');i+=1
   cond=expression(raw[a+1:i-1]);tail=raw[i:].strip();body.append(('} else ' if raw.startswith('ELSE') else '')+'if('+cond+')'+(' {' if tail=='THEN' else ' '+statement(tail,arrays)))
  elif raw.startswith('DO '):
   v,b=raw[3:].split('=',1);bb=split_args(b);a,last=bb[:2];step=bb[2] if len(bb)==3 else '1';v=v.strip().lower();lo=a.lower().replace('nsoil','NSOIL');hi=last.lower().replace('nsoil','NSOIL');cmp='>=' if step=='-1' else '<=';body.append(f'for({v}={lo};{v}{cmp}{hi};{v}+={step}) {{')
  else:body.append(statement(raw,arrays))
 sig=SIG[name]
 if name=='TD_FINDZWAT':sig=sig.replace('const float*p,','const float*p,const float*wlt,')
 return '// Complete native '+name+'; module_sf_noahmplsm.F:'+str(source[:m.start()].count('\n')+1)+'\n__device__ void '+name.lower()+'('+sig+') {\n'+'\n'.join('    '+v for v in local+body)+'\n}\n'
def generate(source):
 assert hashlib.sha256(source).hexdigest()==LSM_SHA256
 s=source.decode();return '// BEGIN GENERATED NATIVE TILE DRAINAGE\n// WRF public domain; licenses/LICENSE-WRF-public-domain.txt.\n'+ '\n'.join(routine(s,n) for n in ('TD_EQUIVALENT_DEPTH','TD_FINDZWAT','TILE_DRAIN','TILE_HOOGHOUDT'))+'// END GENERATED NATIVE TILE DRAINAGE\n'
