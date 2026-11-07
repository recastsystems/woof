"""Offline complete translation of WRF-v4's five irrigation procedures.

Only the pinned WRF public-domain source is consumed. No v5/MPAS code.
"""
from __future__ import annotations
import ast,hashlib,re
from .build import LSM_SHA256
from .generate_dynamic_vic import split_args
LAYERS=('SMCMAX','SMCWLT','SMCREF','BEXP','DKSAT','DWSAT')
SCALARS=('IRR_FRAC','IRR_HAR','IRR_LAI','IRR_MAD','FILOSS','SPRIR_RATE','MICIR_RATE','FIRTFAC','IR_RAIN','PLTDAY','HSDAY')
SIG={
 'TRIGGER_IRRIGATION':'const float*irr,int nroot,int opt_irr,int opt_irrm,const float*zsoil,const float*sh2o,float fveg,float julian,float irrfra,float lai,float sifac,float mifac,float fifac,int&ircntsi,int&ircntmi,int&ircntfi,float&iramtsi,float&iramtmi,float&iramtfi',
 'SPRINKLER_IRRIGATION':'const float*irr,float dt,const float*sh2o,const float*smc,float t2,float windu,float windv,float eair,float sifac,float&iramtsi,float&irevplos,float&irsirate',
 'MICRO_IRRIGATION':'const float*irr,float dt,const float*sh2o,const float*smc,const float*sice,float mifac,float&iramtmi,float&irmirate',
 'FLOOD_IRRIGATION':'const float*irr,float dt,const float*sh2o,const float*smc,const float*sice,float fifac,float&iramtfi,float&irfirate',
 'IRR_PHILIP_INFIL':'const float*irr,const float*smc,const float*sh2o,const float*sice,float dt,float&fsur',
}
ARRAYS=('ZSOIL','SH2O','SMC','SICE')
def expression(s):
 s=s.upper().replace('PARAMETERS%','PAR_').replace('.TRUE.','True').replace('.FALSE.','False')
 for a,b in (('.GT.','>'),('.GE.','>='),('.LT.','<'),('.LE.','<='),('.EQ.','=='),('.NE.','!='),('.AND.',' and '),('.OR.',' or ')):s=s.replace(a,b)
 def e(n):
  if isinstance(n,ast.Name):
   if n.id=='PAR_NROOT':return 'nroot'
   if n.id.startswith('PAR_') and n.id[4:] in SCALARS:return f'irr[{24+SCALARS.index(n.id[4:])}]'
   return 'NSOIL' if n.id=='NSOIL' else n.id.lower()
  if isinstance(n,ast.Constant):return ('true' if n.value else 'false') if isinstance(n.value,bool) else repr(float(n.value))+'f'
  if isinstance(n,ast.UnaryOp) and isinstance(n.op,ast.USub):return 'f_neg('+e(n.operand)+')'
  if isinstance(n,ast.BinOp) and isinstance(n.op,ast.Pow) and isinstance(n.right,ast.Constant) and n.right.value==2:return 'FMUL('+e(n.left)+','+e(n.left)+')'
  if isinstance(n,ast.BinOp):return {ast.Add:'FADD',ast.Sub:'FSUB',ast.Mult:'FMUL',ast.Div:'FDIV',ast.Pow:'runoff_powf'}[type(n.op)]+'('+e(n.left)+','+e(n.right)+')'
  if isinstance(n,ast.Compare):return '('+e(n.left)+{ast.Gt:'>',ast.GtE:'>=',ast.Lt:'<',ast.LtE:'<=',ast.Eq:'==',ast.NotEq:'!='}[type(n.ops[0])]+e(n.comparators[0])+')'
  if isinstance(n,ast.BoolOp):return '('+('&&' if isinstance(n.op,ast.And) else '||').join(e(v) for v in n.values)+')'
  if isinstance(n,ast.Call):
   name=n.func.id;args=[e(a) for a in n.args]
   if name.startswith('PAR_') and name[4:] in LAYERS:return f'irr[{4*LAYERS.index(name[4:])}+(int)('+args[0]+')-1]'
   if name in ARRAYS:return name.lower()+'[(int)('+args[0]+')-1]'
   if name in ('MIN','MAX'):
    v=args[0]
    for a in args[1:]:v=('f_min' if name=='MIN' else 'f_max')+'('+v+','+a+')'
    return v
   return {'SQRT':'sqrtf','EXP':'glibc_expf','ISNAN_LSM':'isnan'}[name]+'('+','.join(args)+')'
  raise ValueError(ast.dump(n))
 return e(ast.parse(s.strip(),mode='eval').body)
def statement(s):
 if s.startswith('CALL '):
  name,args=s[5:].split('(',1);a=[v for v in split_args(args.rsplit(')',1)[0]) if v!='NSOIL'];name=name.strip()
  if name=='WDFCND2':
   _,wdf,wcnd,smc,sice,layer=a;k='(int)('+expression(layer)+')-1'
   return 'd_wdfcnd2('+','.join([expression(smc),expression(sice),*[f'irr[{4*LAYERS.index(n)}+{k}]' for n in ('SMCMAX','BEXP','DWSAT','DKSAT')],'&'+wdf.lower(),'&'+wcnd.lower()])+');'
  return name.lower()+'('+','.join('irr' if v=='PARAMETERS' else expression(v) for v in a)+');'
 a,b=s.split('=',1)
 if a.strip() in ('IRCNTSI','IRCNTMI','IRCNTFI') and b.replace(' ','')==a.strip()+'+1':return '++'+a.strip().lower()+';'
 return expression(a)+'='+expression(b)+';'
def routine(source,name):
 m=re.search(r'^\s*SUBROUTINE\s+'+name+r'\s*\(',source,re.M|re.I);z=re.search(r'END SUBROUTINE\s+'+name,source[m.end():],re.I);t=source[m.start():m.end()+z.start()];t=t[t.upper().index('IMPLICIT NONE'):]
 local=[];body=[];pending=''
 if name=='SPRINKLER_IRRIGATION':local.append('int lane_k;')
 for line in t.splitlines():
  raw=line.split('!',1)[0].strip().upper()
  if not raw:continue
  if '::' in raw:
   if 'INTENT' not in raw and not raw.startswith('TYPE'):
    kind='int' if raw.startswith('INTEGER') else 'bool' if raw.startswith('LOGICAL') else 'float'
    for v in split_args(raw.split('::',1)[1]):local.append(kind+' '+v.lower()+('[NSOIL]' if 'DIMENSION' in raw else '')+';')
   continue
  if raw.startswith(('IMPLICIT','SUBROUTINE')):continue
  if raw.endswith('&'):pending+=raw[:-1];continue
  raw=pending+raw.lstrip('&');pending=''
  if raw.startswith('SICE(:) ='):
   body.append('for(lane_k=1;lane_k<=NSOIL;++lane_k) '+statement(raw.replace('(:)','(LANE_K)')))
  elif raw in ('END IF','ENDIF','END DO','ENDDO'):body.append('}')
  elif raw=='ELSE':body.append('} else {')
  elif re.match(r'(ELSE\s+)?IF\s*\(',raw):
   a=raw.index('(');i=a+1;depth=1
   while depth:depth+=(raw[i]=='(')-(raw[i]==')');i+=1
   cond=expression(raw[a+1:i-1]);tail=raw[i:].strip();body.append(('} else ' if raw.startswith('ELSE') else '')+'if('+cond+')'+(' {' if tail=='THEN' else ' '+statement(tail)))
  elif raw.startswith('DO '):
   v,b=raw[3:].split('=',1);a,last=split_args(b);body.append('for('+v.lower()+'=(int)('+expression(a)+');'+v.lower()+'<=(int)('+expression(last)+');++'+v.lower()+') {')
  else:body.append(statement(raw))
 return '// Complete native '+name+'; module_sf_noahmplsm.F:'+str(source[:m.start()].count('\n')+1)+'\n__device__ void '+name.lower()+'('+SIG[name]+') {\n'+'\n'.join('    '+v for v in local+body)+'\n}\n'
def generate(source):
 assert hashlib.sha256(source).hexdigest()==LSM_SHA256
 return '// BEGIN GENERATED NATIVE IRRIGATION\n// WRF public domain; licenses/LICENSE-WRF-public-domain.txt.\n'+ '\n'.join(routine(source.decode(),n) for n in ('IRR_PHILIP_INFIL','TRIGGER_IRRIGATION','SPRINKLER_IRRIGATION','MICRO_IRRIGATION','FLOOD_IRRIGATION'))+'// END GENERATED NATIVE IRRIGATION\n'
if __name__=='__main__':
 import sys
 from pathlib import Path
 Path(sys.argv[2]).write_text(generate(Path(sys.argv[1]).read_bytes()))
