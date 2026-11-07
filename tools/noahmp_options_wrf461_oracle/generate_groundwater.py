"""Offline WRF-v4 groundwater routine translation; original public-domain source only."""
from __future__ import annotations
import ast,hashlib,re
from .generate_dynamic_vic import split_args
from .build import LSM_SHA256
GROUND_SHA="ff4b2d7bcb4e948d43af2797e29cbec1e3bb2875b3e9bbf2421c39a6f840cf50"
SIG={
 'SHALLOWWATERTABLE':'const float*smcmax,const float*psisat,const float*bexp,const float*zsoil,const float*dzsnso,const float*smceq,float dt,float*smc,float&wtd,float&smcwtd,float&rech,float&qdrain',
 'UPDATEWTD':'const float*dzs,const float*zsoil,const float*smceq,float smcmax,float smcwlt,float psisat,float bexp,float&totwater,float&wtd,float*smc,float*sh2o,float&smcwtd,float&qspring',
 'GW_COLD':'const float*zsoil,const float*dzs,float bexp,float smcmax,float smcwlt,float dwsat,float dksat,float psisat,float deltat,float qlat,float qrf,float&wtd,float*smceq,float*smoi,float*sh2o,float&smcwtd',
 'EQSMOISTURE':'const float*zsoil,float smcmax,float smcwlt,float dwsat,float dksat,float bexp,float*smceq',
}
def translate(source,name):
 m=re.search(r'^\s*SUBROUTINE\s+'+name+r'\s*\(',source,re.M|re.I); z=re.search(r'END\s+SUBROUTINE\s+'+name,source[m.end():],re.I)
 t=source[m.start():m.end()+z.start()];t=t[t.upper().index('IMPLICIT NONE'):]
 arrays={'SMOI':1,'SMC':1,'SH2O':1,'SMCEQ':1,'DZS':1,'DZSNSO':1,'ZSOIL':0 if name=='UPDATEWTD'else 1,'SMCMAX':1,'PSISAT':1,'BEXP':1}
 local=[];body=[];pending=''
 def expression(s):
  s=s.upper().replace('PARAMETERS%','').replace('.TRUE.','True').replace('.FALSE.','False')
  for a,b in (('.GT.','>'),('.GE.','>='),('.LT.','<'),('.LE.','<='),('.EQ.','=='),('.NE.','!='),('.AND.',' and '),('.OR.',' or '),('.NOT.',' not ')):s=s.replace(a,b)
  def e(n):
   if isinstance(n,ast.Name):return 'NSOIL'if n.id=='NSOIL'else n.id.lower()
   if isinstance(n,ast.Constant):return repr(float(n.value))+'f'
   if isinstance(n,ast.UnaryOp):return ('f_neg('+e(n.operand)+')')if isinstance(n.op,ast.USub)else'(!'+e(n.operand)+')'
   if isinstance(n,ast.BinOp):return {ast.Add:'FADD',ast.Sub:'FSUB',ast.Mult:'FMUL',ast.Div:'FDIV',ast.Pow:'runoff_powf'}[type(n.op)]+'('+e(n.left)+','+e(n.right)+')'
   if isinstance(n,ast.Compare):return '('+e(n.left)+{ast.Gt:'>',ast.GtE:'>=',ast.Lt:'<',ast.LtE:'<=',ast.Eq:'==',ast.NotEq:'!='}[type(n.ops[0])]+e(n.comparators[0])+')'
   if isinstance(n,ast.BoolOp):return '('+('&&'if isinstance(n.op,ast.And)else'||').join(e(v)for v in n.values)+')'
   if isinstance(n,ast.Call):
    fn=n.func.id;args=[e(a)for a in n.args]
    if fn in arrays:return fn.lower()+'[(int)('+args[0]+')-'+str(arrays[fn])+']'
    if fn in ('MIN','MAX'):
     v=args[0]
     for a in args[1:]:v=('f_min'if fn=='MIN'else'f_max')+'('+v+','+a+')'
     return v
    return {'ABS':'fabsf','EXP':'glibc_expf','SQRT':'sqrtf'}[fn]+'('+','.join(args)+')'
   raise ValueError(ast.dump(n))
  return e(ast.parse(s.strip(),mode='eval').body)
 def statement(s):
  s=s.strip()
  if s=='EXIT':return 'break;'
  if s.startswith('CALL EQSMOISTURE'):
   return 'groundwater_eqsmoisture(zsoil,smcmax,smcwlt,dwsat,dksat,bexp,smceq);'
  if s=='SMCEQ = SMCMAX':return 'for(int lane_k=0;lane_k<NSOIL;++lane_k)smceq[lane_k]=smcmax;'
  if s.startswith('ZSOIL0(1:NSOIL)'):return 'for(int lane_k=0;lane_k<NSOIL;++lane_k)zsoil0[lane_k+1]=zsoil[lane_k];'
  if s in ('SICE = SMC - SH2O','SH2O = SMC - SICE'):return 'for(int lane_k=1;lane_k<=NSOIL;++lane_k) '+statement(s.replace('SICE','SICE(LANE_K)').replace('SMC','SMC(LANE_K)').replace('SH2O','SH2O(LANE_K)'))
  a,b=s.split('=',1);return expression(a)+'='+expression(b)+';'
 for line in t.splitlines():
  raw=line.split('!',1)[0].strip().upper()
  if not raw:continue
  if raw.endswith('&'):pending+=raw[:-1];continue
  raw=pending+raw.lstrip('&');pending=''
  if '::'in raw:
   if 'INTENT'not in raw and not raw.startswith('TYPE'):
    kind='int'if raw.startswith('INTEGER')else'float'
    for v in split_args(raw.split('::',1)[1]):
     dim=re.search(r'DIMENSION\(([^)]+)\)',raw)
     if dim:
      dim=dim.group(1).strip();base=0 if dim.startswith('0:')else 1;arrays[v]=base;local.append(kind+' '+v.lower()+'[NSOIL'+('+1'if base==0 else'')+'];')
     else:local.append(kind+' '+v.lower()+';')
   continue
  if raw.startswith(('IMPLICIT','SUBROUTINE')):continue
  if raw in ('END IF','ENDIF','END DO','ENDDO'):body.append('}')
  elif raw=='ELSE':body.append('} else {')
  elif re.match(r'(ELSE\s*)?IF\s*\(',raw):
   a=raw.index('(');i=a+1;depth=1
   while depth:depth+=(raw[i]=='(')-(raw[i]==')');i+=1
   cond=expression(raw[a+1:i-1]);tail=raw[i:].strip();body.append(('} else 'if raw.startswith('ELSE')else'')+'if('+cond+')'+(' {'if tail=='THEN'else' '+statement(tail)))
  elif raw.startswith('DO '):
   v,b=raw[3:].split('=',1);args=split_args(b);a,last=args[:2];step=int(args[2])if len(args)>2 else 1
   body.append('for('+v.lower()+'=(int)('+expression(a)+');'+v.lower()+('>='if step<0 else'<=')+'(int)('+expression(last)+');'+v.lower()+(' --'if step<0 else' ++')+') {')
  else:body.append(statement(raw))
 return '// Complete native '+name+'; source line '+str(source[:m.start()].count('\n')+1)+'\n__device__ void groundwater_'+name.lower()+'('+SIG[name]+') {\n'+'\n'.join('    '+v for v in local+body)+'\n}\n'
def generate(lsm,ground,driver):
 assert hashlib.sha256(lsm).hexdigest()==LSM_SHA256
 assert hashlib.sha256(ground).hexdigest()==GROUND_SHA
 assert hashlib.sha256(driver).hexdigest()=='9010a757da994ed8796c63ca97da354eaf60c5c732df4ea9acad5bc62a973890'
 cold=driver.decode();start=cold.index('           IF ( ( BEXP > 0.0 )',cold.index('SUBROUTINE GROUNDWATER_INIT'));end=cold.index('!zero out some arrays',start)
 cold=cold[start:end]
 cold=re.sub(r'\bSMOISEQ\s*\(I,1:NSOIL,J\)\s*=\s*SMCEQ\s*\(1:NSOIL\)','',cold,flags=re.I)
 cold=re.sub(r'\bSMOISEQ\s*\(I,1:NSOIL,J\)\s*=\s*SMCMAX','SMCEQ = SMCMAX',cold,flags=re.I)
 cold=re.sub(r'\bSMOIS\(I,K,J\)','SMOI(K)',cold,flags=re.I)
 cold=re.sub(r'\bSH2O\(I,K,J\)','SH2O(K)',cold,flags=re.I)
 for n in ('WTD','QLAT','QRF'):cold=re.sub(r'\b'+n+r'\(I,J\)',n,cold,flags=re.I)
 cold=cold.replace('SMCWTDXY(I,J)','SMCWTD')
 cold='SUBROUTINE GW_COLD()\nIMPLICIT NONE\nINTEGER :: K,ITER\nREAL :: FRLIQ,SMCEQDEEP,AA,BBB,CC,DD,DX,FUNC,DFUNC,DDZ,EXPON,SMC,FLUX\n'+cold+'END SUBROUTINE GW_COLD\n'
 return '// WRF public domain; licenses/LICENSE-WRF-public-domain.txt.\n'+translate(lsm.decode(),'SHALLOWWATERTABLE')+translate(ground.decode(),'UPDATEWTD')+translate(driver.decode(),'EQSMOISTURE')+translate(cold,'GW_COLD')
if __name__=='__main__':
 import sys
 from pathlib import Path
 body=generate(*(Path(p).read_bytes()for p in sys.argv[1:4]))
 spatial=Path(__file__).with_name('groundwater_spatial.cuh').read_text()
 Path(sys.argv[4]).write_text('namespace noahmp_water_options {\n'+body+'}\n'+spatial)
