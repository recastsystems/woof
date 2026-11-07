"""Offline complete WRF v4 CARBON_CROP/CO2FLUX_CROP/GROWING_GDD translation."""
from __future__ import annotations
import ast,hashlib,re
from .build import LSM_SHA256
from .generate_dynamic_vic import split_args
SCALARS=('Q10MR','FOLN_MX','LFMR25','RTMR25','STMR25','GRAINMR25','FRA_GR','LEFREEZ','MRP','BIO2LAI','GDDTBASE','GDDTCUT','GDDS1','GDDS2','GDDS3','GDDS4','GDDS5','PLTDAY','HSDAY')
STAGES=('LFPT','STPT','RTPT','GRAINPT','LF_OVRC','RT_OVRC','ST_OVRC','DILE_FW','DILE_FC','LFCT','STCT','RTCT')
NAMES=('GROWING_GDD','CO2FLUX_CROP','CARBON_CROP')
INTS={'NSNOW','NSOIL','VEGTYP','IPA','IHA','PGS','J'}
ARRAYS={'ZSOIL':0,'DZSNSO':3,'STC':3,'SMC':0}
def expr(s):
 s=s.upper().replace('PARAMETERS%','PAR_')
 for a,b in (('.GT.','>'),('.GE.','>='),('.LT.','<'),('.LE.','<='),('.EQ.','=='),('.NE.','!='),('.AND.',' and '),('.OR.',' or ')):s=s.replace(a,b)
 def e(n):
  if isinstance(n,ast.Name):
   if n.id.startswith('PAR_'):return 'p.'+n.id[4:].lower()
   return {'NSNOW':'3','NSOIL':'4'}.get(n.id,n.id.lower())
  if isinstance(n,ast.Constant):return repr(float(n.value))+'f'
  if isinstance(n,ast.UnaryOp) and isinstance(n.op,ast.USub):return '-('+e(n.operand)+')'
  if isinstance(n,ast.BinOp):return {ast.Add:'AD',ast.Sub:'SU',ast.Mult:'MU',ast.Div:'DV',ast.Pow:'r_pow'}[type(n.op)]+'('+e(n.left)+','+e(n.right)+')'
  if isinstance(n,ast.Compare):return '('+e(n.left)+{ast.Gt:'>',ast.GtE:'>=',ast.Lt:'<',ast.LtE:'<=',ast.Eq:'==',ast.NotEq:'!='}[type(n.ops[0])]+e(n.comparators[0])+')'
  if isinstance(n,ast.BoolOp):return '('+('&&'if isinstance(n.op,ast.And)else'||').join(e(v)for v in n.values)+')'
  if isinstance(n,ast.Call):
   name=n.func.id;a=[e(v)for v in n.args]
   if name in ARRAYS:return name.lower()+'[(int)('+a[0]+')-1+'+str(ARRAYS[name])+']'
   if name.startswith('PAR_'):return 'p.'+name[4:].lower()+'[(int)('+a[0]+')-1]'
   if name in ('MIN','MAX'):
    v=a[0]
    for b in a[1:]:v=('mn'if name=='MIN'else'mx')+'('+v+','+b+')'
    return v
   return {'EXP':'r_exp','R':'crop_resp'}[name]+'('+','.join(a)+')'
  raise ValueError(ast.dump(n))
 return e(ast.parse(s.strip(),mode='eval').body)
def stmt(s):
 if s=='RETURN':return 'return;'
 if s.startswith('CALL '):
  name,a=s[5:].split('(',1);args=split_args(a.rsplit(')',1)[0]);return name.strip().lower()+'('+','.join('p'if v=='PARAMETERS'else expr(v)for v in args if v not in ('NSNOW','NSOIL'))+');'
 a,b=s.split('=',1);return expr(a)+'='+expr(b)+';'
def routine(source,name):
 m=re.search(r'^\s*SUBROUTINE\s+'+name+r'\s*\(',source,re.M|re.I);z=re.search(r'END SUBROUTINE\s+'+name,source[m.end():],re.I);t=source[m.start():m.end()+z.start()]
 clean='\n'.join(v.split('!',1)[0]for v in t.splitlines()).upper();a=clean.index('(');b=clean.index(')',a);args=[v for v in split_args(clean[a+1:b].replace('&',' '))if v not in ('NSNOW','NSOIL')];t=clean[b+1:]
 sig=[];locals=[];body=[];pending='';decls={}
 for line in t.splitlines():
  raw=line.split('!',1)[0].strip().upper()
  if not raw:continue
  if '::' in raw:
   kind='int'if raw.startswith('INTEGER')else'float'
   for v in split_args(raw.split('::',1)[1]):
    v=v.split('(')[0].strip();decls[v]=(kind,'INOUT'in raw or 'INTENT(OUT)'in raw.replace(' ',''))
    if v not in args and v not in ('R','X'):locals.append(kind+' '+v.lower()+';')
   continue
  if raw.startswith('IMPLICIT'):continue
  if re.match(r'R\(X\)\s*=',raw):continue # unused native statement function
  if raw.endswith('&'):pending+=raw[:-1];continue
  raw=pending+raw.lstrip('&');pending=''
  if raw in ('END IF','ENDIF','END DO','ENDDO'):body.append('}')
  elif raw=='ELSE':body.append('} else {')
  elif re.match(r'(ELSE\s*)?IF\s*\(',raw):
   a=raw.index('(');i=a+1;depth=1
   while depth:depth+=(raw[i]=='(')-(raw[i]==')');i+=1
   tail=raw[i:].strip();body.append(('} else 'if raw.startswith('ELSE')else'')+'if('+expr(raw[a+1:i-1])+')'+(' {'if tail=='THEN'else' '+stmt(tail)))
  elif raw.startswith('DO '):
   v,b=raw[3:].split('=',1);a,last=split_args(b);body.append('for('+v.lower()+'=(int)('+expr(a)+');'+v.lower()+'<=(int)('+expr(last)+');++'+v.lower()+') {')
  else:body.append(stmt(raw))
 for a in args:
  if a=='PARAMETERS':sig.append('const Params&p')
  elif a in ARRAYS and not(name=='CO2FLUX_CROP' and a=='STC'):sig.append('const float*'+a.lower())
  else:
   kind,ref=decls[a];sig.append(kind+('&'if ref else' ')+a.lower())
 return '// Complete native '+name+'; module_sf_noahmplsm.F:'+str(source[:m.start()].count('\n')+1)+'\n__device__ void '+name.lower()+'('+','.join(sig)+') {\n'+'\n'.join('    '+v for v in locals+body)+'\n}\n'
def generate(source):
 assert hashlib.sha256(source).hexdigest()==LSM_SHA256
 struct='struct Params {\n'+''.join('float '+s.lower()+';\n'for s in SCALARS)+''.join('float '+s.lower()+'[8];\n'for s in STAGES)+'float smcmax[4]; int nroot,iswater,isbarren,isice;bool urban_flag;\n};\n'
 return '// WRF v4 public domain; licenses/LICENSE-WRF-public-domain.txt.\nnamespace noahmp_crop {\nusing noahmp_carbon::mn;using noahmp_carbon::mx;\n'+struct+'\n'.join(routine(source.decode(),n)for n in NAMES)+'}\n'
if __name__=='__main__':
 import sys
 from pathlib import Path
 Path(sys.argv[2]).write_text(generate(Path(sys.argv[1]).read_bytes()))
