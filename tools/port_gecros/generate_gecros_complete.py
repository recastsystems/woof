"""Generate every routine of the pinned WRF GECROS module, independently per cell."""
from pathlib import Path
import ast, re, hashlib, json, argparse
import numpy as np
R=Path(__file__).resolve().parent
parser=argparse.ArgumentParser(__doc__)
parser.add_argument('source',type=Path)
parser.add_argument('--output',type=Path,default=R.parents[1]/'woof/core/kernels/noahmp_gecros.cuh')
parser.add_argument('--audit',type=Path,default=R/'source-audit.json')
cli=parser.parse_args()
SOURCE=cli.source
SHA='ad2864562e95678a25276df82ef96395cca61c3a1bd0ab48ddfb8402902cf2f6'
assert hashlib.sha256(SOURCE.read_bytes()).hexdigest()==SHA
source=SOURCE.read_text()
def split(s):
 out=[];start=0;depth=0
 for i,c in enumerate(s):
  depth+=(c=='(')-(c==')')
  if c==',' and depth==0:out.append(s[start:i].strip());start=i+1
 out.append(s[start:].strip());return out
def lines(t):
 result=[];pending=''
 for raw in t.splitlines():
  raw=raw.split('!',1)[0].strip().upper()
  if not raw:continue
  if raw.endswith('&'):pending+=raw[:-1];continue
  result.append(pending+raw.lstrip('&'));pending=''
 assert not pending
 return result
def tree(s):
 for a,b in (('/=','!='),('.GT.','>'),('.GE.','>='),('.LT.','<'),('.LE.','<='),('.EQ.','=='),('.NE.','!='),('.AND.',' and '),('.OR.',' or '),('.NOT.',' not ')):s=s.replace(a,b)
 return ast.parse(s.strip(),mode='eval').body
const={};globals=[]
for line in lines(source.split('CONTAINS')[0]):
 if line.startswith('REAL, PARAMETER'):
  name,value=line.split('::')[1].split('=');const[name.strip()]=value.strip()
 elif line.startswith('REAL ::'):globals.append(line.split('::')[1].strip())
def constant(n):
 if isinstance(n,ast.Constant):return np.float32(n.value)
 if isinstance(n,ast.Name):return values[n.id]
 if isinstance(n,ast.UnaryOp):return -constant(n.operand)
 if isinstance(n,ast.BinOp):
  a,b=constant(n.left),constant(n.right)
  return np.float32({ast.Add:lambda:a+b,ast.Sub:lambda:a-b,ast.Mult:lambda:a*b,ast.Div:lambda:a/b,ast.Pow:lambda:a**b}[type(n.op)]())
 raise ValueError(ast.dump(n))
values={}
for k,v in const.items():values[k]=constant(tree(v))
names={};routines=[]
for m in re.finditer(r'^\s*(SUBROUTINE|FUNCTION)\s+(\w+)\s*\(',source,re.M|re.I):
 kind,name=m.group(1).upper(),m.group(2).upper()
 end=re.search(r'END '+kind+r'\s+'+name,source[m.end():],re.I)
 body=lines(source[m.start():m.end()+end.start()]);header=body.pop(0);args=split(header[header.index('(')+1:header.rindex(')')]);decl={};arrays={};statements=[]
 for line in body:
  match=re.match(r'^(REAL|INTEGER|LOGICAL)\b(.*)',line)
  if match:
   kinddecl,tail=match.groups()
   if '::' in tail:attrs,tail=tail.split('::')
   else:attrs=''
   dims=re.search(r'DIMENSION\((.*?)\)',attrs)
   for v in split(tail):
    if '('in v:v,shape=v.split('(',1);arrays[v.strip()]=shape[:-1]
    if dims:arrays[v.strip()]=dims.group(1)
    v=v.split('=')[0].strip();decl[v]='int'if kinddecl=='INTEGER'else'bool'if kinddecl=='LOGICAL'else'float'
   continue
  if line.startswith(('IMPLICIT','CHARACTER')):continue
  statements.append(line)
 assigned=set(re.findall(r'(?<![<>=!])(\b\w+)\s*(?:\([^=]*?\))?\s*=(?!=)','\n'.join(statements)))
 routines.append(dict(name=name,args=args,kind=kind,decl=decl,arrays=arrays,statements=statements,assigned=assigned))
 names[name]=routines[-1]
# Discover all arguments changed through helper calls, including implicit REAL ABI.
changed=True
while changed:
 changed=False
 for r in routines:
  for line in r['statements']:
   if not line.startswith('CALL '):continue
   n,tail=line[5:].split('(',1);callee=names[n.strip()]
   for formal,actual in zip(callee['args'],split(tail[:-1])):
    if formal in callee['assigned'] and re.fullmatch(r'\w+',actual) and actual not in r['assigned']:
     r['assigned'].add(actual);changed=True
def expression(s,r):
 def e(n):
  if isinstance(n,ast.Name):
   name=n.id
   if name==r['name'] and r['kind']=='FUNCTION':return 'result'
   if name in globals and name not in r['args'] and name not in r['decl']:return 'ctx.'+name.lower()
   return name.lower()
  if isinstance(n,ast.Constant):return str(n.value) if isinstance(n.value,int) else repr(n.value)+'f'
  if isinstance(n,ast.UnaryOp):return ('-'if isinstance(n.op,ast.USub)else'!'if isinstance(n.op,ast.Not)else'+')+'('+e(n.operand)+')'
  if isinstance(n,ast.BinOp):
   a,b=e(n.left),e(n.right)
   if isinstance(n.op,ast.Pow) and isinstance(n.right,ast.Constant):
    v=n.right.value
    if v==2:return 'FM('+a+','+a+')'
    if v==4:return 'p4f('+a+')'
    if v==0.5:return 'FSQ('+a+')'
   return {ast.Add:'FA',ast.Sub:'FS',ast.Mult:'FM',ast.Div:'FD',ast.Pow:'power'}[type(n.op)]+'('+a+','+b+')'
  if isinstance(n,ast.Compare):return '('+e(n.left)+{ast.Gt:'>',ast.GtE:'>=',ast.Lt:'<',ast.LtE:'<=',ast.Eq:'==',ast.NotEq:'!='}[type(n.ops[0])]+e(n.comparators[0])+')'
  if isinstance(n,ast.BoolOp):return '('+('&&'if isinstance(n.op,ast.And)else'||').join(e(v)for v in n.values)+')'
  if isinstance(n,ast.Call):
   name=n.func.id;a=[e(v)for v in n.args]
   if name in r['arrays']:return name.lower()+'[(int)('+a[0]+')-1]'
   if name in names:return name.lower()+'(ctx,'+','.join(a)+')'
   if name in ('MIN','MAX'):
    v=a[0]
    for b in a[1:]:v=('fmnf'if name=='MIN'else'fmxf')+'('+v+','+b+')'
    return v
   return {'EXP':'glibc_expf','LOG':'glibc_logf','SQRT':'FSQ','SIN':'glibc_sinf','COS':'glibc_cosf','ASIN':'glibc_asinf','ACOS':'glibc_acosf','TAN':'glibc_tanf','ABS':'fabsf','INT':'(int)','FLOAT':'(float)','AINT':'truncf','NINT':'roundf'}[name]+'('+','.join(a)+')'
  raise ValueError(ast.dump(n))
 return e(tree(s))
def signature(r):
 sig=['Context&ctx']
 for a in r['args']:
  kind=r['decl'].get(a,'float')
  sig.append((kind+'*'if a in r['arrays']else kind+('&'if a in r['assigned']else' '))+a.lower())
 return '__device__ '+('float'if r['kind']=='FUNCTION'else'void')+' '+r['name'].lower()+'('+','.join(sig)+')'
def routine(r):
 ex=lambda s:expression(s,r)
 def stmt(s):
  if s=='RETURN':return 'return result;'if r['kind']=='FUNCTION'else'return;'
  if s.startswith('STOP '):return '{ asm("trap;"); return; }'
  if s.startswith('CALL '):
   n,tail=s[5:].split('(',1);return n.strip().lower()+'(ctx,'+','.join(ex(a)for a in split(tail[:-1]))+');'
  a,b=s.split('=',1);return ex(a)+'='+ex(b)+';'
 body=[];identifiers=set()
 for line in r['statements']:
  identifiers.update(re.findall(r'\b[A-Z]\w*\b',re.sub(r"'[^']*'",'',line)))
  if line in ('ENDIF','END IF'):body.append('}')
  elif line=='ELSE':body.append('} else {')
  elif re.match(r'(ELSE\s*)?IF\s*\(',line):
   a=line.index('(');i=a+1;depth=1
   while depth:depth+=(line[i]=='(')-(line[i]==')');i+=1
   tail=line[i:].strip();body.append(('} else 'if line.startswith('ELSE')else'')+'if('+ex(line[a+1:i-1])+')'+(' {'if tail=='THEN'else' '+stmt(tail)))
  else:body.append(stmt(line))
 local=identifiers-set(r['args'])-set(const)-set(globals)-set(names)-{'CALL','IF','THEN','ELSE','ENDIF','END','RETURN','AND','OR','GT','GE','LT','LE','EQ','NE','STOP','E','EXP','LOG','SQRT','SIN','COS','ASIN','ACOS','TAN','ABS','INT','FLOAT','AINT','NINT','MIN','MAX'}
 local|=set(r['decl'])-set(r['args'])-set(names)
 decls=[]
 if r['kind']=='FUNCTION':decls.append('float result;')
 for name in sorted(local):decls.append(r['decl'].get(name,'float')+' '+name.lower()+('['+r['arrays'][name].split(':')[-1]+']'if name in r['arrays']else'')+';')
 return signature(r)+' {\n'+'\n'.join('    '+s for s in decls+body)+'\n}\n'
output='// Complete WRF v4.7.1 GECROS, public domain under licenses/LICENSE-WRF-public-domain.txt.\n// Source SHA256 '+SHA+'; generated by the pinned offline oracle.\nnamespace noahmp_gecros {\n'
output+='__device__ float power(float x,float y){if(y==0.0f)return 1.0f;if(x==0.0f)return y>0.0f?0.0f:__int_as_float(0x7f800000);if(x<0.0f && truncf(y)==y){float v=glibc_powf(-x,y);return fmodf(fabsf(y),2.0f)==1.0f?-v:v;}return glibc_powf(x,y); }\n'
output+='struct Context {'+''.join('float '+s.lower()+';'for s in globals)+'};\n'
output+='\n'.join('static constexpr float '+k.lower()+'='+float(v).hex()+'f;'for k,v in values.items())+'\n'
output+='\n'.join(signature(r)+';'for r in routines)+'\n'
output+='\n'.join(routine(r)for r in routines)+'}\n'
cli.output.write_text(output)
cli.audit.write_text(json.dumps(dict(url='https://raw.githubusercontent.com/wrf-model/WRF/f52c197ed39d12e087d02c50f412d90d418f6186/phys/module_sf_gecros.F',sha256=SHA,licence='WRF public-domain distribution; original source header inspected, attribution retained, no separate restrictive notice',routines=[r['name']for r in routines],module_context=globals,constants=len(const)),indent=2)+'\n')
print('complete routines',len(routines),'lines',output.count('\n'))
