"""Assemble the clean column transcription with WOOF runtime bindings."""
from pathlib import Path
import argparse,json,re,importlib.util

def paren_end(s,start):
 depth=0
 for i in range(start,len(s)):
  if s[i]=='(':depth+=1
  elif s[i]==')':
   depth-=1
   if depth==0:return i
 raise ValueError('unbalanced generated C')

def add_context(s,names):
 pattern=re.compile(r'\b('+'|'.join(re.escape(n) for n in names)+r')\s*\(')
 matches=list(pattern.finditer(s));edits=[]
 for m in matches:
  start=s.index('(',m.start());end=paren_end(s,start)
  args=s[start+1:end]
  # Prototypes and definitions have typed first arguments; calls do not.
  typed=bool(re.match(r'\s*(?:real|integer|logical|doublereal|void)\b',args))
  tail='GDTable *gt, GDClock *clock, GDScratch *work' if typed else 'gt, clock, work'
  if args.strip() in ('void',''):edits.append((start+1,end,tail))
  else:edits.append((end,end,', '+tail))
 for a,b,text in sorted(edits,reverse=True):s=s[:a]+text+s[b:]
 return s

def main():
 p=argparse.ArgumentParser();p.add_argument('generated',type=Path);p.add_argument('kernel',type=Path);a=p.parse_args()
 names=[];chunks=[];structs=None;meta={};scratch=[]
 for file in sorted(a.generated.glob('*.c')):
  s=file.read_text();s=re.sub(r'/\*.*?\*/','',s,flags=re.S)
  s=re.sub(r'#include[^\n]*','',s)
  blocks=re.findall(r'struct\s*\{(.*?)\}\s*(gd_tables__|gd_clock__);',s,re.S)
  if len(blocks)!=2:raise ValueError('missing common declarations '+file.name)
  this='typedef struct {'+blocks[0][0]+'} GDTable;\ntypedef struct {'+blocks[1][0]+'} GDClock;\n'
  if structs is None:structs=this
  elif this!=structs:raise ValueError('common layouts differ '+file.name)
  s=re.sub(r'struct\s*\{.*?\}\s*(?:gd_tables__|gd_clock__);','',s,flags=re.S)
  s=re.sub(r'#define gd_(?:tables|clock)__1[^\n]*','',s)
  # Each file's DATA constants retain their names within that one unit.
  unit=file.stem
  globals=re.findall(r'^static\s+(?:real|integer|doublereal|logical)\s+(c_[A-Za-z_0-9]+)',s,re.M)
  for n in set(globals):s=re.sub(r'\b'+re.escape(n)+r'\b',unit+'_'+n,s)
  s=re.sub(r'^static\s+(real|integer|doublereal|logical)\s+',r'GD_DEVICE \1 ',s,flags=re.M)
  # Function-scoped immutable DATA variables are per-thread constants.
  s=re.sub(r'(^[ \t]+)static\s+(real|integer|doublereal|logical)\s+',r'\1\2 ',s,flags=re.M)
  s=re.sub(r'(^[ \t]+)static struct',r'\1struct',s,flags=re.M)
  # Remove built-in declarations: overloads in our support header select
  # float32 or float64 from the actual expression type.
  s=re.sub(r'^[ \t]*(?:double|integer|real)\s+(?:exp|sqrt|log|log10|sin|cos|asin|acos|atan|tan|pow_dd|r_sign|r_mod|r_nint|i_nint|pow_ri|r_lg10|r_int)\s*\([^;]*;','',s,flags=re.M)
  definitions=re.findall(r'^\s*(?:int|real|doublereal|integer)\s+(\w+_)\s*\(',s,re.M)
  names.extend(definitions)
  meta[unit]={'functions':definitions,'c_sha256':__import__('hashlib').sha256(file.read_bytes()).hexdigest()}
  # Per-column mutable work arrays belong to the explicit device workspace.
  # This avoids CUDA's per-thread local-memory ceiling without changing any
  # statement or using a different numerical arm.
  edits=[]
  for fn in definitions:
   fm=re.search(r'\b'+re.escape(fn)+r'\s*\(',s);op=s.index('{',paren_end(s,s.index('(',fm.start())))
   depth=1;end=op+1
   while depth:
    if s[end]=='{':depth+=1
    elif s[end]=='}':depth-=1
    end+=1
   body=s[op+1:end-1];levels=[];d=0
   for ch in body:
    levels.append(d)
    if ch=='{':d+=1
    elif ch=='}':d-=1
   for dm in re.finditer(r'^[ \t]+(real|integer|doublereal|logical)\s+([^;]+);',body,re.M):
    if levels[dm.start()] or '=' in dm[2] or '(' in dm[2]:continue
    entities=dm[2].split(',');repl=[];changed=False
    for entity in entities:
     am=re.fullmatch(r'\s*(\w+)\[(\d+)\]\s*',entity)
     if am:
      var,count=am[1],int(am[2])
      if fn=='clo89_' and var=='cldipt':count//=64
      field=fn+var;scratch.append((dm[1],field,count));repl.append(dm[1]+' *'+var+'=work->'+field+';');changed=True
     else:repl.append(dm[1]+' '+entity.strip()+';')
    if changed:edits.append((op+1+dm.start(),op+1+dm.end(),'\n    '+'\n    '.join(repl)))
  for lo,hi,text in sorted(edits,reverse=True):s=s[:lo]+text+s[hi:]
  chunks.append(s)
 alltext='\n'.join(chunks)
 alltext=add_context(alltext,names)
 alltext=re.sub(r'(^|\n)(\s*)((?:int|real|doublereal|integer)\s+\w+_\s*\()',r'\1\2GD_DEVICE \3',alltext)
 alltext=alltext.replace('extern  int ','extern GD_DEVICE int ').replace('extern real ','extern GD_DEVICE real ').replace('extern  real ','extern GD_DEVICE real ')
 # Calls of built-in arithmetic now bind to WOOF routines.
 for name in ['exp','sqrt','log','log10','sin','cos','asin','acos','tan','atan']:
  alltext=re.sub(r'\b'+name+r'\s*\(', 'gd_'+name+'(',alltext)
 header='''// Clean WRF 4.6.1 column transcription. No third-party runtime is linked.
#ifndef GD_DEVICE
#define GD_DEVICE __device__
#endif
typedef float real;
typedef double doublereal;
typedef int integer;
typedef int logical;
typedef int ftnlen;
#define TRUE_ 1
#define FALSE_ 0
'''+structs+'''\n#define gd_tables__1 (*gt)
#define gd_clock__1 (*clock)
'''
 scratch_decl='typedef struct {\n'+''.join('    '+kind+' '+field+'['+str(count)+'];\n' for kind,field,count in scratch)+'} GDScratch;\n'
 header=header.replace('#define gd_tables__1',scratch_decl+'#define gd_tables__1')
 a.kernel.write_text('\n'.join(line.rstrip() for line in (header+alltext).splitlines())+'\n')
 offset=0
 for kind,field,count in scratch:
  width=8 if kind=='doublereal' else 4;offset=(offset+width-1)//width*width;offset+=count*width
 offset=(offset+7)//8*8
 (a.generated/'scratch-layout.json').write_text(json.dumps({'bytes':offset,'fields':scratch},indent=2)+'\n')
 (a.generated/'assembly-audit.json').write_text(json.dumps(meta,indent=2)+'\n')
 print('functions',names,'bytes',a.kernel.stat().st_size)

if __name__=='__main__':main()
