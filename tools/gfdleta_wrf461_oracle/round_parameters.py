"""Freeze WRF REAL parameter words using its compiler, before C lowering."""
from pathlib import Path
import re,subprocess,struct,json,sys

def fixed_lines(s):
 out=[];pending=''
 for line in s.splitlines():
  if line[5:6] not in ('',' '):pending+=line[6:]
  else:
   if pending:out.append(pending)
   pending=line[6:]
 if pending:out.append(pending)
 return out

folder=Path(sys.argv[1]);allwords={}
for file in sorted(folder.glob('*.f')):
 lines=fixed_lines(file.read_text());types={};defs=[]
 for s in lines:
  m=re.match(r'(REAL|INTEGER|LOGICAL|DOUBLE PRECISION)\s+(.+)',s,re.I)
  if m:
   for name in re.findall(r'\b([A-Z][A-Z0-9_]*)\b',m[2],re.I):types[name.upper()]=m[1].upper()
  m=re.match(r'PARAMETER\((\w+)=(.*)\)$',s,re.I)
  if m:defs.append((m[1].upper(),m[2]))
 if not defs:continue
 params={k for k,v in defs};src=['program constants','use iso_fortran_env,only:int32','implicit none']
 for k,v in defs:
  src += [types[k]+' :: '+k,'PARAMETER('+k+'='+v+')']
 for k,v in defs:
  if types[k]=='REAL':src.append(f"write(*,'(A,\",\",Z8.8)')'{k}',transfer({k},0_int32)")
 src.append('end program');p=folder/(file.stem+'-constants.F90');p.write_text('\n'.join(src)+'\n')
 exe=folder/(file.stem+'-constants')
 r=subprocess.run(['gfortran','-O0','-ffree-line-length-none',str(p),'-o',str(exe)],capture_output=True,text=True)
 if r.returncode:raise RuntimeError(file.name+'\n'+r.stderr)
 values={}
 for line in subprocess.check_output([str(exe)],text=True).splitlines():
  key,word=line.split(',');values[key]=word
 allwords[file.stem]=values
 result=[]
 for s in lines:
  m=re.match(r'PARAMETER\((\w+)=(.*)\)$',s,re.I)
  if m and m[1].upper() in values:
   value=struct.unpack('>f',bytes.fromhex(values[m[1].upper()]))[0]
   text=format(value,'.17g').replace('e','E')
   if '.' not in text and 'E' not in text:text+='.'
   s='PARAMETER('+m[1]+'='+text+')'
  # Emit the original labels from source separately; only PARAMETER lines
  # change, so replace their physical records in the original file below.
  result.append(s)
 original=file.read_text().splitlines();output=[];i=0
 for physical in original:
  if physical[5:6] in ('',' '):
   logical=lines[i];replaced=result[i];i+=1
   replace=logical!=replaced
   if replace:
    while len(replaced)>66:output.append('      '+replaced[:66]);replaced=replaced[66:];break
    output.append(('     &' if len(output) and output[-1].startswith('      PARAMETER') and len(output[-1])==72 else '      ')+replaced)
   else:output.append(physical)
  elif not replace:output.append(physical)
 file.write_text('\n'.join(output)+'\n')
(folder/'parameter-words.json').write_text(json.dumps(allwords,indent=2)+'\n')
print('parameter words',sum(map(len,allwords.values())))
