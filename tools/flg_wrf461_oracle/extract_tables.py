'''Generate a compiler-based dump of WRF FLG coefficient modules.'''
from pathlib import Path
import re
import sys

def statements(source):
    pending=''
    for line in source.splitlines():
        line=line.split('!')[0].strip()
        if not line:
            continue
        pending+=' '+line.lstrip('&').strip()
        if pending.endswith('&'):
            pending=pending[:-1]
            continue
        yield pending.strip()
        pending=''
    if pending:
        raise ValueError('unterminated continuation')

def split_items(text):
    depth=0; start=0
    for idx,ch in enumerate(text):
        if ch=='(': depth+=1
        elif ch==')': depth-=1
        elif ch==',' and depth==0:
            yield text[start:idx].strip(); start=idx+1
    yield text[start:].strip()

def generate(source):
    modules={}; module=None
    for line in statements(source):
        match=re.fullmatch(r'module\s+(\w+)',line,re.I)
        if match:
            module=match[1].lower()
            modules[module]={'decls':{},'data':set()}
        elif re.match(r'end\s*module',line,re.I):
            module=None
        elif module and module not in ('para_file','control_para','module_ra_flg'):
            decl=re.match(r'real\b(.*)',line,re.I)
            if decl and 'parameter' not in line.lower():
                spec=decl[1]; attrs=''
                if '::' in spec:
                    attrs,spec=spec.split('::',1)
                for item in split_items(spec):
                    variable=re.match(r'(\w+)\s*(\([^=]*\))?',item)
                    if not variable: raise ValueError(f'unparsed declaration: {line}')
                    name=variable[1].lower()
                    array=variable[2] is not None or 'dimension' in attrs.lower()
                    if 'private' not in attrs.lower():
                        modules[module]['decls'][name]=array
            data=re.match(r'data\s+(.*?)\s*/',line,re.I)
            if data:
                modules[module]['data'].update(re.findall(r'\b[a-z]\w*\b',data[1].lower()))
    uses=[]; calls=[]
    for mod,meta in modules.items():
        for name,array in meta['decls'].items():
            if name not in meta['data'] and (mod,name)!=('ice6','dps'): continue
            alias=mod+'__'+name
            uses.append(f' use {mod}, only: {alias} => {name}')
            values=f'reshape({alias},[size({alias})])' if array else f'[{alias}]'
            dims=f'shape({alias})' if array else '[integer::]'
            calls.append(f" call dump('{mod}.{name}',{values},{dims})")
    if not uses: raise ValueError('no FLG tables found')
    local_data=[]; routine=None
    wanted={'planck':{'ve','nd'},'qks':{'stanp'},'qki':{'stanp'},'qkio3':{'stanp'},'gascon':{'vv'},'rad_all':{'kg','kg1','kg2'},'qftisf':{'ug','wg','ugwg'}}
    declarations=[]; extra_calls=[]
    for line in statements(source):
        match=re.match(r'subroutine\s+(\w+)\s*\(',line,re.I)
        if match: routine=match[1].lower()
        elif re.match(r'end\s+subroutine',line,re.I): routine=None
        elif routine in wanted:
            data=re.match(r'data\s+(\w+)\s*/',line,re.I)
            initialized=re.match(r'(real|integer)\s*(?:::)?\s*(\w+)\s*\(([^)]+)\)\s*=\s*(.*)',line,re.I)
            if data and data[1].lower() in wanted[routine]:
                name=data[1].lower(); alias=routine+'__'+name
                declaration=('integer' if name.startswith('kg') or name=='nd' else 'real')
                extent='2' if routine=='qftisf' else '18' if name.startswith('kg') else '12'
                declarations.append(declaration+' :: '+alias+'('+extent+')')
                local_data.append(re.sub(r'^(data\s+)'+name,lambda m:m[1]+alias,line,flags=re.I))
                value='real('+alias+')' if declaration=='integer' else alias
                extra_calls.append(f" call dump('{routine}.{name}',{value},[{extent}])")
            elif initialized and initialized[2].lower() in wanted[routine]:
                name=initialized[2].lower(); alias=routine+'__'+name
                declarations.append(initialized[1].lower()+' :: '+alias+'('+initialized[3]+')')
                local_data.append('data '+alias+' /'+initialized[4].strip()[2:-2]+'/')
                extra_calls.append(f" call dump('{routine}.{name}',{alias},[{initialized[3]}])")
    if len(extra_calls)!=12: raise ValueError('FLG routine-local coefficient inventory was not found')
    calls.extend(extra_calls)
    code='program extract_flg_tables\n use para_file, only: mbir\n'+ '\n'.join(uses)+'''
 implicit none
 integer :: ub,um,offset
'''+ '\n'.join(declarations)+'''
'''+ '\n'.join(local_data)+'''
 open(newunit=ub,file='tables.bin',access='stream',form='unformatted',status='replace')
 open(newunit=um,file='tables-index.csv',status='replace')
 write(um,'(A)') 'name,offset,count,shape'
 offset=0
'''+ '\n'.join(calls)+'''
 close(ub); close(um)
contains
 subroutine dump(name,values,dims)
  character(*),intent(in) :: name
  real,intent(in) :: values(:)
  integer,intent(in) :: dims(:)
  integer :: j
  if(storage_size(values)/=32) error stop 'FLG tables must be float32'
  write(ub) values
  write(um,'(A,A,I0,A,I0,A)',advance='no') name,',',offset,',',size(values),','
  do j=1,size(dims)
   if(j>1) write(um,'(A)',advance='no') 'x'
   write(um,'(I0)',advance='no') dims(j)
  enddo
  write(um,'(A)') ''
  offset=offset+4*size(values)
 end subroutine
end program
'''
    return code

if __name__=='__main__':
    Path(sys.argv[2]).write_text(generate(Path(sys.argv[1]).read_text()),encoding='utf-8')
