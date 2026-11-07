"""Lower GFDL initialization with device buffers replacing file operations."""
from pathlib import Path
import re,json,subprocess,importlib.util,sys
path=Path(__file__).with_name('lower_column.py');spec=importlib.util.spec_from_file_location('lower',path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
wrf=Path(sys.argv[1]);out=Path(sys.argv[2]);f2c=Path(sys.argv[3]);out.mkdir(parents=True,exist_ok=True)
cpp=subprocess.run(['cpp','-P','-traditional-cpp',str(wrf/'phys/module_ra_gfdleta.F')],capture_output=True,text=True,check=True).stdout
units=m.routines(m.logical_lines(cpp))
mp=m.routines(m.logical_lines((wrf/'phys/module_mp_etanew.F').read_text()))
units.update({k:mp[k] for k in ['GPVS','FPVSX']})
m.PARAM.update({'RV':('REAL','461.5'),'XMIN':('REAL','180.'),'XMAX':('REAL','330.')})
SELECT=['ANTEMP','COEINT','CO2INS','CO2INT','CO2IN1','CO2PTZ','PATH','QINTRP','QUADSR','SIGP','SINTR2','CO2O3','O3CLIM','GPVS','FPVSX']
STATE={'PA':('REAL','109'),'TRANSA':('REAL','109,109'),'XA':('REAL','109'),'CA':('REAL','109'),'ETA':('REAL','109'),'SEXPV':('REAL','109'),'CORE':('REAL',None),'UEXP':('REAL',None),'SEXP':('REAL',None),
 'P1':('REAL',None),'P2':('REAL',None),'TRNSLO':('REAL',None),'IA':('INTEGER',None),'JA':('INTEGER',None),'N':('INTEGER',None),'STATUS':('INTEGER',None),'RAW':('REAL','109,109,3,3')}
base_globals=dict(m.GLOBALS);base_clock=dict(m.CLOCK)
m.GLOBALS={'GDX_'+k:v for k,v in base_globals.items()};m.CLOCK={'GDC_'+k:v for k,v in base_clock.items()}
old_bounds=m.bounds
def bounds(dim):
 dim=old_bounds(dim)
 replacements={'NLP1':'GDMAX+1','NLP2':'GDMAX+2','NLEVLS':'GDMAX','NLV':'GDMAX','NLP1V':'GDMAX+1','NLP2V':'GDMAX+2','LP1':'GDMAX+1','LP2':'GDMAX+2','KD':'GDMAX','KP':'GDMAX+1','KP2':'GDMAX+2','KM':'GDMAX-1','NL':'GDMAX','NLP':'GDMAX+1','L':'GDMAX'}
 return re.sub(r'\b('+ '|'.join(replacements)+r')\b',lambda z:'('+replacements[z[0]]+')',dim)
m.bounds=bounds

def declared(lines):
 names=set()
 for _,s in lines:
  d=m.declaration(s)
  if d:names |= d[1]
  elif re.match(r'^(REAL|INTEGER|LOGICAL|DOUBLE PRECISION|DIMENSION)\b',s,re.I):
   v=re.sub(r'^(REAL|INTEGER|LOGICAL|DOUBLE PRECISION|DIMENSION)\s*','',s,flags=re.I)
   for item in m.split_top(v):
    match=re.match(r'(\w+)',item)
    if match:names.add(match[1].upper())
 header=lines[0][1]
 if '(' in header:names|={x.strip().upper() for x in m.split_top(header.split('(',1)[1].rsplit(')',1)[0])}
 return names

audit={}
for name in SELECT:
 lines=units[name];local=declared(lines);has_input=any('COMMON/INPUT/' in s.replace(' ','').upper() for _,s in lines)
 if name=='CO2O3':
  # Allocation and file-unit discovery are replaced by the run's buffers.
  declarations=[x for x in lines if re.match(r'^(DIMENSION|INTEGER|REAL|LOGICAL|CHARACTER)\b',x[1],re.I)]
  begin=next(i for i,x in enumerate(lines) if x[1].upper().startswith('LREAD = 0'))
  finish=next(i for i,x in enumerate(lines) if x[1].strip().upper()=='RETURN')
  lines=[lines[0],*declarations,('', 'INTEGER ICO2TP'),('','INTEGER ETARAD_UNIT61,ETARAD_UNIT62,ETARAD_UNIT63'),('','ETARAD_UNIT61=1'),('','ETARAD_UNIT62=2'),('','ETARAD_UNIT63=3'),*lines[begin:finish],('','RETURN'),lines[-1]]
  # The end-of-routine CLOSE branch is orchestration, not numerical code.
  lines=[x for x in lines if not re.match(r'^(CLOSE|IF\s*\(\s*wrf_dm_on_monitor|ENDIF)',x[1],re.I)]
  # Duplicate scalar declarations would obscure the original implicit types.
  lines=[x for x in lines if x[1] not in ['INTEGER ICO2TP','INTEGER ETARAD_UNIT61,ETARAD_UNIT62,ETARAD_UNIT63']]
  local=declared(lines)
 transformed=[]
 for label,s in lines:
  if name=='GPVS' and any(k in s.upper() for k in ['TBPVS0','C1XPVS0','C2XPVS0']):continue
  if re.match(r'^COMMON\s*/INPUT/',s,re.I):continue
  if re.match(r'^(REWIND|CLOSE|OPEN|INQUIRE|ALLOCATE|DEALLOCATE)',s,re.I):continue
  if re.match(r'^CALL\s+wrf_dm_bcast',s,re.I):continue
  if 'READ (ITAPE' in s.upper() or 'READ(ITAPE' in s.upper():
   s='CALL GD_LOAD_TRANSMISSION(ITAPE,KKK)'
  elif re.match(r'^READ\s*\(',s,re.I):continue
  if 'wrf_dm_on_monitor' in s and s.upper().startswith('LOGICAL'):continue
  s=re.sub(r'wrf_dm_on_monitor\(\)','.TRUE.',s,flags=re.I)
  if re.match(r'^CALL\s+wrf_error_fatal',s,re.I):s='GDZ_STATUS=1'
  mapping={**{k:'GDX_'+k for k in base_globals if k not in local},**{k:'GDC_'+k for k in base_clock if k not in local},**{k:'GDZ_'+k for k in STATE if k not in local and k not in ['P1','P2','TRNSLO','IA','JA','N']}}
  if has_input:mapping.update({k:'GDZ_'+k for k in ['P1','P2','TRNSLO','IA','JA','N']})
  for key,value in sorted(mapping.items(),key=lambda x:-len(x[0])):s=re.sub(r'\b'+key+r'\b',value,s,flags=re.I)
  transformed.append((label,s))
 # m.lower supplies the common table and clock blocks before executable code.
 text=m.lower(name,transformed)
 state=[('GDZ_'+k,v) for k,v in STATE.items()]
 decl=[]
 for k,(kind,shape) in state:decl.append(kind+' '+k+('('+shape+')' if shape else ''))
 decl.append('COMMON /GD_INIT_STATE/ '+','.join(k for k,v in state))
 insertion='\n'.join(line for stmt in decl for line in m.emit_fixed('',stmt))+'\n'
 pos=text.index('      INTEGER GDMAX');text=text[:pos]+insertion+text[pos:]
 (out/(name.lower()+'.f')).write_text(text);audit[name]={'source_statements':len(units[name])}

loader='''      SUBROUTINE GD_LOAD_TRANSMISSION(ITAPE,ITEMP)
      INTEGER ITAPE,ITEMP,I,J
'''+''.join(line+'\n' for stmt in m.common_specs(m.GLOBALS,'GD_TABLES')+m.common_specs(m.CLOCK,'GD_CLOCK') for line in m.emit_fixed('',stmt))+insertion+'''      DO J=1,109
      DO I=1,109
      GDZ_TRANSA(I,J)=GDZ_RAW(I,J,ITEMP,ITAPE)
      ENDDO
      ENDDO
      END
'''
(out/'gd_load_transmission.f').write_text(loader)
(out/'init-audit.json').write_text(json.dumps(audit,indent=2)+'\n')
r=subprocess.run([str(f2c),'-R','-r','-a','-kr','-A',*[str(x) for x in sorted(out.glob('*.f'))]],cwd=out,capture_output=True,text=True)
(out/'f2c.log').write_text(r.stdout+r.stderr)
print('return',r.returncode)
for line in (r.stdout+r.stderr).splitlines():
 if 'Error' in line or 'error' in line:print(line)
raise SystemExit(r.returncode)
